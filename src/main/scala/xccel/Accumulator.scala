package xccel

import chisel3._
import chisel3.util._

case class AccumulatorConfig(
  cu: ComputeUnitConfig,
  numBanks: Int,
  wordsPerBank: Int,
  depth: Int,
  readLatency: Int,
  writeLatency: Int,
  // Independent row banks (rows [k*depth/rowBanks, (k+1)*depth/rowBanks) in
  // bank k), each its own OneR1W SRAM: an accumulating beat and a drain may
  // use the read ports of two different banks in the same cycle.
  rowBanks: Int = 1,
  // largest LaneSwitch fanout built (see LaneSwitchConfig): None = up to
  // broadcast, Some(0) = unicast only
  maxFanout: Option[Int] = None,
  // independent row addresses, one per run of numBanks / addrGroups
  // consecutive column banks (see AddrGroups); 1 = all banks share one
  // address. Each group picks its row bank from its own address.
  addrGroups: Int = 1,
) {
  require(depth > 0, "depth must be positive")
  AddrGroups.validate(numBanks, addrGroups)
  require(rowBanks > 0 && depth % rowBanks == 0, "rowBanks must divide depth")
  require(rowBanks == 1 || isPow2(depth / rowBanks), "rows per bank must be a power of two with rowBanks > 1")
  require(wordsPerBank > 0, "wordsPerBank must be positive")
  require(
    numBanks * wordsPerBank >= cu.numOutputs,
    s"numBanks*wordsPerBank (${numBanks * wordsPerBank}) must hold a resultIn beat (cu.numOutputs = ${cu.numOutputs})"
  )

  // words per row: a resultIn beat (cu.numOutputs lanes) or wider, so the
  // write switch can place several beats in one row
  val totalWordsPerRow: Int = numBanks * wordsPerBank

  // cycles from a resultIn beat until a read of its row sees it: every
  // write first waits readLatency (the read-modify-write read), then
  // writeLatency. Also the minimum distance between two accumulating beats
  // to the same row.
  val commitLatency: Int = readLatency + writeLatency

  val rowsPerBank: Int = depth / rowBanks
  // global row address width (all banks)
  val addrWidth: Int = log2Ceil(depth)

  // one row bank
  val sram: SramConfig = SramConfig(
    depth = rowsPerBank,
    width = wordsPerBank * cu.outputWidth,
    numBanks = numBanks,
    maskGranularity = cu.outputWidth,
    readLatency = readLatency,
    writeLatency = writeLatency,
    portType = BankPortType.OneR1W,
  )

  // drain side: a row's words -> one cu.numOutputs-lane beat
  val laneSwitch: LaneSwitchConfig = LaneSwitchConfig(
    numIn = totalWordsPerRow,
    numOut = cu.numOutputs,
    inPerOut = 1,
    laneWidth = cu.outputWidth,
    maxFanout = maxFanout,
  )

  // write side: a resultIn beat, zero-padded to the row, -> the row's words.
  // The same switch routes each lane's write-enable, so only the words a
  // beat lands on are written (the SRAM write mask) and the others keep
  // their value. resultIn lane i lands on word (i - rotate) mod
  // totalWordsPerRow, so rotate = totalWordsPerRow - k places the beat at
  // word k.
  val writeSwitch: LaneSwitchConfig = LaneSwitchConfig(
    numIn = totalWordsPerRow,
    numOut = totalWordsPerRow,
    inPerOut = 1,
    laneWidth = cu.outputWidth,
    maxFanout = maxFanout,
  )
}

class AccumulatorIO(cfg: AccumulatorConfig) extends Bundle {
  val resultIn = Flipped(Valid(Vec(cfg.cu.numOutputs, SInt(cfg.cu.outputWidth.W))))
  // group 0's row address, then groups 1 .. addrGroups-1 (AddrGroups)
  val writeAddr = Input(UInt(cfg.addrWidth.W))
  val writeGroupAddr = AddrGroups.extra(cfg.addrGroups, UInt(cfg.addrWidth.W)).map(Input(_))
  // true = add resultIn to the row's current value, false = overwrite it
  val accumulate = Input(Bool())
  // where the beat's lanes land in the row (writeSwitch), with resultIn
  val writeRotate = Input(UInt(cfg.writeSwitch.rotateWidth.W))
  val writeFanout = Input(UInt(cfg.writeSwitch.fanoutWidth.W))
  val writeInterleave = Input(UInt(cfg.writeSwitch.interleaveWidth.W))

  val readAddr = Input(UInt(cfg.addrWidth.W))
  val readGroupAddr = AddrGroups.extra(cfg.addrGroups, UInt(cfg.addrWidth.W)).map(Input(_))
  val readEn = Input(Bool())
  val rotate = Input(UInt(cfg.laneSwitch.rotateWidth.W))
  val fanout = Input(UInt(cfg.laneSwitch.fanoutWidth.W))
  val interleave = Input(UInt(cfg.laneSwitch.interleaveWidth.W))
  val accOut = Valid(Vec(cfg.cu.numOutputs, UInt(cfg.cu.outputWidth.W)))
}

class Accumulator(cfg: AccumulatorConfig) extends Module {
  val io = IO(new AccumulatorIO(cfg))

  private val srams = Seq.fill(cfg.rowBanks)(Module(new Sram(cfg.sram)))
  private val localWidth = cfg.sram.addrWidth

  // row bank of a global row address, and the row within it
  private def bankOf(addr: UInt): UInt = if (cfg.rowBanks == 1) 0.U else addr >> localWidth
  private def local(addr: UInt): UInt = if (cfg.rowBanks == 1) addr else addr(localWidth - 1, 0)

  // word i (0 = least significant) of a bank's wordsPerBank-wide row
  private def wordSlice(row: UInt, i: Int): UInt =
    row((i + 1) * cfg.cu.outputWidth - 1, i * cfg.cu.outputWidth)

  // each address group's row address, group 0 first (AddrGroups)
  private val writeAddrs = AddrGroups.all(io.writeAddr, io.writeGroupAddr)
  private val readAddrs = AddrGroups.all(io.readAddr, io.readGroupAddr)
  private val banksPerGroup = cfg.numBanks / cfg.addrGroups

  // Read-modify-write: an accumulating beat reads its row's old value
  // through its row bank's read port (so a drain of the same row bank, in
  // any address group, must not be scheduled in the same cycle), is held
  // readLatency cycles until that value returns, then written as old + new.
  // Non-accumulating beats take the same delay, so write timing is one
  // constant (commitLatency). The beat is routed onto the row's words on
  // arrival.
  private def padded[T <: Data](lanes: Seq[T], zero: T): Vec[T] =
    VecInit(Seq.tabulate(cfg.totalWordsPerRow)(i => if (i < lanes.length) lanes(i) else zero))
  private val routed = LaneSwitch.connect(cfg.writeSwitch,
    padded(io.resultIn.bits.map(_.asUInt), 0.U(cfg.cu.outputWidth.W)),
    io.writeRotate, io.writeFanout, io.writeInterleave)
  private val routedMask = LaneSwitch.connect(cfg.writeSwitch.copy(laneWidth = 1),
    padded(Seq.fill(cfg.cu.numOutputs)(1.U(1.W)), 0.U(1.W)),
    io.writeRotate, io.writeFanout, io.writeInterleave)

  private val rmw = io.resultIn.valid && io.accumulate
  private val wValid = ShiftRegister(io.resultIn.valid, cfg.readLatency, false.B, true.B)
  private val wAccumulate = ShiftRegister(rmw, cfg.readLatency, false.B, true.B)
  private val wAddrs = writeAddrs.map(a => ShiftRegister(a, cfg.readLatency, 0.U(cfg.addrWidth.W), true.B))
  private val wData = ShiftRegister(routed, cfg.readLatency)
  private val wMask = ShiftRegister(routedMask, cfg.readLatency)

  // each row bank's rows as flat words, from its read-only ports (even index)
  private val bankWords = srams.zipWithIndex.map { case (sram, b) =>
    // per address group: does its RMW read or its drain use this row bank
    val rmwHere = writeAddrs.map(a => rmw && bankOf(a) === b.U)
    val drainHere = readAddrs.map(a => io.readEn && bankOf(a) === b.U)
    for (n <- 0 until cfg.numBanks) {
      val g = n / banksPerGroup
      val rp = sram.io.ports(2 * n)
      rp.addr := Mux(rmwHere(g), local(writeAddrs(g)), local(readAddrs(g)))
      rp.en := rmwHere(g) || drainHere(g)
      rp.wen := false.B // unused by Sram's readOnlyAccess
      rp.wdata := 0.U
      rp.wmask := 0.U
    }
    VecInit((0 until cfg.numBanks).flatMap { n =>
      (0 until cfg.wordsPerBank).map(i => wordSlice(sram.io.ports(2 * n).rdata, i))
    })
  }
  // the row words, each address group's from its own row bank (banks(g))
  private def fromBanks(banks: Seq[UInt]): Vec[UInt] =
    if (cfg.rowBanks == 1) bankWords.head
    else if (cfg.addrGroups == 1) VecInit(bankWords)(banks.head)
    else VecInit(Seq.tabulate(cfg.totalWordsPerRow)(i =>
      VecInit(bankWords.map(_(i)))(banks(i / cfg.wordsPerBank / banksPerGroup))))

  private val oldWords = fromBanks(wAddrs.map(bankOf))
  private val sum = VecInit(Seq.tabulate(cfg.totalWordsPerRow)(i =>
    Mux(wAccumulate, oldWords(i).asSInt + wData(i).asSInt, wData(i).asSInt)
  ))

  // write-only port of each bank pair (odd index, per Sram's OneR1W layout)
  for ((sram, b) <- srams.zipWithIndex; n <- 0 until cfg.numBanks) {
    val wp = sram.io.ports(2 * n + 1) // ports: [bank0-read, bank0-write, bank1-read, bank1-write, ...]
    val wAddr = wAddrs(n / banksPerGroup)
    wp.addr := local(wAddr)
    wp.en := wValid && bankOf(wAddr) === b.U
    wp.wen := true.B // unused by Sram's writeOnlyAccess; set for clarity
    wp.wdata := Cat((0 until cfg.wordsPerBank).reverse.map(i => sum(n * cfg.wordsPerBank + i).asUInt))
    // one mask bit per word (maskGranularity = outputWidth)
    wp.wmask := Cat((0 until cfg.wordsPerBank).reverse.map(i => wMask(n * cfg.wordsPerBank + i)))
  }

  private val rotatePipe = ShiftRegister(io.rotate, cfg.readLatency, 0.U(cfg.laneSwitch.rotateWidth.W), true.B)
  private val fanoutPipe =
    ShiftRegister(io.fanout, cfg.readLatency, 0.U(cfg.laneSwitch.fanoutWidth.W), true.B)
  private val interleavePipe =
    ShiftRegister(io.interleave, cfg.readLatency, 0.U(cfg.laneSwitch.interleaveWidth.W), true.B)
  private val drainBanks = readAddrs.map(a => ShiftRegister(bankOf(a), cfg.readLatency, 0.U, true.B))

  io.accOut.bits := LaneSwitch.connect(cfg.laneSwitch, fromBanks(drainBanks), rotatePipe, fanoutPipe, interleavePipe)
  io.accOut.valid := ShiftRegister(io.readEn, cfg.readLatency, false.B, true.B)
}
