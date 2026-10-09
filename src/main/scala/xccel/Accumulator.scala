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
) {
  require(depth > 0, "depth must be positive")
  require(rowBanks > 0 && depth % rowBanks == 0, "rowBanks must divide depth")
  require(rowBanks == 1 || isPow2(depth / rowBanks), "rows per bank must be a power of two with rowBanks > 1")
  require(wordsPerBank > 0, "wordsPerBank must be positive")
  require(
    numBanks * wordsPerBank == cu.numOutputs,
    s"numBanks*wordsPerBank (${numBanks * wordsPerBank}) must equal cu.numOutputs (${cu.numOutputs}):" +
      " each resultIn lane maps onto exactly one word of the row, one-to-one -- the write side has no LaneSwitch"
  )

  val totalWordsPerRow: Int = numBanks * wordsPerBank // == cu.numOutputs

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
    depth = totalWordsPerRow * rowsPerBank,
    wordLength = cu.outputWidth,
    numBanks = numBanks,
    bankBandwidth = wordsPerBank * cu.outputWidth,
    maskGranularity = cu.outputWidth,
    readLatency = readLatency,
    writeLatency = writeLatency,
    portType = BankPortType.OneR1W,
  )

  val laneSwitch: LaneSwitchConfig = LaneSwitchConfig(
    numIn = totalWordsPerRow,
    numOut = cu.numOutputs,
    inPerOut = 1,
    laneWidth = cu.outputWidth,
    maxFanout = maxFanout,
  )
}

class AccumulatorIO(cfg: AccumulatorConfig) extends Bundle {
  val resultIn = Flipped(Valid(Vec(cfg.cu.numOutputs, SInt(cfg.cu.outputWidth.W))))
  val writeAddr = Input(UInt(cfg.addrWidth.W))
  // true = add resultIn to the row's current value, false = overwrite it
  val accumulate = Input(Bool())

  val readAddr = Input(UInt(cfg.addrWidth.W))
  val readEn = Input(Bool())
  val shiftAmt = Input(UInt(cfg.laneSwitch.shiftWidth.W))
  val fanout = Input(UInt(cfg.laneSwitch.fanoutWidth.W))
  val accOut = Valid(Vec(cfg.cu.numOutputs, UInt(cfg.cu.outputWidth.W)))
}

class Accumulator(cfg: AccumulatorConfig) extends Module {
  val io = IO(new AccumulatorIO(cfg))

  private val srams = Seq.fill(cfg.rowBanks)(Module(new Sram(cfg.sram)))
  private val fullMask = ((BigInt(1) << cfg.sram.maskBits) - 1).U(cfg.sram.maskBits.W)
  private val localWidth = cfg.sram.addrWidth

  // row bank of a global row address, and the row within it
  private def bankOf(addr: UInt): UInt = if (cfg.rowBanks == 1) 0.U else addr >> localWidth
  private def local(addr: UInt): UInt = if (cfg.rowBanks == 1) addr else addr(localWidth - 1, 0)

  // word i (0 = least significant) of a bank's wordsPerBank-wide row
  private def wordSlice(row: UInt, i: Int): UInt =
    row((i + 1) * cfg.cu.outputWidth - 1, i * cfg.cu.outputWidth)

  // Read-modify-write: an accumulating beat reads its row's old value
  // through its row bank's read channel (so a drain of the same row bank
  // must not be scheduled in the same cycle), is held readLatency cycles
  // until that value returns, then written as old + new. Non-accumulating
  // beats take the same delay, so write timing is one constant
  // (commitLatency).
  private val rmw = io.resultIn.valid && io.accumulate
  private val wValid = ShiftRegister(io.resultIn.valid, cfg.readLatency, false.B, true.B)
  private val wAccumulate = ShiftRegister(rmw, cfg.readLatency, false.B, true.B)
  private val wAddr = ShiftRegister(io.writeAddr, cfg.readLatency, 0.U(cfg.addrWidth.W), true.B)
  private val wData = ShiftRegister(io.resultIn.bits, cfg.readLatency)

  // each row bank's rows as flat words, from its read-only channels (even index)
  private val bankWords = srams.zipWithIndex.map { case (sram, b) =>
    val rmwHere = rmw && bankOf(io.writeAddr) === b.U
    val drainHere = io.readEn && bankOf(io.readAddr) === b.U
    for (n <- 0 until cfg.numBanks) {
      val rch = sram.io.banks(2 * n)
      rch.addr := Mux(rmwHere, local(io.writeAddr), local(io.readAddr))
      rch.en := rmwHere || drainHere
      rch.wen := false.B // unused by Sram's readOnlyAccess
      rch.wdata := 0.U
      rch.wmask := 0.U
    }
    VecInit((0 until cfg.numBanks).flatMap { n =>
      (0 until cfg.wordsPerBank).map(i => wordSlice(sram.io.banks(2 * n).rdata, i))
    })
  }
  private def fromBank(bank: UInt): Vec[UInt] =
    if (cfg.rowBanks == 1) bankWords.head else VecInit(bankWords)(bank)

  private val oldWords = fromBank(bankOf(wAddr))
  private val sum = VecInit(Seq.tabulate(cfg.totalWordsPerRow)(i =>
    Mux(wAccumulate, oldWords(i).asSInt + wData(i), wData(i))
  ))

  // write-only channel of each bank pair (odd index, per Sram's OneR1W layout)
  for ((sram, b) <- srams.zipWithIndex; n <- 0 until cfg.numBanks) {
    val wch = sram.io.banks(2 * n + 1) // banks: [bank0-read, bank0-write, bank1-read, bank1-write, ...]
    wch.addr := local(wAddr)
    wch.en := wValid && bankOf(wAddr) === b.U
    wch.wen := true.B // unused by Sram's writeOnlyAccess; set for clarity
    wch.wdata := Cat((0 until cfg.wordsPerBank).reverse.map(i => sum(n * cfg.wordsPerBank + i).asUInt))
    wch.wmask := fullMask
  }

  private val shiftAmtPipe = ShiftRegister(io.shiftAmt, cfg.readLatency, 0.U(cfg.laneSwitch.shiftWidth.W), true.B)
  private val fanoutPipe =
    ShiftRegister(io.fanout, cfg.readLatency, 0.U(cfg.laneSwitch.fanoutWidth.W), true.B)
  private val drainBank = ShiftRegister(bankOf(io.readAddr), cfg.readLatency, 0.U, true.B)

  io.accOut.bits := LaneSwitch.connect(cfg.laneSwitch, fromBank(drainBank), shiftAmtPipe, fanoutPipe)
  io.accOut.valid := ShiftRegister(io.readEn, cfg.readLatency, false.B, true.B)
}
