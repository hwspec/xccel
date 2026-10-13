package xccel

import chisel3._
import chisel3.util._

case class WeightBufferConfig(
  cu: ComputeUnitConfig,
  numBanks: Int,
  wordsPerBank: Int,
  tiles: Int,
  readLatency: Int,
  writeLatency: Int,
  portType: BankPortType = BankPortType.OneRW,
  // lanes of the weight beat handed to the compute unit (its weightBeatElems);
  // 0 = cu.numOutputs (one array row per beat)
  weightLanes: Int = 0,
  // largest LaneSwitch fanout built (see LaneSwitchConfig): None = up to
  // broadcast, Some(0) = unicast only
  maxFanout: Option[Int] = None,
  // independent row addresses, one per run of numBanks / addrGroups
  // consecutive banks (see AddrGroups); 1 = all banks share one address
  addrGroups: Int = 1,
) {
  require(tiles > 0, "tiles must be positive")
  AddrGroups.validate(numBanks, addrGroups)
  require(wordsPerBank > 0, "wordsPerBank must be positive")

  val totalWordsPerRow: Int = numBanks * wordsPerBank
  val numWeightLanes: Int = if (weightLanes > 0) weightLanes else cu.numOutputs
  val rowsTotal: Int = cu.numInputs * tiles

  val sram: SramConfig = SramConfig(
    depth = rowsTotal,
    width = wordsPerBank * cu.weightWidth,
    numBanks = numBanks,
    maskGranularity = cu.weightWidth,
    readLatency = readLatency,
    writeLatency = writeLatency,
    portType = portType,
  )

  val laneSwitch: LaneSwitchConfig = LaneSwitchConfig(
    numIn = totalWordsPerRow,
    numOut = numWeightLanes,
    inPerOut = 1,
    laneWidth = cu.weightWidth,
    maxFanout = maxFanout,
  )
  // write side: a write's words -> the row's words. The same switch routes
  // each word's write enable (wmask), so only the words the data lands on
  // are written and the others keep their value: a beat narrower than the
  // row (e.g. a Requantizer beat) can be placed anywhere in it. wdata word i
  // lands on word (i - rotate) mod totalWordsPerRow.
  val writeSwitch: LaneSwitchConfig = LaneSwitchConfig(
    numIn = totalWordsPerRow,
    numOut = totalWordsPerRow,
    inPerOut = 1,
    laneWidth = cu.weightWidth,
    maxFanout = maxFanout,
  )

}

class WeightBufferIO(cfg: WeightBufferConfig) extends Bundle {
  // read stream (the program) and write stream (host / other source);
  // how they map onto SRAM ports depends on cfg.portType (Sram.connectReadWrite)
  // group 0's row address, then groups 1 .. addrGroups-1 (AddrGroups)
  val readAddr = Input(UInt(cfg.sram.addrWidth.W))
  val readGroupAddr = AddrGroups.extra(cfg.addrGroups, UInt(cfg.sram.addrWidth.W)).map(Input(_))
  val readEn = Input(Bool())
  val writeAddr = Input(UInt(cfg.sram.addrWidth.W))
  val writeGroupAddr = AddrGroups.extra(cfg.addrGroups, UInt(cfg.sram.addrWidth.W)).map(Input(_))
  val writeEn = Input(Bool())
  val wdata = Input(Vec(cfg.totalWordsPerRow, UInt(cfg.cu.weightWidth.W)))
  // which wdata words carry data (before the write switch)
  val wmask = Input(Vec(cfg.totalWordsPerRow, Bool()))
  val writeRotate = Input(UInt(cfg.writeSwitch.rotateWidth.W))
  val writeFanout = Input(UInt(cfg.writeSwitch.fanoutWidth.W))
  val writeInterleave = Input(UInt(cfg.writeSwitch.interleaveWidth.W))
  val rotate = Input(UInt(cfg.laneSwitch.rotateWidth.W))
  val fanout = Input(UInt(cfg.laneSwitch.fanoutWidth.W))
  val interleave = Input(UInt(cfg.laneSwitch.interleaveWidth.W))

  // one weight beat of the compute unit (HasStationaryWeights.weightWriteData)
  val weightOut = Valid(Vec(cfg.numWeightLanes, UInt(cfg.cu.weightWidth.W)))
}

class WeightBuffer(cfg: WeightBufferConfig) extends Module {
  val io = IO(new WeightBufferIO(cfg))

  private val sram = Module(new Sram(cfg.sram))

  // word i (0 = least significant) of a bank's wordsPerBank-wide row
  private def wordSlice(row: UInt, i: Int): UInt =
    row((i + 1) * cfg.cu.weightWidth - 1, i * cfg.cu.weightWidth)

  // the write's words and their enables, routed onto the row
  private val routed =
    LaneSwitch.connect(cfg.writeSwitch, io.wdata, io.writeRotate, io.writeFanout, io.writeInterleave)
  private val routedMask = LaneSwitch.connect(cfg.writeSwitch.copy(laneWidth = 1), VecInit(io.wmask.map(_.asUInt)),
    io.writeRotate, io.writeFanout, io.writeInterleave)

  // pack each bank's wordsPerBank words (word 0 = LSBs) into one row
  private val rdata = Sram.connectReadWrite(
    cfg.sram,
    sram.io,
    AddrGroups.perBank(AddrGroups.all(io.readAddr, io.readGroupAddr), cfg.numBanks),
    io.readEn,
    AddrGroups.perBank(AddrGroups.all(io.writeAddr, io.writeGroupAddr), cfg.numBanks),
    io.writeEn,
    Seq.tabulate(cfg.numBanks)(n => Cat((0 until cfg.wordsPerBank).reverse.map(i => routed(n * cfg.wordsPerBank + i)))),
    // one mask bit per word (maskGranularity = the word width)
    Seq.tabulate(cfg.numBanks)(n => Cat((0 until cfg.wordsPerBank).reverse.map(i => routedMask(n * cfg.wordsPerBank + i)))),
  )

  private val flatWords = VecInit((0 until cfg.numBanks).flatMap { n =>
    (0 until cfg.wordsPerBank).map(i => wordSlice(rdata(n), i))
  })

  private val rotatePipe = ShiftRegister(io.rotate, cfg.readLatency, 0.U(cfg.laneSwitch.rotateWidth.W), true.B)
  private val fanoutPipe =
    ShiftRegister(io.fanout, cfg.readLatency, 0.U(cfg.laneSwitch.fanoutWidth.W), true.B)
  private val interleavePipe =
    ShiftRegister(io.interleave, cfg.readLatency, 0.U(cfg.laneSwitch.interleaveWidth.W), true.B)

  io.weightOut.bits := LaneSwitch.connect(cfg.laneSwitch, flatWords, rotatePipe, fanoutPipe, interleavePipe)
  io.weightOut.valid := ShiftRegister(io.readEn, cfg.readLatency, false.B, true.B)
}
