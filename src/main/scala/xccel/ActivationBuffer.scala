package xccel

import chisel3._
import chisel3.util._

case class ActivationBufferConfig(
  cu: ComputeUnitConfig,
  numBanks: Int,
  wordsPerBank: Int,
  depth: Int,
  readLatency: Int,
  writeLatency: Int,
  portType: BankPortType = BankPortType.OneRW,
  // Line buffer disabled: im2col is done by accumulating kn2row instead (one
  // 1x1 pass per kernel position, partial sums added in Accumulator). Kept
  // commented out for reference.
  // windowDepth: Int = 1, // For Im2Col (1 : disable line buffer, >1 : enable line buffer)
  // largest LaneSwitch fanout built (see LaneSwitchConfig): None = up to
  // broadcast, Some(0) = unicast only
  maxFanout: Option[Int] = None,
) {
  require(depth > 0, "depth must be positive")
  require(wordsPerBank > 0, "wordsPerBank must be positive")
  // require(windowDepth > 0, "windowDepth must be positive")
  // require(
  //   cu.numInputs % windowDepth == 0,
  //   s"cu.numInputs (${cu.numInputs}) must be a multiple of windowDepth ($windowDepth):" +
  //     " each of the windowDepth resident rows contributes an equal-width slice of one sample"
  // )

  val totalWordsPerRow: Int = numBanks * wordsPerBank
  // val rowWidth: Int = cu.numInputs / windowDepth // per-row lane count fed to LaneSwitch
  val rowWidth: Int = cu.numInputs

  val sram: SramConfig = SramConfig(
    depth = totalWordsPerRow * depth,
    wordLength = cu.inputWidth,
    numBanks = numBanks,
    bankBandwidth = wordsPerBank * cu.inputWidth,
    maskGranularity = cu.inputWidth,
    readLatency = readLatency,
    writeLatency = writeLatency,
    portType = portType,
  )

  val laneSwitch: LaneSwitchConfig = LaneSwitchConfig(
    numIn = totalWordsPerRow,
    numOut = rowWidth,
    inPerOut = 1,
    laneWidth = cu.inputWidth,
    maxFanout = maxFanout,
  )
}

class ActivationBufferIO(cfg: ActivationBufferConfig) extends Bundle {
  // read stream (the program) and write stream (host / other source);
  // how they map onto SRAM ports depends on cfg.portType (Sram.connectReadWrite)
  val readAddr = Input(UInt(cfg.sram.addrWidth.W))
  val readEn = Input(Bool())
  val writeAddr = Input(UInt(cfg.sram.addrWidth.W))
  val writeEn = Input(Bool())
  val wdata = Input(Vec(cfg.totalWordsPerRow, UInt(cfg.cu.inputWidth.W)))
  val rotate = Input(UInt(cfg.laneSwitch.rotateWidth.W))
  val fanout = Input(UInt(cfg.laneSwitch.fanoutWidth.W))
  // Line buffer disabled: im2col is done by accumulating kn2row instead (one
  // 1x1 pass per kernel position, partial sums added in Accumulator). Kept
  // commented out for reference.
  // // windowDepth>1 only: true = this row is new, shift it into the line
  // // buffer's history; false = re-select within the currently resident
  // // rows (e.g. a different rotate column) without disturbing history.
  // // Ignored when windowDepth=1 (no history exists to shift).
  // val shiftIn = Input(Bool())

  // matches ComputeUnitIO.operandIn exactly
  val operandOut = Valid(Vec(cfg.cu.numInputs, UInt(cfg.cu.inputWidth.W)))
}

class ActivationBuffer(cfg: ActivationBufferConfig) extends Module {
  val io = IO(new ActivationBufferIO(cfg))

  private val sram = Module(new Sram(cfg.sram))

  // word i (0 = least significant) of a bank's wordsPerBank-wide row
  private def wordSlice(row: UInt, i: Int): UInt =
    row((i + 1) * cfg.cu.inputWidth - 1, i * cfg.cu.inputWidth)

  // pack each bank's wordsPerBank words (word 0 = LSBs) into one row
  private val rdata = Sram.connectReadWrite(
    cfg.sram,
    sram.io,
    io.readAddr,
    io.readEn,
    io.writeAddr,
    io.writeEn,
    Seq.tabulate(cfg.numBanks)(n => Cat((0 until cfg.wordsPerBank).reverse.map(i => io.wdata(n * cfg.wordsPerBank + i)))),
  )

  private val flatWords = VecInit((0 until cfg.numBanks).flatMap { n =>
    (0 until cfg.wordsPerBank).map(i => wordSlice(rdata(n), i))
  })

  private val rotatePipe = ShiftRegister(io.rotate, cfg.readLatency, 0.U(cfg.laneSwitch.rotateWidth.W), true.B)
  private val fanoutPipe =
    ShiftRegister(io.fanout, cfg.readLatency, 0.U(cfg.laneSwitch.fanoutWidth.W), true.B)

  io.operandOut.valid := ShiftRegister(io.readEn, cfg.readLatency, false.B, true.B)

  // Line buffer disabled: im2col is done by accumulating kn2row instead (one
  // 1x1 pass per kernel position, partial sums added in Accumulator). Kept
  // commented out for reference.
  // // line buffer (when windowDepth=1, no line buffer generates)
  // private val historyDepth = cfg.windowDepth - 1
  // private val rows: Seq[Vec[UInt]] = if (historyDepth == 0) {
  //   Seq(flatWords)
  // } else {
  //   val zeroRow = VecInit(Seq.fill(cfg.totalWordsPerRow)(0.U(cfg.cu.inputWidth.W)))
  //   val history = RegInit(VecInit(Seq.fill(historyDepth)(zeroRow)))
  //   val shiftInPipe = ShiftRegister(io.en && !io.wen && io.shiftIn, cfg.readLatency, false.B, true.B)
  //   when(shiftInPipe) {
  //     history := VecInit(Seq.tabulate(historyDepth)(i => if (i < historyDepth - 1) history(i + 1) else flatWords))
  //   }
  //   Seq.tabulate(historyDepth)(i => history(i)) :+ flatWords
  // }

  // io.operandOut.bits :=
  //   VecInit(rows.flatMap(row => LaneSwitch.connect(cfg.laneSwitch, row, rotatePipe, fanoutPipe).toSeq))
  io.operandOut.bits := LaneSwitch.connect(cfg.laneSwitch, flatWords, rotatePipe, fanoutPipe)
}
