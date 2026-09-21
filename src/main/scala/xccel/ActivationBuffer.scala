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
  windowDepth: Int = 1, // For Im2Col (1 : disable line buffer, >1 : enable line buffer)
) {
  require(depth > 0, "depth must be positive")
  require(wordsPerBank > 0, "wordsPerBank must be positive")
  require(windowDepth > 0, "windowDepth must be positive")
  require(
    cu.numInputs % windowDepth == 0,
    s"cu.numInputs (${cu.numInputs}) must be a multiple of windowDepth ($windowDepth):" +
      " each of the windowDepth resident rows contributes an equal-width slice of one sample"
  )

  val totalWordsPerRow: Int = numBanks * wordsPerBank
  val rowWidth: Int = cu.numInputs / windowDepth // per-row port count fed to PortAdapter

  val sram: SramConfig = SramConfig(
    depth = totalWordsPerRow * depth,
    wordLength = cu.inputWidth,
    numBanks = numBanks,
    bankBandwidth = wordsPerBank * cu.inputWidth,
    maskGranularity = cu.inputWidth,
    readLatency = readLatency,
    writeLatency = writeLatency,
  )

  val adapter: PortAdapterConfig = PortAdapterConfig(
    totalGrains = totalWordsPerRow,
    grainWidth = cu.inputWidth,
    numPorts = rowWidth,
    grainsPerPort = 1,
  )
}

class ActivationBufferIO(cfg: ActivationBufferConfig) extends Bundle {
  val addr = Input(UInt(cfg.sram.addrWidth.W))
  val en = Input(Bool())
  val wen = Input(Bool())
  val wdata = Input(Vec(cfg.totalWordsPerRow, UInt(cfg.cu.inputWidth.W)))
  val shiftAmt = Input(UInt(cfg.adapter.shiftWidth.W))
  val groupShift = Input(UInt(cfg.adapter.groupShiftWidth.W))
  // windowDepth>1 only: true = this row is new, shift it into the line
  // buffer's history; false = re-select within the currently resident
  // rows (e.g. a different shiftAmt column) without disturbing history.
  // Ignored when windowDepth=1 (no history exists to shift).
  val shiftIn = Input(Bool())

  // matches ComputeUnitIO.operandIn exactly
  val operandOut = Valid(Vec(cfg.cu.numInputs, UInt(cfg.cu.inputWidth.W)))
}

class ActivationBuffer(cfg: ActivationBufferConfig) extends Module {
  val io = IO(new ActivationBufferIO(cfg))

  private val sram = Module(new Sram(cfg.sram))
  private val fullMask = ((BigInt(1) << cfg.sram.maskBits) - 1).U(cfg.sram.maskBits.W)

  // word i (0 = least significant) of a bank's wordsPerBank-wide row
  private def wordSlice(row: UInt, i: Int): UInt =
    row((i + 1) * cfg.cu.inputWidth - 1, i * cfg.cu.inputWidth)

  for (n <- 0 until cfg.numBanks) {
    val bank = sram.io.banks(n)
    bank.addr := io.addr
    bank.en := io.en
    bank.wen := io.wen
    // pack this bank's wordsPerBank words (word 0 = LSBs) into one row
    bank.wdata := Cat((0 until cfg.wordsPerBank).reverse.map(i => io.wdata(n * cfg.wordsPerBank + i)))
    bank.wmask := fullMask
  }

  private val flatWords = VecInit((0 until cfg.numBanks).flatMap { n =>
    val bank = sram.io.banks(n)
    (0 until cfg.wordsPerBank).map(i => wordSlice(bank.rdata, i))
  })

  private val shiftAmtPipe = ShiftRegister(io.shiftAmt, cfg.readLatency, 0.U(cfg.adapter.shiftWidth.W), true.B)
  private val groupShiftPipe =
    ShiftRegister(io.groupShift, cfg.readLatency, 0.U(cfg.adapter.groupShiftWidth.W), true.B)

  io.operandOut.valid := ShiftRegister(io.en && !io.wen, cfg.readLatency, false.B, true.B)

  // line buffer (when windowDepth=1, no line buffer generates)
  private val historyDepth = cfg.windowDepth - 1
  private val rows: Seq[Vec[UInt]] = if (historyDepth == 0) {
    Seq(flatWords)
  } else {
    val zeroRow = VecInit(Seq.fill(cfg.totalWordsPerRow)(0.U(cfg.cu.inputWidth.W)))
    val history = RegInit(VecInit(Seq.fill(historyDepth)(zeroRow)))
    val shiftInPipe = ShiftRegister(io.en && !io.wen && io.shiftIn, cfg.readLatency, false.B, true.B)
    when(shiftInPipe) {
      history := VecInit(Seq.tabulate(historyDepth)(i => if (i < historyDepth - 1) history(i + 1) else flatWords))
    }
    Seq.tabulate(historyDepth)(i => history(i)) :+ flatWords
  }

  io.operandOut.bits :=
    VecInit(rows.flatMap(row => PortAdapter.connect(cfg.adapter, row, shiftAmtPipe, groupShiftPipe).toSeq))
}
