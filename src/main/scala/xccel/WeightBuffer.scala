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
) {
  require(tiles > 0, "tiles must be positive")
  require(wordsPerBank > 0, "wordsPerBank must be positive")

  val totalWordsPerRow: Int = numBanks * wordsPerBank
  val rowsTotal: Int = cu.numInputs * tiles

  val sram: SramConfig = SramConfig(
    depth = totalWordsPerRow * rowsTotal,
    wordLength = cu.weightWidth,
    numBanks = numBanks,
    bankBandwidth = wordsPerBank * cu.weightWidth,
    maskGranularity = cu.weightWidth,
    readLatency = readLatency,
    writeLatency = writeLatency,
  )

  val adapter: PortAdapterConfig = PortAdapterConfig(
    totalGrains = totalWordsPerRow,
    grainWidth = cu.weightWidth,
    numPorts = cu.numOutputs,
    grainsPerPort = 1,
  )
}

class WeightBufferIO(cfg: WeightBufferConfig) extends Bundle {
  val addr = Input(UInt(cfg.sram.addrWidth.W))
  val en = Input(Bool())
  val wen = Input(Bool())
  val wdata = Input(Vec(cfg.totalWordsPerRow, UInt(cfg.cu.weightWidth.W)))
  val shiftAmt = Input(UInt(cfg.adapter.shiftWidth.W))
  val groupShift = Input(UInt(cfg.adapter.groupShiftWidth.W))

  // matches ComputeUnitIO.weightIn exactly
  val weightOut = Valid(Vec(cfg.cu.numOutputs, UInt(cfg.cu.weightWidth.W)))
}

class WeightBuffer(cfg: WeightBufferConfig) extends Module {
  val io = IO(new WeightBufferIO(cfg))

  private val sram = Module(new Sram(cfg.sram))
  private val fullMask = ((BigInt(1) << cfg.sram.maskBits) - 1).U(cfg.sram.maskBits.W)

  // word i (0 = least significant) of a bank's wordsPerBank-wide row
  private def wordSlice(row: UInt, i: Int): UInt =
    row((i + 1) * cfg.cu.weightWidth - 1, i * cfg.cu.weightWidth)

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

  io.weightOut.bits := PortAdapter.connect(cfg.adapter, flatWords, shiftAmtPipe, groupShiftPipe)
  io.weightOut.valid := ShiftRegister(io.en && !io.wen, cfg.readLatency, false.B, true.B)
}
