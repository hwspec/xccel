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
) {
  require(depth > 0, "depth must be positive")
  require(wordsPerBank > 0, "wordsPerBank must be positive")
  require(
    numBanks * wordsPerBank == cu.numOutputs,
    s"numBanks*wordsPerBank (${numBanks * wordsPerBank}) must equal cu.numOutputs (${cu.numOutputs}):" +
      " each resultIn lane maps onto exactly one grain slot, one-to-one -- the write side has no PortAdapter"
  )

  val totalWordsPerRow: Int = numBanks * wordsPerBank // == cu.numOutputs

  val sram: SramConfig = SramConfig(
    depth = totalWordsPerRow * depth,
    wordLength = cu.outputWidth,
    numBanks = numBanks,
    bankBandwidth = wordsPerBank * cu.outputWidth,
    maskGranularity = cu.outputWidth,
    readLatency = readLatency,
    writeLatency = writeLatency,
    portType = BankPortType.OneR1W,
  )

  val adapter: PortAdapterConfig = PortAdapterConfig(
    totalGrains = totalWordsPerRow,
    grainWidth = cu.outputWidth,
    numPorts = cu.numOutputs,
    grainsPerPort = 1,
  )
}

class AccumulatorIO(cfg: AccumulatorConfig) extends Bundle {
  val resultIn = Flipped(Valid(Vec(cfg.cu.numOutputs, SInt(cfg.cu.outputWidth.W))))
  val writeAddr = Input(UInt(cfg.sram.addrWidth.W))

  val readAddr = Input(UInt(cfg.sram.addrWidth.W))
  val readEn = Input(Bool())
  val shiftAmt = Input(UInt(cfg.adapter.shiftWidth.W))
  val groupShift = Input(UInt(cfg.adapter.groupShiftWidth.W))
  val accOut = Valid(Vec(cfg.cu.numOutputs, UInt(cfg.cu.outputWidth.W)))
}

class Accumulator(cfg: AccumulatorConfig) extends Module {
  val io = IO(new AccumulatorIO(cfg))

  private val sram = Module(new Sram(cfg.sram))
  private val fullMask = ((BigInt(1) << cfg.sram.maskBits) - 1).U(cfg.sram.maskBits.W)

  // word i (0 = least significant) of a bank's wordsPerBank-wide row
  private def wordSlice(row: UInt, i: Int): UInt =
    row((i + 1) * cfg.cu.outputWidth - 1, i * cfg.cu.outputWidth)

  // write-only channel of each bank pair (odd index, per Sram's OneR1W layout)
  for (n <- 0 until cfg.numBanks) {
    val wch = sram.io.banks(2 * n + 1) // banks: [bank0-read, bank0-write, bank1-read, bank1-write, ...]
    wch.addr := io.writeAddr
    wch.en := io.resultIn.valid
    wch.wen := true.B // unused by Sram's writeOnlyAccess; set for clarity
    wch.wdata := Cat(
      (0 until cfg.wordsPerBank).reverse.map(i => io.resultIn.bits(n * cfg.wordsPerBank + i).asUInt)
    )
    wch.wmask := fullMask
  }

  // read-only channel of each bank pair (even index)
  for (n <- 0 until cfg.numBanks) {
    val rch = sram.io.banks(2 * n)
    rch.addr := io.readAddr
    rch.en := io.readEn
    rch.wen := false.B // unused by Sram's readOnlyAccess
    rch.wdata := 0.U
    rch.wmask := 0.U
  }

  private val flatWords = VecInit((0 until cfg.numBanks).flatMap { n =>
    val rch = sram.io.banks(2 * n)
    (0 until cfg.wordsPerBank).map(i => wordSlice(rch.rdata, i))
  })

  private val shiftAmtPipe = ShiftRegister(io.shiftAmt, cfg.readLatency, 0.U(cfg.adapter.shiftWidth.W), true.B)
  private val groupShiftPipe =
    ShiftRegister(io.groupShift, cfg.readLatency, 0.U(cfg.adapter.groupShiftWidth.W), true.B)

  io.accOut.bits := PortAdapter.connect(cfg.adapter, flatWords, shiftAmtPipe, groupShiftPipe)
  io.accOut.valid := ShiftRegister(io.readEn, cfg.readLatency, false.B, true.B)
}
