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
  // words per weight beat handed to the compute unit (its weightBeatElems);
  // 0 = cu.numOutputs (one array row per beat)
  weightPorts: Int = 0,
) {
  require(tiles > 0, "tiles must be positive")
  require(wordsPerBank > 0, "wordsPerBank must be positive")

  val totalWordsPerRow: Int = numBanks * wordsPerBank
  val numWeightPorts: Int = if (weightPorts > 0) weightPorts else cu.numOutputs
  val rowsTotal: Int = cu.numInputs * tiles

  val sram: SramConfig = SramConfig(
    depth = totalWordsPerRow * rowsTotal,
    wordLength = cu.weightWidth,
    numBanks = numBanks,
    bankBandwidth = wordsPerBank * cu.weightWidth,
    maskGranularity = cu.weightWidth,
    readLatency = readLatency,
    writeLatency = writeLatency,
    portType = portType,
  )

  val adapter: PortAdapterConfig = PortAdapterConfig(
    totalGrains = totalWordsPerRow,
    grainWidth = cu.weightWidth,
    numPorts = numWeightPorts,
    grainsPerPort = 1,
  )
}

class WeightBufferIO(cfg: WeightBufferConfig) extends Bundle {
  // read stream (the program) and write stream (host / other source);
  // how they map onto SRAM ports depends on cfg.portType (Sram.connectReadWrite)
  val readAddr = Input(UInt(cfg.sram.addrWidth.W))
  val readEn = Input(Bool())
  val writeAddr = Input(UInt(cfg.sram.addrWidth.W))
  val writeEn = Input(Bool())
  val wdata = Input(Vec(cfg.totalWordsPerRow, UInt(cfg.cu.weightWidth.W)))
  val shiftAmt = Input(UInt(cfg.adapter.shiftWidth.W))
  val groupShift = Input(UInt(cfg.adapter.groupShiftWidth.W))

  // one weight beat of the compute unit (HasStationaryWeights.weightWriteData)
  val weightOut = Valid(Vec(cfg.numWeightPorts, UInt(cfg.cu.weightWidth.W)))
}

class WeightBuffer(cfg: WeightBufferConfig) extends Module {
  val io = IO(new WeightBufferIO(cfg))

  private val sram = Module(new Sram(cfg.sram))

  // word i (0 = least significant) of a bank's wordsPerBank-wide row
  private def wordSlice(row: UInt, i: Int): UInt =
    row((i + 1) * cfg.cu.weightWidth - 1, i * cfg.cu.weightWidth)

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

  private val shiftAmtPipe = ShiftRegister(io.shiftAmt, cfg.readLatency, 0.U(cfg.adapter.shiftWidth.W), true.B)
  private val groupShiftPipe =
    ShiftRegister(io.groupShift, cfg.readLatency, 0.U(cfg.adapter.groupShiftWidth.W), true.B)

  io.weightOut.bits := PortAdapter.connect(cfg.adapter, flatWords, shiftAmtPipe, groupShiftPipe)
  io.weightOut.valid := ShiftRegister(io.readEn, cfg.readLatency, false.B, true.B)
}
