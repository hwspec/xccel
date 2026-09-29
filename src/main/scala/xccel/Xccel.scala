package xccel

import chisel3._
import chisel3.util._

case class XccelConfig(
  cu: ComputeUnitConfig,
  weightBuffer: WeightBufferConfig,
  activationBuffer: ActivationBufferConfig,
  accumulator: AccumulatorConfig,
  requantizedWidth: Int,
  programLength: Int,
  maxRunLength: Int,
) {
  require(weightBuffer.cu == cu, "weightBuffer must share this Xccel's ComputeUnitConfig")
  require(activationBuffer.cu == cu, "activationBuffer must share this Xccel's ComputeUnitConfig")
  require(accumulator.cu == cu, "accumulator must share this Xccel's ComputeUnitConfig")

  val requantizer: RequantizerConfig = RequantizerConfig(cu.numOutputs, cu.outputWidth, requantizedWidth)
  val decoder: DecoderConfig = DecoderConfig(weightBuffer, activationBuffer, accumulator, programLength, maxRunLength)
}

class XccelIO(cfg: XccelConfig) extends Bundle {
  val start = Input(Bool())
  val running = Output(Bool())

  // microcode load, host-driven before start
  val progWriteEn = Input(Bool())
  val progWriteAddr = Input(UInt(cfg.decoder.wordIdxWidth.W))
  val progWriteData = Input(new MicroWord(cfg.decoder))

  // host-facing load ports. Before start (running=0) the host writes with
  // its own addr/en. While running, only the data ports are used: the
  // program's wbWrite/abWrite slots give addr/en, and the host must present
  // that cycle's data on wbWriteData/abWriteData.
  val wbWriteAddr = Input(UInt(cfg.weightBuffer.sram.addrWidth.W))
  val wbWriteEn = Input(Bool())
  val wbWriteData = Input(Vec(cfg.weightBuffer.totalWordsPerRow, UInt(cfg.cu.weightWidth.W)))

  val abWriteAddr = Input(UInt(cfg.activationBuffer.sram.addrWidth.W))
  val abWriteEn = Input(Bool())
  val abWriteData = Input(Vec(cfg.activationBuffer.totalWordsPerRow, UInt(cfg.cu.inputWidth.W)))

  // host drain of Accumulator -- same rule, yields to the program's
  // accRead slot whenever that slot is enabled
  val accReadAddr = Input(UInt(cfg.accumulator.sram.addrWidth.W))
  val accReadEn = Input(Bool())
  val accumOut = Valid(Vec(cfg.cu.numOutputs, UInt(cfg.requantizedWidth.W)))
}

class Xccel(cfg: XccelConfig) extends Module {
  val io = IO(new XccelIO(cfg))

  private val decoder = Module(new Decoder(cfg.decoder))
  decoder.io.start := io.start
  decoder.io.progWriteEn := io.progWriteEn
  decoder.io.progWriteAddr := io.progWriteAddr
  decoder.io.progWriteData := io.progWriteData
  io.running := decoder.io.running

  private val weightBuffer = Module(new WeightBuffer(cfg.weightBuffer))
  private val activationBuffer = Module(new ActivationBuffer(cfg.activationBuffer))
  private val compute = Module(new SystolicArray(cfg.cu))
  private val accumulator = Module(new Accumulator(cfg.accumulator))
  private val requantizer = Module(new Requantizer(cfg.requantizer))

  // -- WeightBuffer: reads from the program; writes from the program while
  // running (data from the host) or from the host before start. With a
  // OneRW SRAM reads and writes share one port (a write in a cycle with a
  // read is dropped); OneR1W/TwoRW give each its own.
  private val wb = decoder.io.wbRead
  weightBuffer.io.readAddr := wb.addr
  weightBuffer.io.readEn := wb.en
  weightBuffer.io.writeAddr := Mux(decoder.io.running, decoder.io.wbWrite.addr, io.wbWriteAddr)
  weightBuffer.io.writeEn := Mux(decoder.io.running, decoder.io.wbWrite.en, io.wbWriteEn)
  weightBuffer.io.wdata := io.wbWriteData
  weightBuffer.io.shiftAmt := wb.shiftAmt
  weightBuffer.io.groupShift := wb.groupShift

  // the weight port has no valid of its own: its strobe and row address
  // come straight from the program's weightPort slot
  compute.weightWriteEn := decoder.io.weightPort.en
  compute.weightWriteAddr := decoder.io.weightPort.addr
  compute.weightWriteData := weightBuffer.io.weightOut.bits
  compute.weightWriteSlot := decoder.io.weightPort.slot
  compute.weightSelect := decoder.io.weightSelect

  // -- ActivationBuffer: same read/write arrangement, then a direct
  // connect into the compute unit (shapes match exactly)
  private val ab = decoder.io.abRead
  activationBuffer.io.readAddr := ab.addr
  activationBuffer.io.readEn := ab.en
  activationBuffer.io.writeAddr := Mux(decoder.io.running, decoder.io.abWrite.addr, io.abWriteAddr)
  activationBuffer.io.writeEn := Mux(decoder.io.running, decoder.io.abWrite.en, io.abWriteEn)
  activationBuffer.io.wdata := io.abWriteData
  activationBuffer.io.shiftAmt := ab.shiftAmt
  activationBuffer.io.groupShift := ab.groupShift
  // activationBuffer.io.shiftIn := ab.shiftIn // line buffer disabled (accumulating kn2row)

  compute.io.operandIn <> activationBuffer.io.operandOut

  // -- resultOut -> Accumulator: the write strobe is resultIn.valid
  // (self-timed), the address and add-vs-overwrite come from the program.
  // An accumulating beat uses Accumulator's read channel, so the program
  // must not schedule an accRead in the same cycle.
  accumulator.io.resultIn <> compute.io.resultOut
  accumulator.io.writeAddr := decoder.io.accWrite.addr
  accumulator.io.accumulate := decoder.io.accWrite.accumulate

  // -- drain: the program's accRead slot, otherwise the host
  private val acc = decoder.io.accRead
  accumulator.io.readAddr := Mux(acc.en, acc.addr, io.accReadAddr)
  accumulator.io.readEn := acc.en || io.accReadEn
  accumulator.io.shiftAmt := Mux(acc.en, acc.shiftAmt, 0.U)
  accumulator.io.groupShift := Mux(acc.en, acc.groupShift, 0.U)

  requantizer.io.in <> accumulator.io.accOut
  io.accumOut <> requantizer.io.out
}
