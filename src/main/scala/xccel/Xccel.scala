package xccel

import chisel3._
import chisel3.util._

case class XccelConfig(
  cu: ComputeUnitConfig,
  weightBuffer: WeightBufferConfig,
  activationBuffer: ActivationBufferConfig,
  accumulator: AccumulatorConfig,
  requantizedWidth: Int,
) {
  require(weightBuffer.cu == cu, "weightBuffer must share this Xccel's ComputeUnitConfig")
  require(activationBuffer.cu == cu, "activationBuffer must share this Xccel's ComputeUnitConfig")
  require(accumulator.cu == cu, "accumulator must share this Xccel's ComputeUnitConfig")
  require(
    weightBuffer.tiles == 1,
    "Xccel's schedule only knows how to stream a single resident weight tile" +
      " (pc directly indexes rows 0..numInputs-1); tiles>1 needs a real tile-select, not built here"
  )
  require(
    accumulator.depth >= activationBuffer.depth,
    "Accumulator needs room for at least one result per streamed sample"
  )

  val numSamples: Int = activationBuffer.depth
  val latency: Int = cu.numInputs + cu.numOutputs - 1
  val requantizer: RequantizerConfig = RequantizerConfig(cu.numOutputs, cu.outputWidth, requantizedWidth)
}

class XccelIO(cfg: XccelConfig) extends Bundle {
  val start = Input(Bool())
  val stop = Input(Bool())
  val running = Output(Bool())

  // host-facing load ports -- only take effect outside each buffer's own
  // scheduled read window
  val wbWriteAddr = Input(UInt(cfg.weightBuffer.sram.addrWidth.W))
  val wbWriteEn = Input(Bool())
  val wbWriteData = Input(Vec(cfg.weightBuffer.totalWordsPerRow, UInt(cfg.cu.weightWidth.W)))

  val abWriteAddr = Input(UInt(cfg.activationBuffer.sram.addrWidth.W))
  val abWriteEn = Input(Bool())
  val abWriteData = Input(Vec(cfg.activationBuffer.totalWordsPerRow, UInt(cfg.cu.inputWidth.W)))

  // Accumulator's drain -- always available (OneR1W has its own
  // independent read bus, so it never needs to share with the schedule)
  val accReadAddr = Input(UInt(cfg.accumulator.sram.addrWidth.W))
  val accReadEn = Input(Bool())
  val accumOut = Valid(Vec(cfg.cu.numOutputs, UInt(cfg.requantizedWidth.W)))
}

class Xccel(cfg: XccelConfig) extends Module {
  val io = IO(new XccelIO(cfg))

  private val wbAddrW = cfg.weightBuffer.sram.addrWidth
  private val abAddrW = cfg.activationBuffer.sram.addrWidth
  private val accAddrW = cfg.accumulator.sram.addrWidth

  private val pc = Module(new PC())
  pc.io.start := io.start
  pc.io.stop := io.stop
  io.running := pc.io.running

  private val pcVal = pc.io.pc
  private val numInputsU = cfg.cu.numInputs.U
  private val activationEnd = (cfg.cu.numInputs + cfg.numSamples).U

  private val weightLoadPhase = pc.io.running && pcVal < numInputsU
  private val activationPhase = pc.io.running && pcVal >= numInputsU && pcVal < activationEnd

  private val weightBuffer = Module(new WeightBuffer(cfg.weightBuffer))
  private val activationBuffer = Module(new ActivationBuffer(cfg.activationBuffer))
  private val compute = Module(new SystolicArray(cfg.cu))
  private val accumulator = Module(new Accumulator(cfg.accumulator))
  private val requantizer = Module(new Requantizer(cfg.requantizer))

  // -- weight load: WeightBuffer's shared bus is either the schedule's
  // own read request, or the host's write port, whichever applies
  weightBuffer.io.addr := Mux(weightLoadPhase, pcVal(wbAddrW - 1, 0), io.wbWriteAddr)
  weightBuffer.io.en := Mux(weightLoadPhase, true.B, io.wbWriteEn)
  weightBuffer.io.wen := Mux(weightLoadPhase, false.B, true.B)
  weightBuffer.io.wdata := io.wbWriteData
  weightBuffer.io.shiftAmt := 0.U
  weightBuffer.io.groupShift := 0.U

  compute.weightWriteEn := weightBuffer.io.weightOut.valid
  compute.weightWriteData := weightBuffer.io.weightOut.bits
  compute.weightWriteAddr :=
    ShiftRegister(weightBuffer.io.addr, cfg.weightBuffer.readLatency, 0.U(wbAddrW.W), true.B)

  // -- activation sweep: same host-vs-schedule mux, then a direct
  // connect into the compute unit (shapes match exactly)
  activationBuffer.io.addr := Mux(activationPhase, (pcVal - numInputsU)(abAddrW - 1, 0), io.abWriteAddr)
  activationBuffer.io.en := Mux(activationPhase, true.B, io.abWriteEn)
  activationBuffer.io.wen := Mux(activationPhase, false.B, true.B)
  activationBuffer.io.wdata := io.abWriteData
  activationBuffer.io.shiftAmt := 0.U
  activationBuffer.io.groupShift := 0.U
  activationBuffer.io.shiftIn := true.B

  compute.io.operandIn <> activationBuffer.io.operandOut

  // -- resultOut -> Accumulator: writeAddr is the sample index, recovered
  // by delaying the activation-phase address by the exact fixed latency
  // from an ActivationBuffer read request to its corresponding resultOut
  // beat (abReadLatency + systolic latency) -- same trick
  // WeightBufferSystolicHarness uses for weightWriteAddr, garbage outside
  // the activation window is harmless since it's only ever consumed on
  // cycles Accumulator itself gates by resultIn.valid
  accumulator.io.resultIn <> compute.io.resultOut
  private val sampleIndex = (pcVal - numInputsU)(accAddrW - 1, 0)
  accumulator.io.writeAddr :=
    ShiftRegister(sampleIndex, cfg.activationBuffer.readLatency + cfg.latency, 0.U(accAddrW.W), true.B)

  // -- drain: always available, independent of the schedule (OneR1W)
  accumulator.io.readAddr := io.accReadAddr
  accumulator.io.readEn := io.accReadEn
  accumulator.io.shiftAmt := 0.U
  accumulator.io.groupShift := 0.U

  requantizer.io.in <> accumulator.io.accOut
  io.accumOut <> requantizer.io.out
}
