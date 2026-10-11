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
  requantMultWidth: Int = 16,
  requantParamSets: Int = 1,
  softmaxStats: Int = 1,
  residualParamSets: Int = 2,
  // arrays in the compute unit: 1 = SystolicArray (or ReductionTree), n > 1 =
  // SystolicArrayN (or ReductionTreeN), a (n / computeKernelCols) x computeKernelCols window of
  // kernel positions per pass over input rows of computeRowPitch beats, with
  // one array row per weight beat or (computeWideWeights) row r of every array
  computeArrays: Int = 1,
  computeKernelCols: Int = 0, // 0 = computeArrays (one kernel row)
  computeRowPitch: Int = 0,
  computeWideWeights: Boolean = false,
  softmaxRowBeats: Int = 0, // Softmax row buffer (beats per row; 0: three drains per row)
  computeTree: Boolean = false, // ReductionTree(N) instead of SystolicArray(N)
) {
  require(weightBuffer.cu == cu, "weightBuffer must share this Xccel's ComputeUnitConfig")
  require(activationBuffer.cu == cu, "activationBuffer must share this Xccel's ComputeUnitConfig")
  require(accumulator.cu == cu, "accumulator must share this Xccel's ComputeUnitConfig")

  val kernelCols: Int = if (computeKernelCols > 0) computeKernelCols else computeArrays
  val wideWeights: Boolean = computeArrays > 1 && computeWideWeights
  // operand presented -> its result on resultOut
  val computeLatency: Int =
    if (computeTree) ReductionTree.latency(cu.numInputs) else cu.numInputs + cu.numOutputs - 1
  // the compute unit's weight beats: how many, and how many words each
  val weightBeats: Int = if (computeArrays == 1 || wideWeights) cu.numInputs else computeArrays * cu.numInputs
  val weightBeatElems: Int = if (wideWeights) computeArrays * cu.numOutputs else cu.numOutputs
  require(weightBuffer.numWeightLanes == weightBeatElems,
    s"weightBuffer.weightLanes must be the compute unit's weight beat ($weightBeatElems lanes)")

  val requantizer: RequantizerConfig =
    RequantizerConfig(cu.numOutputs, cu.outputWidth, requantizedWidth, requantMultWidth, requantParamSets)

  // Layer-to-layer path (Requantizer output -> ActivationBuffer write) exists
  // only when a requantized value is exactly one ActivationBuffer element and
  // one requantized beat fits in one ActivationBuffer row (spare lanes are
  // written as zero). Otherwise abWrite.fromRequantizer has no effect.
  val feedback: Boolean =
    requantizedWidth == cu.inputWidth && cu.numOutputs <= activationBuffer.totalWordsPerRow

  // Requantizer -> WeightBuffer path (activations used as weights, directly
  // or through the numOutputs x numOutputs Transposer), under the same kind
  // of condition. Otherwise wbWrite.fromRequantizer/transposed have no effect.
  val weightFeedback: Boolean =
    requantizedWidth == cu.weightWidth && cu.numOutputs <= weightBuffer.totalWordsPerRow
  val softmax: SoftmaxConfig = SoftmaxConfig(cu.numOutputs, requantizedWidth, softmaxStats, rowBeats = softmaxRowBeats)

  // Residual-add path: an ActivationBuffer beat goes through a scaler (a
  // Requantizer, int8 in and out: round(a * scale) saturated to int8) into
  // Accumulator without the array, so a + b is two passes plus a drain. It
  // exists only when an ActivationBuffer beat is one Accumulator row
  // (numInputs == numOutputs); otherwise residual.en has no effect.
  val residual: Boolean = cu.numInputs == cu.numOutputs
  val residualScaler: RequantizerConfig =
    RequantizerConfig(cu.numInputs, cu.inputWidth, cu.inputWidth, requantMultWidth, residualParamSets)
  val decoder: DecoderConfig = DecoderConfig(
    weightBuffer, activationBuffer, accumulator, programLength, maxRunLength, requantParamSets, softmaxStats,
    residualParamSets, weightBeats, softmaxRowBeats)
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
  val accReadAddr = Input(UInt(cfg.accumulator.addrWidth.W))
  val accReadEn = Input(Bool())
  // Requantizer parameter set for host drains (while running, the
  // program's requant slot selects it)
  val accReadParamSet = Input(UInt(cfg.requantizer.setWidth.W))
  val accumOut = Valid(Vec(cfg.cu.numOutputs, UInt(cfg.requantizedWidth.W)))

  // residual scaler parameter load, host-driven before start
  val residualParamWriteEn = Input(Bool())
  val residualParamWriteSet = Input(UInt(cfg.residualScaler.setWidth.W))
  val residualParamWriteData = Input(new RequantParams(cfg.residualScaler))

  // Softmax constants, host-driven before start
  val softmaxParamWriteEn = Input(Bool())
  val softmaxParamWriteData = Input(new SoftmaxParams(cfg.softmax))

  // Requantizer parameter load, host-driven before start
  val requantParamWriteEn = Input(Bool())
  val requantParamWriteSet = Input(UInt(cfg.requantizer.setWidth.W))
  val requantParamWriteData = Input(new RequantParams(cfg.requantizer))
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
  private val compute: ComputeUnit with HasStationaryWeights =
    if (cfg.computeArrays > 1 && cfg.computeTree)
      Module(new ReductionTreeN(cfg.cu, cfg.computeArrays, cfg.kernelCols, cfg.computeRowPitch, cfg.wideWeights))
    else if (cfg.computeArrays > 1)
      Module(new SystolicArrayN(cfg.cu, cfg.computeArrays, cfg.kernelCols, cfg.computeRowPitch, cfg.wideWeights))
    else if (cfg.computeTree) Module(new ReductionTree(cfg.cu))
    else Module(new SystolicArray(cfg.cu))
  private val accumulator = Module(new Accumulator(cfg.accumulator))
  private val requantizer = Module(new Requantizer(cfg.requantizer))
  // Softmax sits on the Requantizer output and passes it through unless the
  // program's softmax slot says otherwise; every consumer of requantized
  // values (host drain, AB / WB / Transposer paths) reads `post`.
  private val softmax = Module(new Softmax(cfg.softmax))
  softmax.io.in <> requantizer.io.out
  softmax.io.ctrl.op := decoder.io.softmax.op
  softmax.io.ctrl.first := decoder.io.softmax.first
  softmax.io.ctrl.lanes := decoder.io.softmax.lanes
  softmax.io.ctrl.stat := decoder.io.softmax.stat
  for (b <- softmax.io.ctrl.buf; d <- decoder.io.softmax.buf) b := d
  softmax.io.paramWriteEn := io.softmaxParamWriteEn
  softmax.io.paramWriteData := io.softmaxParamWriteData
  private val post = softmax.io.out

  // -- WeightBuffer: reads from the program; writes from the program while
  // running (data from the host) or from the host before start. With a
  // OneRW SRAM reads and writes share one port (a write in a cycle with a
  // read is dropped); OneR1W/TwoRW give each its own.
  private val wb = decoder.io.wbRead
  weightBuffer.io.readAddr := wb.addr
  weightBuffer.io.readEn := wb.en
  weightBuffer.io.writeAddr := Mux(decoder.io.running, decoder.io.wbWrite.addr, io.wbWriteAddr)
  weightBuffer.io.writeEn := Mux(decoder.io.running, decoder.io.wbWrite.en, io.wbWriteEn)
  // write data: the host's, the Requantizer output, or a Transposer column
  if (cfg.weightFeedback) {
    val transposer = Module(new Transposer(cfg.cu.numOutputs, cfg.cu.weightWidth))
    transposer.io.capture := decoder.io.transposer.capture
    transposer.io.row := decoder.io.transposer.row
    transposer.io.in := post.bits
    transposer.io.column := decoder.io.wbWrite.column
    def pad(v: Vec[UInt]): Vec[UInt] = VecInit(Seq.tabulate(cfg.weightBuffer.totalWordsPerRow)(i =>
      if (i < cfg.cu.numOutputs) v(i) else 0.U(cfg.cu.weightWidth.W)))
    val ww = decoder.io.wbWrite
    weightBuffer.io.wdata := Mux(decoder.io.running && ww.transposed, pad(transposer.io.out),
      Mux(decoder.io.running && ww.fromRequantizer, pad(post.bits), io.wbWriteData))
  } else {
    weightBuffer.io.wdata := io.wbWriteData
  }
  weightBuffer.io.rotate := wb.rotate
  weightBuffer.io.fanout := wb.fanout

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
  // write data: the host's, or the Requantizer output (layer-to-layer path)
  // when the program's abWrite slot asks for it
  if (cfg.feedback) {
    val fromRequantizer = VecInit(Seq.tabulate(cfg.activationBuffer.totalWordsPerRow)(i =>
      if (i < cfg.cu.numOutputs) post.bits(i) else 0.U(cfg.cu.inputWidth.W)))
    activationBuffer.io.wdata :=
      Mux(decoder.io.running && decoder.io.abWrite.fromRequantizer, fromRequantizer, io.abWriteData)
  } else {
    activationBuffer.io.wdata := io.abWriteData
  }
  activationBuffer.io.rotate := ab.rotate
  activationBuffer.io.fanout := ab.fanout
  // activationBuffer.io.shiftIn := ab.shiftIn // line buffer disabled (accumulating kn2row)

  // a beat the program routes to the residual scaler never enters the array
  compute.io.operandIn.bits := activationBuffer.io.operandOut.bits
  compute.io.operandIn.valid := activationBuffer.io.operandOut.valid && !(cfg.residual.B && decoder.io.residual.en)

  // -- resultOut -> Accumulator: the write strobe is resultIn.valid
  // (self-timed), the address and add-vs-overwrite come from the program.
  // An accumulating beat uses Accumulator's read port, so the program
  // must not schedule an accRead in the same cycle.
  if (cfg.residual) {
    val scaler = Module(new Requantizer(cfg.residualScaler))
    scaler.io.in := activationBuffer.io.operandOut
    scaler.io.paramSet := decoder.io.residual.paramSet
    scaler.io.paramWriteEn := io.residualParamWriteEn
    scaler.io.paramWriteSet := io.residualParamWriteSet
    scaler.io.paramWriteData := io.residualParamWriteData
    val scaled = VecInit(scaler.io.out.bits.map(_.asSInt.pad(cfg.cu.outputWidth)))
    accumulator.io.resultIn.valid := Mux(decoder.io.residual.en, scaler.io.out.valid, compute.io.resultOut.valid)
    accumulator.io.resultIn.bits := Mux(decoder.io.residual.en, scaled, compute.io.resultOut.bits)
  } else {
    accumulator.io.resultIn <> compute.io.resultOut
  }
  accumulator.io.writeAddr := decoder.io.accWrite.addr
  accumulator.io.accumulate := decoder.io.accWrite.accumulate

  // -- drain: the program's accRead slot, otherwise the host
  private val acc = decoder.io.accRead
  accumulator.io.readAddr := Mux(acc.en, acc.addr, io.accReadAddr)
  accumulator.io.readEn := acc.en || io.accReadEn
  accumulator.io.rotate := Mux(acc.en, acc.rotate, 0.U)
  accumulator.io.fanout := Mux(acc.en, acc.fanout, 0.U)

  requantizer.io.in <> accumulator.io.accOut
  requantizer.io.paramSet := Mux(decoder.io.running, decoder.io.requantParamSet, io.accReadParamSet)
  requantizer.io.paramWriteEn := io.requantParamWriteEn
  requantizer.io.paramWriteSet := io.requantParamWriteSet
  requantizer.io.paramWriteData := io.requantParamWriteData
  io.accumOut <> post
}
