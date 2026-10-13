// =============================================================================
// AXI4-Lite bridge for Xccel
// Package: xccel
//
// Register map (offsets relative to the MMIO base, 1 MB window via addr[19:0]):
//
//  0x0000  soft_reset_rw        [RW] write 1: soft reset; reads 1 when done
//  0x0010  start_rw             [RW] write 1: pulse Xccel.start (reads 0)
//  0x0020  status_r             [R]  [0] running, [1] done (halted since the last start)
//  0x0030  cycles_lo_r          [R]  cycles spent running in the last run [31:0]
//  0x0040  cycles_hi_r          [R]  ... [63:32]
//
//  Wide host inputs go through a staging buffer: write its 32-bit words, then
//  write the target index to a commit register, which issues one write of
//  the staged bits (little-endian: stage word 0 = bits [31:0]):
//  0x0100  prog_commit_w        program word index      (MicroWord)
//  0x0110  wb_commit_w          WeightBuffer row        (row data, lane 0 = bits [7:0])
//  0x0120  ab_commit_w          ActivationBuffer row    (row data)
//  0x0130  requant_commit_w     Requantizer set         (RequantParams)
//  0x0140  residual_commit_w    residual scaler set     (RequantParams)
//  0x0150  softmax_commit_w     (value ignored)         (SoftmaxParams)
//  Commits take effect only while Xccel is idle (the host's job to ensure).
//
//  Accumulator drain (through Requantizer and Softmax):
//  0x0200  acc_read_w           write a row index: starts the read
//  0x0210  acc_param_set_rw     Requantizer parameter set for host drains
//  0x0220  acc_result_valid_r   1 once the result of the last acc_read is captured
//
//  0x1000 + i*0x10  stage[i]_rw   staging word i (stage_words words)
//  0x2000 + i*0x10  result[i]_r   captured drain result word i (result_words words)
//
//  params.json also carries the Xccel configuration, and layout.json the bit
//  layout of every staged bundle (MicroWord, RequantParams, SoftmaxParams).
// =============================================================================

package xccel

import axi._
import axi.AxiLiteResp._
import axi.AxiModuleParamsHelper._
import upickle.default._

import chisel3._
import chisel3.util._

case class XccelModuleParams(
  // addresses (suffixed _r, _w, or _rw)
  soft_reset_rw: Long,
  start_rw: Long,
  status_r: Long,
  cycles_lo_r: Long,
  cycles_hi_r: Long,
  prog_commit_w: Long,
  wb_commit_w: Long,
  ab_commit_w: Long,
  requant_commit_w: Long,
  residual_commit_w: Long,
  softmax_commit_w: Long,
  acc_read_w: Long,
  acc_param_set_rw: Long,
  acc_result_valid_r: Long,
  stage_base_rw: Long,
  result_base_r: Long,
  stage_words: Int,
  result_words: Int,
  reset_cycles: Int,
  // Xccel configuration
  rows: Int, // array numInputs (R)
  cols: Int, // array numOutputs (C)
  input_width: Int,
  weight_width: Int,
  output_width: Int,
  requantized_width: Int,
  wb_rows: Int,
  ab_rows: Int,
  acc_rows: Int,
  wb_read_latency: Int,
  wb_write_latency: Int,
  ab_read_latency: Int,
  ab_write_latency: Int,
  acc_read_latency: Int,
  acc_write_latency: Int,
  acc_row_banks: Int, // independent Accumulator row banks (1: one OneR1W SRAM)
  wb_port_type: String, // OneRW, OneR1W or TwoRW
  ab_port_type: String,
  program_length: Int,
  max_run_length: Int,
  requant_mult_width: Int,
  requant_param_sets: Int,
  softmax_stats: Int,
  residual_param_sets: Int,
  compute_arrays: Int, // SystolicArrays in the compute unit (> 1: SystolicArrayN)
  compute_kernel_cols: Int, // SystolicArrayN window columns (0: compute_arrays)
  compute_row_pitch: Int, // SystolicArrayN beats per input row (window rows > 1)
  compute_wide_weights: Boolean, // SystolicArrayN: row r of every array per weight beat
  wb_words_per_bank: Int, // WeightBuffer row = cols banks x wb_words_per_bank words
  softmax_row_beats: Int, // Softmax row buffer beats per row (0: none)
  compute_tree: Boolean, // ReductionTree arrays instead of SystolicArrays
  // largest LaneSwitch fanout per buffer (-1: up to broadcast, 0: unicast only)
  wb_max_fanout: Int,
  ab_max_fanout: Int,
  acc_max_fanout: Int,
  // independent row addresses per buffer: banks / addr_groups consecutive
  // banks share one (1: one address for the whole row)
  wb_addr_groups: Int = 1,
  ab_addr_groups: Int = 1,
  acc_addr_groups: Int = 1,
  // Accumulator row = cols banks x acc_words_per_bank words (> 1: several
  // result beats per row, placed by the accWrite rotate)
  acc_words_per_bank: Int = 1,
) extends AxiModuleParams with AxiModuleDefParams {
  val moduleName = "Xccel"

  private def portType(s: String): BankPortType = s match {
    case "OneRW" => BankPortType.OneRW
    case "OneR1W" => BankPortType.OneR1W
    case "TwoRW" => BankPortType.TwoRW
    case other => throw new IllegalArgumentException(s"unknown port type $other")
  }

  private def maxFanout(n: Int): Option[Int] = if (n < 0) None else Some(n)

  def xccelConfig: XccelConfig = {
    val cu = ComputeUnitConfig(rows, cols, input_width, weight_width, output_width)
    require(wb_rows % rows == 0, "wb_rows must be a multiple of rows (whole weight tiles)")
    XccelConfig(
      cu,
      WeightBufferConfig(cu, numBanks = cols, wordsPerBank = wb_words_per_bank, tiles = wb_rows / rows,
        readLatency = wb_read_latency, writeLatency = wb_write_latency, portType = portType(wb_port_type),
        weightLanes = if (compute_arrays > 1 && compute_wide_weights) compute_arrays * cols else 0,
        maxFanout = maxFanout(wb_max_fanout), addrGroups = wb_addr_groups),
      ActivationBufferConfig(cu, numBanks = rows, wordsPerBank = 1, depth = ab_rows,
        readLatency = ab_read_latency, writeLatency = ab_write_latency, portType = portType(ab_port_type),
        maxFanout = maxFanout(ab_max_fanout), addrGroups = ab_addr_groups),
      AccumulatorConfig(cu, numBanks = cols, wordsPerBank = acc_words_per_bank, depth = acc_rows,
        readLatency = acc_read_latency, writeLatency = acc_write_latency, rowBanks = acc_row_banks,
        maxFanout = maxFanout(acc_max_fanout), addrGroups = acc_addr_groups),
      requantizedWidth = requantized_width,
      programLength = program_length,
      maxRunLength = max_run_length,
      requantMultWidth = requant_mult_width,
      requantParamSets = requant_param_sets,
      softmaxStats = softmax_stats,
      residualParamSets = residual_param_sets,
      computeArrays = compute_arrays,
      computeKernelCols = compute_kernel_cols,
      computeRowPitch = compute_row_pitch,
      computeWideWeights = compute_wide_weights,
      softmaxRowBeats = softmax_row_beats,
      computeTree = compute_tree,
    )
  }
}

object XccelModuleParams {
  implicit val rw: ReadWriter[XccelModuleParams] = macroRW

  // BraggNN scale: a 16 x 16 array
  def default(): XccelModuleParams = XccelModuleParams(
    soft_reset_rw = 0x0000L,
    start_rw = 0x0010L,
    status_r = 0x0020L,
    cycles_lo_r = 0x0030L,
    cycles_hi_r = 0x0040L,
    prog_commit_w = 0x0100L,
    wb_commit_w = 0x0110L,
    ab_commit_w = 0x0120L,
    requant_commit_w = 0x0130L,
    residual_commit_w = 0x0140L,
    softmax_commit_w = 0x0150L,
    acc_read_w = 0x0200L,
    acc_param_set_rw = 0x0210L,
    acc_result_valid_r = 0x0220L,
    stage_base_rw = 0x1000L,
    result_base_r = 0x2000L,
    stage_words = 64,
    result_words = 4,
    reset_cycles = 8,
    rows = 16,
    cols = 16,
    input_width = 8,
    weight_width = 8,
    output_width = 32,
    requantized_width = 8,
    wb_rows = 4096,
    ab_rows = 2048,
    acc_rows = 1024,
    wb_read_latency = 2,
    wb_write_latency = 2,
    ab_read_latency = 1,
    ab_write_latency = 1,
    acc_read_latency = 2,
    acc_write_latency = 2,
    acc_row_banks = 1,
    wb_port_type = "OneR1W",
    ab_port_type = "OneR1W",
    program_length = 4096,
    max_run_length = 256,
    requant_mult_width = 16,
    requant_param_sets = 64,
    softmax_stats = 2,
    residual_param_sets = 2,
    compute_arrays = 1,
    compute_kernel_cols = 0,
    compute_row_pitch = 0,
    compute_wide_weights = false,
    wb_words_per_bank = 1,
    softmax_row_beats = 0,
    compute_tree = false,
    // BraggNN programs read unicast only
    wb_max_fanout = 0,
    ab_max_fanout = 0,
    acc_max_fanout = 0,
  )
}

// Bit layout of a Chisel type as packed by asUInt: path -> (lsb, width).
// A Bundle packs its first declared field in the most significant bits; a
// Vec packs element 0 in the least significant bits. Zero-width fields (e.g.
// a unicast-only fanout) are left out, so setting one is an unknown field.
object BitLayout {
  def of(d: Data): Seq[(String, Int, Int)] = {
    def walk(d: Data, path: String, lsb: Int): Seq[(String, Int, Int)] = d match {
      case r: Record =>
        // Record.elements lists fields in reverse declaration order, which is
        // asUInt's order from the least significant bits up
        var at = lsb
        r.elements.toSeq.flatMap { case (name, child) =>
          val sub = walk(child, if (path.isEmpty) name else s"$path.$name", at)
          at += child.getWidth
          sub
        }
      case v: Vec[_] =>
        var at = lsb
        v.zipWithIndex.flatMap { case (child, i) =>
          val sub = walk(child, s"$path.$i", at)
          at += child.getWidth
          sub
        }
      case e if e.getWidth == 0 => Seq()
      case e => Seq((path, lsb, e.getWidth))
    }
    walk(d, "", 0)
  }

  def json(d: Data): ujson.Obj = ujson.Obj(
    "width" -> d.getWidth,
    "fields" -> ujson.Obj.from(of(d).map { case (p, lsb, w) => p -> ujson.Arr(lsb, w) }),
  )
}

class AxiXccel(p: XccelModuleParams) extends chisel3.Module with axi.HasAxiLite32IO {
  override val S = IO(new AxiLite32IO())

  private val cfg = p.xccelConfig
  require(cfg.feedback && cfg.weightFeedback && cfg.residual, "AxiXccel expects rows == cols with 8-bit data paths")

  // -------------------------------------------------------------------------
  // Soft reset
  // -------------------------------------------------------------------------
  val softResetReg = RegInit(false.B)
  val softResetDoneReg = RegInit(false.B)
  val resetCounterReg = RegInit(0.U(log2Ceil(p.reset_cycles + 1).W))
  when(resetCounterReg > 0.U) {
    resetCounterReg := resetCounterReg - 1.U
  }.otherwise {
    softResetReg := false.B
    softResetDoneReg := true.B
  }
  val combinedReset = softResetReg || reset.asBool

  val dut = withReset(combinedReset) { Module(new Xccel(cfg)) }

  // -------------------------------------------------------------------------
  // Staging buffer and the staged bundles
  // -------------------------------------------------------------------------
  val stageReg = Reg(Vec(p.stage_words, UInt(32.W)))
  val staged = stageReg.asUInt
  private val microWord = new MicroWord(cfg.decoder)
  private val requantParams = new RequantParams(cfg.requantizer)
  private val residualParams = new RequantParams(cfg.residualScaler)
  private val softmaxParams = new SoftmaxParams(cfg.softmax)
  private val wbRow = Vec(cfg.weightBuffer.totalWordsPerRow, UInt(cfg.cu.weightWidth.W))
  private val abRow = Vec(cfg.activationBuffer.totalWordsPerRow, UInt(cfg.cu.inputWidth.W))
  for (t <- Seq(microWord, requantParams, residualParams, softmaxParams, wbRow, abRow))
    require(t.getWidth <= 32 * p.stage_words, s"stage_words too small for a ${t.getWidth}-bit bundle")
  require(cfg.cu.numOutputs * cfg.requantizedWidth <= 32 * p.result_words, "result_words too small")

  AxiXccel.layout = ujson.Obj(
    "microword" -> BitLayout.json(microWord),
    "requant_params" -> BitLayout.json(requantParams),
    "residual_params" -> BitLayout.json(residualParams),
    "softmax_params" -> BitLayout.json(softmaxParams),
    "microword_consts" -> ujson.Obj(
      "softmax_op" -> ujson.Obj("None" -> SoftmaxOp.None, "Max" -> SoftmaxOp.Max, "Sum" -> SoftmaxOp.Sum,
        "Divide" -> SoftmaxOp.Divide, "Out" -> SoftmaxOp.Out),
      "softmax_div_latency" -> cfg.softmax.divLatency,
      "softmax_sum_op" -> ujson.Obj("None" -> SoftmaxSumOp.None, "Sum" -> SoftmaxSumOp.Sum,
        "Divide" -> SoftmaxSumOp.Divide),
      "accumulator_commit_latency" -> cfg.accumulator.commitLatency,
      "compute_latency" -> cfg.computeLatency,
      "compute_tree" -> cfg.computeTree,
    ),
  )

  // one-cycle commit pulses (index latched with them)
  val startPulse = RegInit(false.B)
  val progCommit = RegInit(false.B)
  val wbCommit = RegInit(false.B)
  val abCommit = RegInit(false.B)
  val requantCommit = RegInit(false.B)
  val residualCommit = RegInit(false.B)
  val softmaxCommit = RegInit(false.B)
  val accRead = RegInit(false.B)
  val commitIndex = Reg(UInt(32.W))
  for (pulse <- Seq(startPulse, progCommit, wbCommit, abCommit, requantCommit, residualCommit, softmaxCommit, accRead))
    when(pulse) { pulse := false.B }

  val accParamSetReg = RegInit(0.U(32.W))
  val resultReg = Reg(Vec(p.result_words, UInt(32.W)))
  val resultValidReg = RegInit(false.B)

  dut.io.start := startPulse
  dut.io.progWriteEn := progCommit
  dut.io.progWriteAddr := commitIndex
  dut.io.progWriteData := staged.asTypeOf(microWord)
  dut.io.wbWriteEn := wbCommit
  dut.io.wbWriteAddr := commitIndex
  dut.io.wbWriteData := staged.asTypeOf(wbRow)
  dut.io.abWriteEn := abCommit
  dut.io.abWriteAddr := commitIndex
  dut.io.abWriteData := staged.asTypeOf(abRow)
  dut.io.requantParamWriteEn := requantCommit
  dut.io.requantParamWriteSet := commitIndex
  dut.io.requantParamWriteData := staged.asTypeOf(requantParams)
  dut.io.residualParamWriteEn := residualCommit
  dut.io.residualParamWriteSet := commitIndex
  dut.io.residualParamWriteData := staged.asTypeOf(residualParams)
  dut.io.softmaxParamWriteEn := softmaxCommit
  dut.io.softmaxParamWriteData := staged.asTypeOf(softmaxParams)
  dut.io.accReadEn := accRead
  dut.io.accReadAddr := commitIndex
  dut.io.accReadParamSet := accParamSetReg

  when(dut.io.accumOut.valid && !dut.io.running) {
    resultReg := dut.io.accumOut.bits.asUInt.pad(32 * p.result_words).asTypeOf(resultReg)
    resultValidReg := true.B
  }

  // -------------------------------------------------------------------------
  // Run status and cycle counter
  // -------------------------------------------------------------------------
  val doneReg = RegInit(false.B)
  val cyclesReg = RegInit(0.U(64.W))
  val wasRunning = RegNext(dut.io.running, false.B)
  when(startPulse) {
    doneReg := false.B
    cyclesReg := 0.U
  }.elsewhen(dut.io.running) {
    cyclesReg := cyclesReg + 1.U
  }
  when(wasRunning && !dut.io.running) { doneReg := true.B }

  // -------------------------------------------------------------------------
  // AXI-Lite write path
  // -------------------------------------------------------------------------
  val awHoldValidReg = RegInit(false.B)
  val awHoldAddrReg = Reg(UInt(20.W))
  val wHoldValidReg = RegInit(false.B)
  val wHoldDataReg = Reg(UInt(32.W))
  val wHoldStrbReg = Reg(UInt(4.W))
  val bvalidReg = RegInit(false.B)
  val brespReg = RegInit(0.U(2.W))

  S.AXI.awready := !awHoldValidReg && !bvalidReg
  S.AXI.wready := !wHoldValidReg && !bvalidReg
  when(S.AXI.awvalid && S.AXI.awready) {
    awHoldValidReg := true.B
    awHoldAddrReg := S.AXI.awaddr(19, 0)
  }
  when(S.AXI.wvalid && S.AXI.wready) {
    wHoldValidReg := true.B
    wHoldDataReg := S.AXI.wdata
    wHoldStrbReg := S.AXI.wstrb
  }

  private def inWindow(a: UInt, base: Long, words: Int): Bool = a >= base.U && a < (base + words.toLong * 0x10L).U

  when(awHoldValidReg && wHoldValidReg && !bvalidReg) {
    val a = awHoldAddrReg
    val wdata = wHoldDataReg
    val bresp = WireDefault(OKAY.U)
    def commit(pulse: Bool): Unit = { pulse := true.B; commitIndex := wdata }

    when(wHoldStrbReg =/= "b1111".U) {
      bresp := SLVERR.U
    }.elsewhen(a === p.soft_reset_rw.U) {
      softResetReg := true.B
      resetCounterReg := p.reset_cycles.U
      softResetDoneReg := false.B
    }.elsewhen(a === p.start_rw.U) {
      startPulse := wdata(0)
    }.elsewhen(a === p.prog_commit_w.U) {
      commit(progCommit)
    }.elsewhen(a === p.wb_commit_w.U) {
      commit(wbCommit)
    }.elsewhen(a === p.ab_commit_w.U) {
      commit(abCommit)
    }.elsewhen(a === p.requant_commit_w.U) {
      commit(requantCommit)
    }.elsewhen(a === p.residual_commit_w.U) {
      commit(residualCommit)
    }.elsewhen(a === p.softmax_commit_w.U) {
      commit(softmaxCommit)
    }.elsewhen(a === p.acc_read_w.U) {
      commit(accRead)
      resultValidReg := false.B
    }.elsewhen(a === p.acc_param_set_rw.U) {
      accParamSetReg := wdata
    }.elsewhen(inWindow(a, p.stage_base_rw, p.stage_words)) {
      stageReg((a - p.stage_base_rw.U) >> 4) := wdata
    }.otherwise {
      bresp := SLVERR.U
    }

    brespReg := bresp
    bvalidReg := true.B
    awHoldValidReg := false.B
    wHoldValidReg := false.B
  }
  when(bvalidReg && S.AXI.bready) { bvalidReg := false.B }
  S.AXI.bvalid := bvalidReg
  S.AXI.bresp := brespReg

  // -------------------------------------------------------------------------
  // AXI-Lite read path (all registers, one cycle)
  // -------------------------------------------------------------------------
  val rdataReg = Reg(UInt(32.W))
  val rvalidReg = RegInit(false.B)
  S.AXI.arready := !rvalidReg
  S.AXI.rvalid := rvalidReg
  S.AXI.rdata := rdataReg
  S.AXI.rresp := OKAY.U

  when(S.AXI.arvalid && S.AXI.arready) {
    val a = S.AXI.araddr(19, 0)
    rvalidReg := true.B
    when(a === p.soft_reset_rw.U) {
      rdataReg := softResetDoneReg
    }.elsewhen(a === p.start_rw.U) {
      rdataReg := 0.U
    }.elsewhen(a === p.status_r.U) {
      rdataReg := Cat(doneReg, dut.io.running)
    }.elsewhen(a === p.cycles_lo_r.U) {
      rdataReg := cyclesReg(31, 0)
    }.elsewhen(a === p.cycles_hi_r.U) {
      rdataReg := cyclesReg(63, 32)
    }.elsewhen(a === p.acc_param_set_rw.U) {
      rdataReg := accParamSetReg
    }.elsewhen(a === p.acc_result_valid_r.U) {
      rdataReg := resultValidReg
    }.elsewhen(inWindow(a, p.stage_base_rw, p.stage_words)) {
      rdataReg := stageReg((a - p.stage_base_rw.U) >> 4)
    }.elsewhen(inWindow(a, p.result_base_r, p.result_words)) {
      rdataReg := resultReg((a - p.result_base_r.U) >> 4)
    }.otherwise {
      rdataReg := 0xbad00000L.U | S.AXI.araddr(19, 0)
    }
  }
  when(rvalidReg && S.AXI.rready) { rvalidReg := false.B }
}

// The configuration of the list-scheduled BraggNN program
// (gwtests/braggnn_program_opt.py): two Accumulator row banks, four softmax
// rows in flight. Generated into generated/AxiXccelOpt/.
class AxiXccelOpt(p: XccelModuleParams) extends AxiXccel(p)

object AxiXccelOpt extends App {
  val p = checkParamEnv(XccelModuleParams.default().copy(acc_row_banks = 2, softmax_stats = 4), "XCCEL_MODULE_PARAMS")
  val dir = EmitVerilog.generate(new AxiXccelOpt(p), p)
  os.write.over(os.Path(dir) / "layout.json", ujson.write(AxiXccel.layout, indent = 2))
  println(s"wrote ${os.Path(dir) / "layout.json"}")
}

// AxiXccelOpt with a three-array compute unit (SystolicArrayN): a 3x3 conv
// computes a whole kernel row per pass. Generated into generated/AxiXccelOptN/.
class AxiXccelOptN(p: XccelModuleParams) extends AxiXccel(p)

object AxiXccelOptN extends App {
  val p = checkParamEnv(
    XccelModuleParams.default().copy(acc_row_banks = 2, softmax_stats = 4, compute_arrays = 3), "XCCEL_MODULE_PARAMS")
  val dir = EmitVerilog.generate(new AxiXccelOptN(p), p)
  os.write.over(os.Path(dir) / "layout.json", ujson.write(AxiXccel.layout, indent = 2))
  println(s"wrote ${os.Path(dir) / "layout.json"}")
}

// AxiXccelOpt with a 3 x 3 window of arrays (SystolicArrayN, input rows of
// 9 beats) fed row r of all nine by one 144-word weight beat: a 3x3 conv
// computes its whole kernel per pass. WeightBuffer rows are 16 banks x 9
// words. Generated into generated/AxiXccelOpt9/.
class AxiXccelOpt9(p: XccelModuleParams) extends AxiXccel(p)

object AxiXccelOpt9 extends App {
  val p = checkParamEnv(
    XccelModuleParams.default().copy(acc_row_banks = 2, softmax_stats = 4, compute_arrays = 9, compute_kernel_cols = 3,
      compute_row_pitch = 9, compute_wide_weights = true, wb_words_per_bank = 9, wb_rows = 2048),
    "XCCEL_MODULE_PARAMS")
  val dir = EmitVerilog.generate(new AxiXccelOpt9(p), p)
  os.write.over(os.Path(dir) / "layout.json", ujson.write(AxiXccel.layout, indent = 2))
  println(s"wrote ${os.Path(dir) / "layout.json"}")
}

// AxiXccelOpt9 on 32 x 32 arrays (9,216 PEs). WeightBuffer rows are 32 banks
// x 9 words (2,304 bits), so the staging buffer and drain result grow.
// Generated into generated/AxiXccelOpt9x32/.
class AxiXccelOpt9x32(p: XccelModuleParams) extends AxiXccel(p)

object AxiXccelOpt9x32 extends App {
  val p = checkParamEnv(
    XccelModuleParams.default().copy(rows = 32, cols = 32, stage_words = 80, result_words = 8, acc_row_banks = 2,
      softmax_stats = 4, compute_arrays = 9, compute_kernel_cols = 3, compute_row_pitch = 9,
      compute_wide_weights = true, wb_words_per_bank = 9, wb_rows = 2048),
    "XCCEL_MODULE_PARAMS")
  val dir = EmitVerilog.generate(new AxiXccelOpt9x32(p), p)
  os.write.over(os.Path(dir) / "layout.json", ujson.write(AxiXccel.layout, indent = 2))
  println(s"wrote ${os.Path(dir) / "layout.json"}")
}

// AxiXccelOpt9x32 with a Softmax row buffer (3 beats of 32 lanes per row)
// and eight rows in flight: logits are drained once, and the Sum / Out
// stages read the buffer. Generated into generated/AxiXccelOpt9x32B/.
class AxiXccelOpt9x32B(p: XccelModuleParams) extends AxiXccel(p)

object AxiXccelOpt9x32B extends App {
  val p = checkParamEnv(
    XccelModuleParams.default().copy(rows = 32, cols = 32, stage_words = 80, result_words = 8, acc_row_banks = 2,
      softmax_stats = 8, compute_arrays = 9, compute_kernel_cols = 3, compute_row_pitch = 9,
      compute_wide_weights = true, wb_words_per_bank = 9, wb_rows = 2048, softmax_row_beats = 3),
    "XCCEL_MODULE_PARAMS")
  val dir = EmitVerilog.generate(new AxiXccelOpt9x32B(p), p)
  os.write.over(os.Path(dir) / "layout.json", ujson.write(AxiXccel.layout, indent = 2))
  println(s"wrote ${os.Path(dir) / "layout.json"}")
}

// AxiXccelOpt9x32B with ReductionTree arrays (latency 7 instead of 63).
// Generated into generated/AxiXccelOpt9x32T/.
class AxiXccelOpt9x32T(p: XccelModuleParams) extends AxiXccel(p)

object AxiXccelOpt9x32T extends App {
  val p = checkParamEnv(
    XccelModuleParams.default().copy(rows = 32, cols = 32, stage_words = 80, result_words = 8, acc_row_banks = 2,
      softmax_stats = 8, compute_arrays = 9, compute_kernel_cols = 3, compute_row_pitch = 9,
      compute_wide_weights = true, wb_words_per_bank = 9, wb_rows = 2048, softmax_row_beats = 3, compute_tree = true),
    "XCCEL_MODULE_PARAMS")
  val dir = EmitVerilog.generate(new AxiXccelOpt9x32T(p), p)
  os.write.over(os.Path(dir) / "layout.json", ujson.write(AxiXccel.layout, indent = 2))
  println(s"wrote ${os.Path(dir) / "layout.json"}")
}

// AxiXccelOpt9x32T on 64 x 64 arrays (36,864 multipliers). WeightBuffer
// rows are 64 banks x 9 words (4,608 bits), softmax rows 2 beats of 64 lanes.
// Generated into generated/AxiXccelOpt9x64T/.
class AxiXccelOpt9x64T(p: XccelModuleParams) extends AxiXccel(p)

object AxiXccelOpt9x64T extends App {
  val p = checkParamEnv(
    XccelModuleParams.default().copy(rows = 64, cols = 64, stage_words = 144, result_words = 16, acc_row_banks = 2,
      softmax_stats = 8, compute_arrays = 9, compute_kernel_cols = 3, compute_row_pitch = 9,
      compute_wide_weights = true, wb_words_per_bank = 9, wb_rows = 2048, softmax_row_beats = 2, compute_tree = true),
    "XCCEL_MODULE_PARAMS")
  val dir = EmitVerilog.generate(new AxiXccelOpt9x64T(p), p)
  os.write.over(os.Path(dir) / "layout.json", ujson.write(AxiXccel.layout, indent = 2))
  println(s"wrote ${os.Path(dir) / "layout.json"}")
}

object AxiXccel extends App {
  // set during elaboration, written next to params.json
  var layout: ujson.Obj = ujson.Obj()

  val p = checkParamEnv(XccelModuleParams.default(), "XCCEL_MODULE_PARAMS")
  val dir = EmitVerilog.generate(new AxiXccel(p), p)
  os.write.over(os.Path(dir) / "layout.json", ujson.write(layout, indent = 2))
  println(s"wrote ${os.Path(dir) / "layout.json"}")
}
