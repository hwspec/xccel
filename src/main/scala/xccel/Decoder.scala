package xccel

import chisel3._
import chisel3.util._

// ADVANCE: offset advances `addr` (a genuinely new row each step, e.g.
// loading new resident rows into a line buffer). RESELECT: offset
// advances `shiftAmt` instead, `addr` stays fixed at baseAddr (re-select
// within whatever is already resident, e.g. sweeping columns of an
// already-loaded window). Both cost the same 1 cycle/step -- the
// distinction is about correctness (does this step's row need to be
// captured into ActivationBuffer's line-buffer history), not throughput.
object DecoderOp extends ChiselEnum {
  val Advance, Reselect = Value
}

// `destination` is deliberately not a field here: today every source has
// exactly one legal destination (WeightBufferRead -> SystolicWeightPort,
// ActivationBufferRead -> SystolicOperandPort, AccumulatorRead ->
// RequantizerIn), so it would carry no information. It would need to be
// reintroduced if a source ever gains more than one legal destination.
object DecoderSource extends ChiselEnum {
  val WeightBufferRead, ActivationBufferRead, AccumulatorRead, Halt = Value
}

class Instruction(cfg: DecoderConfig) extends Bundle {
  val source = DecoderSource()
  val op = DecoderOp()
  val baseAddr = UInt(cfg.maxAddrWidth.W)
  val baseShiftAmt = UInt(cfg.maxShiftWidth.W)
  val groupShift = UInt(cfg.maxGroupShiftWidth.W)
  val length = UInt(cfg.lengthWidth.W)
}

// programLength: how many instructions the program store holds.
// maxRunLength: the largest `length` (steps) any single instruction can
// specify -- sizes the `length` field's width.
case class DecoderConfig(
  weightBuffer: WeightBufferConfig,
  activationBuffer: ActivationBufferConfig,
  accumulator: AccumulatorConfig,
  programLength: Int,
  maxRunLength: Int,
) {
  require(programLength > 0, "programLength must be positive")
  require(maxRunLength > 0, "maxRunLength must be positive")
  require(
    weightBuffer.cu == activationBuffer.cu && activationBuffer.cu == accumulator.cu,
    "all three buffers must share the same ComputeUnitConfig"
  )

  val maxAddrWidth: Int =
    Seq(weightBuffer.sram.addrWidth, activationBuffer.sram.addrWidth, accumulator.sram.addrWidth).max
  val maxShiftWidth: Int =
    Seq(weightBuffer.adapter.shiftWidth, activationBuffer.adapter.shiftWidth, accumulator.adapter.shiftWidth).max
  val maxGroupShiftWidth: Int = Seq(
    weightBuffer.adapter.groupShiftWidth,
    activationBuffer.adapter.groupShiftWidth,
    accumulator.adapter.groupShiftWidth,
  ).max
  val lengthWidth: Int = log2Ceil(maxRunLength + 1)
  val instrIdxWidth: Int = math.max(1, log2Ceil(programLength))

  // fixed per-(source,destination) delay table: today's only legal
  // non-self-timed pair is WeightBufferRead -> SystolicWeightPort (the
  // weight port has no valid of its own -- see HasStationaryWeights).
  // Self-timed destinations (ActivationBufferRead/AccumulatorRead's
  // targets) need no entry here: their own buffer's Valid output already
  // carries correct timing, independent of the decoder.
  val weightPortDelay: Int = weightBuffer.readLatency
}

class DecoderIO(cfg: DecoderConfig) extends Bundle {
  val start = Input(Bool())
  val running = Output(Bool())

  // program load, host-driven before start
  val progWriteEn = Input(Bool())
  val progWriteAddr = Input(UInt(cfg.instrIdxWidth.W))
  val progWriteData = Input(new Instruction(cfg))

  val wbAddr = Output(UInt(cfg.weightBuffer.sram.addrWidth.W))
  val wbEn = Output(Bool())
  val wbShiftAmt = Output(UInt(cfg.weightBuffer.adapter.shiftWidth.W))
  val wbGroupShift = Output(UInt(cfg.weightBuffer.adapter.groupShiftWidth.W))
  // fixed-delay-shadowed copy of wbAddr, for the non-self-timed weight
  // port (compute.weightWriteEn still comes directly from
  // weightBuffer.io.weightOut.valid, not from here)
  val weightWriteAddr = Output(UInt(cfg.weightBuffer.sram.addrWidth.W))

  val abAddr = Output(UInt(cfg.activationBuffer.sram.addrWidth.W))
  val abEn = Output(Bool())
  val abShiftAmt = Output(UInt(cfg.activationBuffer.adapter.shiftWidth.W))
  val abGroupShift = Output(UInt(cfg.activationBuffer.adapter.groupShiftWidth.W))
  val abShiftIn = Output(Bool())

  val accAddr = Output(UInt(cfg.accumulator.sram.addrWidth.W))
  val accEn = Output(Bool())
  val accShiftAmt = Output(UInt(cfg.accumulator.adapter.shiftWidth.W))
  val accGroupShift = Output(UInt(cfg.accumulator.adapter.groupShiftWidth.W))
}

class Decoder(cfg: DecoderConfig) extends Module {
  val io = IO(new DecoderIO(cfg))

  private val prog = Reg(Vec(cfg.programLength, new Instruction(cfg)))
  when(io.progWriteEn) {
    prog(io.progWriteAddr) := io.progWriteData
  }

  private val running = RegInit(false.B)
  private val instrIdx = RegInit(0.U(cfg.instrIdxWidth.W))
  private val offset = RegInit(0.U(cfg.lengthWidth.W))

  private val instr = prog(instrIdx)
  private val isHalt = instr.source === DecoderSource.Halt
  private val atLastStep = offset === (instr.length - 1.U)

  when(io.start) {
    running := true.B
    instrIdx := 0.U
    offset := 0.U
  }.elsewhen(running) {
    when(isHalt) {
      running := false.B
    }.elsewhen(atLastStep) {
      offset := 0.U
      instrIdx := instrIdx + 1.U
    }.otherwise {
      offset := offset + 1.U
    }
  }
  io.running := running

  private val active = running && !isHalt
  private val advancing = instr.op === DecoderOp.Advance
  private val addr = Mux(advancing, instr.baseAddr + offset, instr.baseAddr)
  private val shiftAmt = Mux(advancing, instr.baseShiftAmt, instr.baseShiftAmt + offset)

  private val isWb = active && instr.source === DecoderSource.WeightBufferRead
  private val isAb = active && instr.source === DecoderSource.ActivationBufferRead
  private val isAcc = active && instr.source === DecoderSource.AccumulatorRead

  io.wbAddr := addr(cfg.weightBuffer.sram.addrWidth - 1, 0)
  io.wbEn := isWb
  io.wbShiftAmt := shiftAmt(cfg.weightBuffer.adapter.shiftWidth - 1, 0)
  io.wbGroupShift := instr.groupShift(cfg.weightBuffer.adapter.groupShiftWidth - 1, 0)
  io.weightWriteAddr :=
    ShiftRegister(io.wbAddr, cfg.weightPortDelay, 0.U(cfg.weightBuffer.sram.addrWidth.W), true.B)

  io.abAddr := addr(cfg.activationBuffer.sram.addrWidth - 1, 0)
  io.abEn := isAb
  io.abShiftAmt := shiftAmt(cfg.activationBuffer.adapter.shiftWidth - 1, 0)
  io.abGroupShift := instr.groupShift(cfg.activationBuffer.adapter.groupShiftWidth - 1, 0)
  // ADVANCE: every step is a genuinely new row, always shift it in.
  // RESELECT: only the instruction's LAST step commits the shift (the
  // row at baseAddr becomes resident history right before the next
  // instruction moves past it) -- every earlier step must leave history
  // untouched so they all see the same window, just different columns.
  io.abShiftIn := isAb && (advancing || atLastStep)

  io.accAddr := addr(cfg.accumulator.sram.addrWidth - 1, 0)
  io.accEn := isAcc
  io.accShiftAmt := shiftAmt(cfg.accumulator.adapter.shiftWidth - 1, 0)
  io.accGroupShift := instr.groupShift(cfg.accumulator.adapter.groupShiftWidth - 1, 0)
}
