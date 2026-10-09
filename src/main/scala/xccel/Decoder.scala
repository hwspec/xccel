package xccel

import chisel3._
import chisel3.util._

// Horizontal microcode: a program is a sequence of fixed-size MicroWords,
// each holding one slot per element the decoder drives, side by side.
// A word is held for `cycles` cycles; within those cycles every strided
// field advances by its stride once per cycle (modulo the field width, so
// an all-ones stride steps backwards by one). Nothing is derived inside
// the decoder -- every cross-element timing relationship (e.g. the weight
// port shadowing a WeightBuffer read by its readLatency) is the program's
// job, expressed by placing the right slot in the right word.

// Microcode slot for a buffer read (WeightBuffer, Accumulator drain).
class ReadSlot(addrWidth: Int, shiftWidth: Int, fanoutWidth: Int) extends Bundle {
  val en = Bool()
  val addr = UInt(addrWidth.W)
  val addrStride = UInt(addrWidth.W)
  val shiftAmt = UInt(shiftWidth.W)
  val shiftStride = UInt(shiftWidth.W)
  val fanout = UInt(fanoutWidth.W)
}

// ActivationBuffer read. The line-buffer shiftIn is disabled along with
// ActivationBuffer's line buffer (im2col is now done by accumulating kn2row).
// // ActivationBuffer read: adds the line-buffer shiftIn, held for the whole
// // word. A RESELECT sweep (shiftIn only on its last step) is therefore two
// // words: cycles-1 cycles with shiftIn=0, then 1 cycle with shiftIn=1.
class ActivationReadSlot(addrWidth: Int, shiftWidth: Int, fanoutWidth: Int)
    extends ReadSlot(addrWidth, shiftWidth, fanoutWidth) {
  // val shiftIn = Bool()
}

class WriteAddrSlot(addrWidth: Int) extends Bundle {
  val addr = UInt(addrWidth.W)
  val addrStride = UInt(addrWidth.W)
}

// Accumulator write side: its strobe is self-timed (resultIn.valid), so
// only the address and whether to add to or overwrite the row.
class AccWriteSlot(addrWidth: Int) extends WriteAddrSlot(addrWidth) {
  val accumulate = Bool()
}

// Buffer write (WeightBuffer, ActivationBuffer): the program decides when
// and where; the data comes from the host's data port in that same cycle.
class WriteSlot(addrWidth: Int) extends WriteAddrSlot(addrWidth) {
  val en = Bool()
}

// WeightBuffer write: the data comes from the host, or, with
// fromRequantizer, from the Requantizer output in that same cycle (an
// activation used as weights, e.g. g in attention . g), or, with
// transposed, from Transposer column `column` (e.g. phi^T in
// theta . phi^T). transposed takes precedence over fromRequantizer.
class WeightWriteSlot(addrWidth: Int, transposerIndexWidth: Int) extends WriteSlot(addrWidth) {
  val fromRequantizer = Bool()
  val transposed = Bool()
  val column = UInt(transposerIndexWidth.W)
  val columnStride = UInt(transposerIndexWidth.W)
}

// Residual add: route the ActivationBuffer beat arriving this cycle
// (abRead + activationBuffer.readLatency) through the residual scaler
// (parameter set paramSet) straight into Accumulator instead of the array.
class ResidualSlot(setWidth: Int) extends Bundle {
  val en = Bool()
  val paramSet = UInt(setWidth.W)
}

// Softmax: what the Softmax unit does with the Requantizer beat of this
// cycle (SoftmaxOp: None passes it through), for which row in flight, and
// with a row buffer (rowBeats > 0) what its sum and out stages do.
class SoftmaxSlot(lanesWidth: Int, statWidth: Int, rowBeats: Int) extends Bundle {
  val op = UInt(SoftmaxOp.width.W)
  val first = Bool()
  val lanes = UInt(lanesWidth.W)
  val stat = UInt(statWidth.W)
  val buf = if (rowBeats > 0) Some(new SoftmaxBufCtrl(lanesWidth, statWidth, math.max(1, log2Ceil(math.max(rowBeats, 1))))) else None
}

// Transposer: capture the Requantizer output of this cycle into row `row`.
class TransposerSlot(indexWidth: Int) extends Bundle {
  val capture = Bool()
  val row = UInt(indexWidth.W)
  val rowStride = UInt(indexWidth.W)
}

// ActivationBuffer write: the data comes from the host, or, with
// fromRequantizer, from the Requantizer output in that same cycle (the
// layer-to-layer path: place it accumulator.readLatency after the accRead
// that drains the row).
class ActivationWriteSlot(addrWidth: Int) extends WriteSlot(addrWidth) {
  val fromRequantizer = Bool()
}

// Weight port: has no valid of its own, so it carries its own strobe, plus
// which of the two weight register slots to write.
class WritePortSlot(addrWidth: Int) extends WriteSlot(addrWidth) {
  val slot = Bool()
}

// Requantizer: which parameter set the beat entering the Requantizer this
// cycle uses (aligned with Accumulator's accOut, i.e. accumulator.readLatency
// after the accRead).
class RequantSlot(setWidth: Int) extends Bundle {
  val paramSet = UInt(setWidth.W)
}

// Compute unit: which weight slot the operand arriving on operandIn this
// cycle uses (HasStationaryWeights.weightSelect).
class ComputeSlot extends Bundle {
  val weightSelect = Bool()
}

class MicroWord(cfg: DecoderConfig) extends Bundle {
  val wbRead = new ReadSlot(
    cfg.weightBuffer.sram.addrWidth,
    cfg.weightBuffer.laneSwitch.shiftWidth,
    cfg.weightBuffer.laneSwitch.fanoutWidth,
  )
  val wbWrite = new WeightWriteSlot(cfg.weightBuffer.sram.addrWidth, cfg.transposerIndexWidth)
  val weightPort = new WritePortSlot(cfg.weightPortAddrWidth)
  val abWrite = new ActivationWriteSlot(cfg.activationBuffer.sram.addrWidth)
  val abRead = new ActivationReadSlot(
    cfg.activationBuffer.sram.addrWidth,
    cfg.activationBuffer.laneSwitch.shiftWidth,
    cfg.activationBuffer.laneSwitch.fanoutWidth,
  )
  val compute = new ComputeSlot
  val accWrite = new AccWriteSlot(cfg.accumulator.addrWidth)
  val accRead = new ReadSlot(
    cfg.accumulator.addrWidth,
    cfg.accumulator.laneSwitch.shiftWidth,
    cfg.accumulator.laneSwitch.fanoutWidth,
  )
  val requant = new RequantSlot(cfg.requantSetWidth)
  val transposer = new TransposerSlot(cfg.transposerIndexWidth)
  val softmax = new SoftmaxSlot(cfg.softmaxLanesWidth, cfg.softmaxStatWidth, cfg.softmaxRowBeats)
  val residual = new ResidualSlot(cfg.residualSetWidth)
  val cycles = UInt(cfg.cyclesWidth.W) // cycles this word is held, must be >= 1
  val halt = Bool() // stop after this word's last cycle
}

// Per-cycle resolved control signals (strides already applied).
class ReadCtrl(addrWidth: Int, shiftWidth: Int, fanoutWidth: Int) extends Bundle {
  val en = Bool()
  val addr = UInt(addrWidth.W)
  val shiftAmt = UInt(shiftWidth.W)
  val fanout = UInt(fanoutWidth.W)
}

class ActivationReadCtrl(addrWidth: Int, shiftWidth: Int, fanoutWidth: Int)
    extends ReadCtrl(addrWidth, shiftWidth, fanoutWidth) {
  // val shiftIn = Bool() // line buffer disabled, see ActivationReadSlot
}

class WriteCtrl(addrWidth: Int) extends Bundle {
  val en = Bool()
  val addr = UInt(addrWidth.W)
}

class WeightWriteCtrl(addrWidth: Int, transposerIndexWidth: Int) extends WriteCtrl(addrWidth) {
  val fromRequantizer = Bool()
  val transposed = Bool()
  val column = UInt(transposerIndexWidth.W)
}

class TransposerCtrl(indexWidth: Int) extends Bundle {
  val capture = Bool()
  val row = UInt(indexWidth.W)
}

class ActivationWriteCtrl(addrWidth: Int) extends WriteCtrl(addrWidth) {
  val fromRequantizer = Bool()
}

class WritePortCtrl(addrWidth: Int) extends WriteCtrl(addrWidth) {
  val slot = Bool()
}

class AccWriteCtrl(addrWidth: Int) extends Bundle {
  val addr = UInt(addrWidth.W)
  val accumulate = Bool()
}

// programLength: how many MicroWords the program store holds.
// maxRunLength: the largest `cycles` any single word can specify.
case class DecoderConfig(
  weightBuffer: WeightBufferConfig,
  activationBuffer: ActivationBufferConfig,
  accumulator: AccumulatorConfig,
  programLength: Int,
  maxRunLength: Int,
  requantParamSets: Int = 1,
  softmaxStats: Int = 1,
  residualParamSets: Int = 2,
  // weight beats the compute unit addresses (HasStationaryWeights.weightTotalBeats)
  weightPortBeats: Int = 0, // 0 = numInputs (SystolicArray)
  softmaxRowBeats: Int = 0, // Softmax row buffer beats (0: none)
) {
  require(programLength > 0, "programLength must be positive")
  require(maxRunLength > 0, "maxRunLength must be positive")
  require(
    weightBuffer.cu == activationBuffer.cu && activationBuffer.cu == accumulator.cu,
    "all three buffers must share the same ComputeUnitConfig"
  )

  // matches HasStationaryWeights' weightWriteAddr
  val weightPortAddrWidth: Int =
    math.max(1, log2Ceil(if (weightPortBeats > 0) weightPortBeats else weightBuffer.cu.numInputs))
  val cyclesWidth: Int = log2Ceil(maxRunLength + 1)
  val requantSetWidth: Int = math.max(1, log2Ceil(requantParamSets))
  // Transposer is numOutputs x numOutputs (one Requantizer beat per row)
  val transposerIndexWidth: Int = math.max(1, log2Ceil(weightBuffer.cu.numOutputs))
  // Softmax works on numOutputs-lane beats
  val softmaxLanesWidth: Int = log2Ceil(weightBuffer.cu.numOutputs + 1)
  val softmaxStatWidth: Int = math.max(1, log2Ceil(softmaxStats))
  val residualSetWidth: Int = math.max(1, log2Ceil(residualParamSets))
  val wordIdxWidth: Int = math.max(1, log2Ceil(programLength))
}

class DecoderIO(cfg: DecoderConfig) extends Bundle {
  val start = Input(Bool())
  val running = Output(Bool())

  // program load, host-driven before start
  val progWriteEn = Input(Bool())
  val progWriteAddr = Input(UInt(cfg.wordIdxWidth.W))
  val progWriteData = Input(new MicroWord(cfg))

  val wbRead = Output(new ReadCtrl(
    cfg.weightBuffer.sram.addrWidth,
    cfg.weightBuffer.laneSwitch.shiftWidth,
    cfg.weightBuffer.laneSwitch.fanoutWidth,
  ))
  val wbWrite = Output(new WeightWriteCtrl(cfg.weightBuffer.sram.addrWidth, cfg.transposerIndexWidth))
  val weightPort = Output(new WritePortCtrl(cfg.weightPortAddrWidth))
  val abWrite = Output(new ActivationWriteCtrl(cfg.activationBuffer.sram.addrWidth))
  val abRead = Output(new ActivationReadCtrl(
    cfg.activationBuffer.sram.addrWidth,
    cfg.activationBuffer.laneSwitch.shiftWidth,
    cfg.activationBuffer.laneSwitch.fanoutWidth,
  ))
  val weightSelect = Output(Bool())
  val requantParamSet = Output(UInt(cfg.requantSetWidth.W))
  val transposer = Output(new TransposerCtrl(cfg.transposerIndexWidth))
  val softmax = Output(new SoftmaxSlot(cfg.softmaxLanesWidth, cfg.softmaxStatWidth, cfg.softmaxRowBeats))
  val residual = Output(new ResidualSlot(cfg.residualSetWidth))
  val accWrite = Output(new AccWriteCtrl(cfg.accumulator.addrWidth))
  val accRead = Output(new ReadCtrl(
    cfg.accumulator.addrWidth,
    cfg.accumulator.laneSwitch.shiftWidth,
    cfg.accumulator.laneSwitch.fanoutWidth,
  ))
}

class Decoder(cfg: DecoderConfig) extends Module {
  val io = IO(new DecoderIO(cfg))

  // Program store: an SRAM (SyncReadMem, one-cycle read). The word for the
  // next cycle is read one cycle ahead (nextIdx), so `word` is always the
  // current one and the timing seen outside is that of a combinational
  // store. The host must not write the program while running.
  private val prog = SyncReadMem(cfg.programLength, new MicroWord(cfg))
  when(io.progWriteEn) {
    prog.write(io.progWriteAddr, io.progWriteData)
  }

  private val running = RegInit(false.B)
  private val wordIdx = RegInit(0.U(cfg.wordIdxWidth.W))
  private val step = RegInit(0.U(cfg.cyclesWidth.W))

  private val nextIdx = Wire(UInt(cfg.wordIdxWidth.W))
  private val word = prog.read(nextIdx)
  private val atLastStep = step === (word.cycles - 1.U)
  nextIdx := Mux(io.start, 0.U, Mux(running && atLastStep, wordIdx + 1.U, wordIdx))

  when(io.start) {
    running := true.B
    wordIdx := 0.U
    step := 0.U
  }.elsewhen(running) {
    when(atLastStep) {
      step := 0.U
      wordIdx := wordIdx + 1.U
      when(word.halt) { running := false.B }
    }.otherwise {
      step := step + 1.U
    }
  }
  io.running := running

  // every strided field restarts from its word's base value on a word
  // boundary; an adder per field instead of a base + step * stride multiply
  private val restart = io.start || (running && atLastStep)
  private def strided(base: UInt, stride: UInt): UInt = {
    val offset = RegInit(0.U(base.getWidth.W))
    when(restart) { offset := 0.U }.elsewhen(running) { offset := offset + stride }
    base + offset
  }

  private def resolveRead(slot: ReadSlot, ctrl: ReadCtrl): Unit = {
    ctrl.en := running && slot.en
    ctrl.addr := strided(slot.addr, slot.addrStride)
    ctrl.shiftAmt := strided(slot.shiftAmt, slot.shiftStride)
    ctrl.fanout := slot.fanout
  }

  resolveRead(word.wbRead, io.wbRead)
  resolveRead(word.abRead, io.abRead)
  // io.abRead.shiftIn := running && word.abRead.en && word.abRead.shiftIn // line buffer disabled
  resolveRead(word.accRead, io.accRead)

  private def resolveWrite(slot: WriteSlot, ctrl: WriteCtrl): Unit = {
    ctrl.en := running && slot.en
    ctrl.addr := strided(slot.addr, slot.addrStride)
  }

  resolveWrite(word.wbWrite, io.wbWrite)
  io.wbWrite.fromRequantizer := word.wbWrite.fromRequantizer
  io.wbWrite.transposed := word.wbWrite.transposed
  io.wbWrite.column := strided(word.wbWrite.column, word.wbWrite.columnStride)
  io.residual.en := running && word.residual.en
  io.residual.paramSet := word.residual.paramSet
  io.softmax := word.softmax
  io.softmax.op := Mux(running, word.softmax.op, SoftmaxOp.None.U)
  for (b <- io.softmax.buf; w <- word.softmax.buf) {
    b.store := running && w.store
    b.sumOp := Mux(running, w.sumOp, SoftmaxSumOp.None.U)
    b.outEn := running && w.outEn
  }
  io.transposer.capture := running && word.transposer.capture
  io.transposer.row := strided(word.transposer.row, word.transposer.rowStride)
  resolveWrite(word.abWrite, io.abWrite)
  io.abWrite.fromRequantizer := word.abWrite.fromRequantizer
  resolveWrite(word.weightPort, io.weightPort)
  io.weightPort.slot := word.weightPort.slot

  io.weightSelect := running && word.compute.weightSelect
  io.requantParamSet := Mux(running, word.requant.paramSet, 0.U)

  io.accWrite.addr := strided(word.accWrite.addr, word.accWrite.addrStride)
  io.accWrite.accumulate := running && word.accWrite.accumulate
}
