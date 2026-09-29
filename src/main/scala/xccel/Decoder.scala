package xccel

import chisel3._
import chisel3.util._

// Horizontal microcode: a program is a sequence of fixed-size MicroWords,
// each holding one slot per element the decoder drives, side by side.
// A word is held for `length` cycles; within those cycles every strided
// field advances by its stride once per cycle (modulo the field width, so
// an all-ones stride steps backwards by one). Nothing is derived inside
// the decoder -- every cross-element timing relationship (e.g. the weight
// port shadowing a WeightBuffer read by its readLatency) is the program's
// job, expressed by placing the right slot in the right word.

// Microcode slot for a buffer read (WeightBuffer, Accumulator drain).
class ReadSlot(addrWidth: Int, shiftWidth: Int, groupShiftWidth: Int) extends Bundle {
  val en = Bool()
  val addr = UInt(addrWidth.W)
  val addrStride = UInt(addrWidth.W)
  val shiftAmt = UInt(shiftWidth.W)
  val shiftStride = UInt(shiftWidth.W)
  val groupShift = UInt(groupShiftWidth.W)
}

// ActivationBuffer read. The line-buffer shiftIn is disabled along with
// ActivationBuffer's line buffer (im2col is now done by accumulating kn2row).
// // ActivationBuffer read: adds the line-buffer shiftIn, held for the whole
// // word. A RESELECT sweep (shiftIn only on its last step) is therefore two
// // words: length-1 cycles with shiftIn=0, then 1 cycle with shiftIn=1.
class ActivationReadSlot(addrWidth: Int, shiftWidth: Int, groupShiftWidth: Int)
    extends ReadSlot(addrWidth, shiftWidth, groupShiftWidth) {
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

// Weight port: has no valid of its own, so it carries its own strobe, plus
// which of the two weight register slots to write.
class WritePortSlot(addrWidth: Int) extends WriteSlot(addrWidth) {
  val slot = Bool()
}

// Compute unit: which weight slot the operand arriving on operandIn this
// cycle uses (HasStationaryWeights.weightSelect).
class ComputeSlot extends Bundle {
  val weightSelect = Bool()
}

class MicroWord(cfg: DecoderConfig) extends Bundle {
  val wbRead = new ReadSlot(
    cfg.weightBuffer.sram.addrWidth,
    cfg.weightBuffer.adapter.shiftWidth,
    cfg.weightBuffer.adapter.groupShiftWidth,
  )
  val wbWrite = new WriteSlot(cfg.weightBuffer.sram.addrWidth)
  val weightPort = new WritePortSlot(cfg.weightPortAddrWidth)
  val abWrite = new WriteSlot(cfg.activationBuffer.sram.addrWidth)
  val abRead = new ActivationReadSlot(
    cfg.activationBuffer.sram.addrWidth,
    cfg.activationBuffer.adapter.shiftWidth,
    cfg.activationBuffer.adapter.groupShiftWidth,
  )
  val compute = new ComputeSlot
  val accWrite = new AccWriteSlot(cfg.accumulator.sram.addrWidth)
  val accRead = new ReadSlot(
    cfg.accumulator.sram.addrWidth,
    cfg.accumulator.adapter.shiftWidth,
    cfg.accumulator.adapter.groupShiftWidth,
  )
  val length = UInt(cfg.lengthWidth.W) // cycles this word is held, must be >= 1
  val halt = Bool() // stop after this word's last cycle
}

// Per-cycle resolved control signals (strides already applied).
class ReadCtrl(addrWidth: Int, shiftWidth: Int, groupShiftWidth: Int) extends Bundle {
  val en = Bool()
  val addr = UInt(addrWidth.W)
  val shiftAmt = UInt(shiftWidth.W)
  val groupShift = UInt(groupShiftWidth.W)
}

class ActivationReadCtrl(addrWidth: Int, shiftWidth: Int, groupShiftWidth: Int)
    extends ReadCtrl(addrWidth, shiftWidth, groupShiftWidth) {
  // val shiftIn = Bool() // line buffer disabled, see ActivationReadSlot
}

class WriteCtrl(addrWidth: Int) extends Bundle {
  val en = Bool()
  val addr = UInt(addrWidth.W)
}

class WritePortCtrl(addrWidth: Int) extends WriteCtrl(addrWidth) {
  val slot = Bool()
}

class AccWriteCtrl(addrWidth: Int) extends Bundle {
  val addr = UInt(addrWidth.W)
  val accumulate = Bool()
}

// programLength: how many MicroWords the program store holds.
// maxRunLength: the largest `length` any single word can specify.
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

  // matches HasStationaryWeights' weightWriteAddr for a compute unit whose
  // weightTotalBeats is numInputs (SystolicArray)
  val weightPortAddrWidth: Int = math.max(1, log2Ceil(weightBuffer.cu.numInputs))
  val lengthWidth: Int = log2Ceil(maxRunLength + 1)
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
    cfg.weightBuffer.adapter.shiftWidth,
    cfg.weightBuffer.adapter.groupShiftWidth,
  ))
  val wbWrite = Output(new WriteCtrl(cfg.weightBuffer.sram.addrWidth))
  val weightPort = Output(new WritePortCtrl(cfg.weightPortAddrWidth))
  val abWrite = Output(new WriteCtrl(cfg.activationBuffer.sram.addrWidth))
  val abRead = Output(new ActivationReadCtrl(
    cfg.activationBuffer.sram.addrWidth,
    cfg.activationBuffer.adapter.shiftWidth,
    cfg.activationBuffer.adapter.groupShiftWidth,
  ))
  val weightSelect = Output(Bool())
  val accWrite = Output(new AccWriteCtrl(cfg.accumulator.sram.addrWidth))
  val accRead = Output(new ReadCtrl(
    cfg.accumulator.sram.addrWidth,
    cfg.accumulator.adapter.shiftWidth,
    cfg.accumulator.adapter.groupShiftWidth,
  ))
}

class Decoder(cfg: DecoderConfig) extends Module {
  val io = IO(new DecoderIO(cfg))

  private val prog = Reg(Vec(cfg.programLength, new MicroWord(cfg)))
  when(io.progWriteEn) {
    prog(io.progWriteAddr) := io.progWriteData
  }

  private val running = RegInit(false.B)
  private val wordIdx = RegInit(0.U(cfg.wordIdxWidth.W))
  private val step = RegInit(0.U(cfg.lengthWidth.W))

  private val word = prog(wordIdx)
  private val atLastStep = step === (word.length - 1.U)

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
    ctrl.groupShift := slot.groupShift
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
  resolveWrite(word.abWrite, io.abWrite)
  resolveWrite(word.weightPort, io.weightPort)
  io.weightPort.slot := word.weightPort.slot

  io.weightSelect := running && word.compute.weightSelect

  io.accWrite.addr := strided(word.accWrite.addr, word.accWrite.addrStride)
  io.accWrite.accumulate := running && word.accWrite.accumulate
}
