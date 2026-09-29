package xccel

import chisel3._
import chisel3.util._

case class ComputeUnitConfig(numInputs: Int, numOutputs: Int, inputWidth: Int, weightWidth: Int, outputWidth: Int) {
  require(numInputs > 0, "numInputs must be positive")
  require(numOutputs > 0, "numOutputs must be positive")
  require(inputWidth > 0, "inputWidth must be positive")
  require(weightWidth > 0, "weightWidth must be positive")
  require(outputWidth > 0, "outputWidth must be positive")
}

class ComputeUnitIO(cfg: ComputeUnitConfig) extends Bundle {
  val operandIn = Flipped(Valid(Vec(cfg.numInputs, UInt(cfg.inputWidth.W))))
  val resultOut = Valid(Vec(cfg.numOutputs, SInt(cfg.outputWidth.W)))
}

abstract class ComputeUnit(val cfg: ComputeUnitConfig) extends Module {
  val io = IO(new ComputeUnitIO(cfg))
}

trait HasStationaryWeights { this: ComputeUnit =>
  def weightBeatElems: Int
  def weightBeatWidth: Int
  def weightTotalBeats: Int

  private val weightAddrWidth = math.max(1, log2Ceil(weightTotalBeats))
  val weightWriteAddr = IO(Input(UInt(weightAddrWidth.W)))
  val weightWriteEn = IO(Input(Bool()))
  val weightWriteData = IO(Input(Vec(weightBeatElems, UInt(weightBeatWidth.W))))
  // Two weight register slots per PE: writes land in slot weightWriteSlot
  // while the unit computes from the other. weightSelect picks the slot for
  // the operand presented on operandIn in the same cycle and travels with
  // it, so a slot switch never splits one operand across two weight sets.
  val weightWriteSlot = IO(Input(Bool()))
  val weightSelect = IO(Input(Bool()))
}
