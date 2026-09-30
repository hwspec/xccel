package xccel

import chisel3._
import chisel3.util._

// Weight-stationary compute unit with one adder tree per output column
// instead of a systolic grid: the same ComputeUnit / HasStationaryWeights
// interface as SystolicArray (weights written one row of numOutputs per beat,
// two slots per multiplier, weightSelect travelling with its operand), the
// same results, and a latency of
//
//   ReductionTree.latency(numInputs) = 2 + log2Ceil(numInputs)
//
// cycles: an input register, the multipliers (registered), then one
// registered adder level per halving. Every weight of the selected slot is
// used one cycle after its operand is presented, at the multiply (no row
// skew): a pass's weights must all be written by its first operand's
// arrival, and a slot may be written again from its last operand's multiply
// cycle on (a write takes effect the cycle after).
//
// Every operand lane fans out to all numOutputs columns (a broadcast, where
// SystolicArray passes it neighbour to neighbour).
object ReductionTree {
  def latency(numInputs: Int): Int = 2 + log2Ceil(numInputs)
}

class ReductionTree(cfg: ComputeUnitConfig) extends ComputeUnit(cfg) with HasStationaryWeights {
  private def rows = cfg.numInputs
  private def cols = cfg.numOutputs

  def weightBeatElems = cols
  def weightBeatWidth = cfg.weightWidth
  def weightTotalBeats = rows

  private val weightRegs = Reg(Vec(2, Vec(rows, Vec(cols, UInt(cfg.weightWidth.W)))))
  when(weightWriteEn) {
    weightRegs(weightWriteSlot.asUInt)(weightWriteAddr) := weightWriteData
  }

  // input register: the operand and its slot select
  private val act = RegNext(io.operandIn.bits)
  private val sel = RegNext(weightSelect, false.B)

  // multipliers, registered. The slot is a 2:1 mux on statically indexed
  // weights: a dynamic index into weightRegs per multiplier would copy the
  // whole rows x cols type at every access during elaboration
  private val products = RegNext(VecInit(Seq.tabulate(cols) { c =>
    VecInit(Seq.tabulate(rows)(r => act(r).asSInt * Mux(sel, weightRegs(1)(r)(c), weightRegs(0)(r)(c)).asSInt))
  }))

  // one registered adder level per halving, sums kept at outputWidth (wraps
  // like SystolicArray's partial sums)
  private def reduce(level: Seq[SInt]): SInt =
    if (level.length == 1) level.head
    else reduce(level.grouped(2).map(g => RegNext(g.map(_.pad(cfg.outputWidth)).reduce(_ +% _))).toSeq)

  private val sums = Seq.tabulate(cols)(c => reduce(products(c).map(_.pad(cfg.outputWidth))))
  io.resultOut.bits := VecInit(sums.map(_(cfg.outputWidth - 1, 0).asSInt))
  io.resultOut.valid := ShiftRegister(io.operandIn.valid, ReductionTree.latency(rows), false.B, true.B)
}
