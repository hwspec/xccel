package xccel

import chisel3._

// An ArrayWindow of ReductionTrees: n arrays in a (n / kernelCols) x
// kernelCols window over input rows of rowPitch beats (see ArrayWindow).
// Latency ReductionTree.latency(rows); every weight is used one cycle after
// its operand arrives (no row skew).
class ReductionTreeN(
  unitCfg: ComputeUnitConfig,
  n: Int,
  kernelCols: Int,
  rowPitch: Int = 0,
  wideWeights: Boolean = false,
) extends ArrayWindow(unitCfg, n, kernelCols, rowPitch, wideWeights) {
  // cfg is ComputeUnit's (already set when ArrayWindow calls this)
  protected def makeArray(): ComputeUnit with HasStationaryWeights = Module(new ReductionTree(cfg))
}
