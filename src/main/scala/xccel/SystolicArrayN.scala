package xccel

import chisel3._

// An ArrayWindow of SystolicArrays: n arrays in a (n / kernelCols) x
// kernelCols window over input rows of rowPitch beats (see ArrayWindow).
// Latency rows + cols - 1; each PE row uses its weight one cycle later than
// the row above (row skew).
class SystolicArrayN(
  unitCfg: ComputeUnitConfig,
  n: Int,
  kernelCols: Int,
  rowPitch: Int = 0,
  wideWeights: Boolean = false,
) extends ArrayWindow(unitCfg, n, kernelCols, rowPitch, wideWeights) {
  // cfg is ComputeUnit's (already set when ArrayWindow calls this)
  protected def makeArray(): ComputeUnit with HasStationaryWeights = Module(new SystolicArray(cfg))
}
