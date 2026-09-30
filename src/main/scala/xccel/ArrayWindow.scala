package xccel

import chisel3._
import chisel3.util._

// n weight-stationary arrays (made by makeArray) behind one ComputeUnit interface,
// arranged as a kernelRows x kernelCols window (n = kernelRows * kernelCols,
// array a = i * kernelCols + j). Every operand beat goes to all of them, and
// array (i, j)'s results are delayed
//
//   d(i, j) = (kernelRows - 1 - i) * rowPitch + (kernelCols - 1 - j)
//
// cycles and summed, so the result that leaves with beat t is
//
//   sum_{i,j}  W_(i,j) . b_{t - d(i,j)}
//
// For an input streamed raster order with rowPitch beats per input row
// (rows narrower than rowPitch padded with idle cycles), that is a whole
// kernelRows x kernelCols window of a conv per pass, the result of output
// pixel (y, x) leaving with the beat of input (y + kernelRows - 1,
// x + kernelCols - 1): kn2row with the output shift done inside the unit.
// The last array is not delayed, so the latency is that of one array
// (SystolicArray: rows + cols - 1, ReductionTree: 2 + log2Ceil(rows)). Results whose window leaves the input (a stream's
// first beats, windows crossing a row end) mix unrelated beats; the program
// discards them. Cycles without an operand feed zeros.
//
// A 1x1 layer loads only the last array and needs zeros in the others on
// its operand's nonzero lanes; the maxDelay cycles before its first beat
// must not carry beats that met nonzero weights there.
//
// Weight beats:
//   wideWeights = false  one array row per beat, addressed a * rows + r
//   wideWeights = true   row r of every array per beat (array a's weights
//                        in elements a * cols until (a + 1) * cols), addressed r
// weightWriteSlot / weightSelect apply to every array alike.
//
// SystolicArrayN and ReductionTreeN are the two kinds of array.
abstract class ArrayWindow(
  cfg: ComputeUnitConfig,
  val n: Int,
  val kernelCols: Int,
  val rowPitch: Int,
  val wideWeights: Boolean,
) extends ComputeUnit(cfg) with HasStationaryWeights {
  require(n > 0 && kernelCols > 0 && n % kernelCols == 0, "n must be a positive multiple of kernelCols")
  val kernelRows: Int = n / kernelCols
  require(kernelRows == 1 || rowPitch >= kernelCols, "rowPitch must cover a kernel row")
  private def rows = cfg.numInputs
  private def cols = cfg.numOutputs

  // one array of the window (called n times during construction)
  protected def makeArray(): ComputeUnit with HasStationaryWeights

  def delay(a: Int): Int = (kernelRows - 1 - a / kernelCols) * rowPitch + (kernelCols - 1 - a % kernelCols)
  val maxDelay: Int = delay(0)

  def weightBeatElems = if (wideWeights) n * cols else cols
  def weightBeatWidth = cfg.weightWidth
  def weightTotalBeats = if (wideWeights) rows else n * rows

  private val operand = VecInit(io.operandIn.bits.map(v => Mux(io.operandIn.valid, v, 0.U)))
  private val arrays = Seq.fill(n)(makeArray())
  for ((a, k) <- arrays.zipWithIndex) {
    a.io.operandIn.valid := io.operandIn.valid
    a.io.operandIn.bits := operand
    if (wideWeights) {
      a.weightWriteEn := weightWriteEn
      a.weightWriteAddr := weightWriteAddr
      a.weightWriteData := VecInit(Seq.tabulate(cols)(c => weightWriteData(k * cols + c)))
    } else {
      a.weightWriteEn := weightWriteEn && weightWriteAddr / rows.U === k.U
      a.weightWriteAddr := weightWriteAddr % rows.U
      a.weightWriteData := weightWriteData
    }
    a.weightWriteSlot := weightWriteSlot
    a.weightSelect := weightSelect
  }

  private val delayed = arrays.zipWithIndex.map { case (a, k) => ShiftRegister(a.io.resultOut.bits, delay(k)) }
  io.resultOut.bits := VecInit(Seq.tabulate(cols)(c => delayed.map(_(c)).reduce(_ + _)))
  io.resultOut.valid := arrays.last.io.resultOut.valid
}
