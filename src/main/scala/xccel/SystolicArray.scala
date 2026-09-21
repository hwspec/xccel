package xccel

import chisel3._
import chisel3.util._

class SystolicArray(cfg: ComputeUnitConfig) extends ComputeUnit(cfg) with HasStationaryWeights {
  private def rows = cfg.numInputs
  private def cols = cfg.numOutputs

  def weightBeatElems = cols
  def weightBeatWidth = cfg.weightWidth
  def weightTotalBeats = rows

  private val weightRegs = Reg(Vec(rows, Vec(cols, UInt(cfg.weightWidth.W))))
  when(weightWriteEn) {
    weightRegs(weightWriteAddr) := weightWriteData
  }

  private val rowInputs =
    VecInit(Seq.tabulate(rows)(r => ShiftRegister(io.operandIn.bits(r), r, 0.U(cfg.inputWidth.W), true.B)))

  private val actRegs = Reg(Vec(rows, Vec(cols, UInt(cfg.inputWidth.W))))
  private val psumRegs = Reg(Vec(rows, Vec(cols, SInt(cfg.outputWidth.W))))

  for (r <- 0 until rows) {
    for (c <- 0 until cols) {
      val actHere = if (c == 0) rowInputs(r) else actRegs(r)(c - 1)
      val psumHere = if (r == 0) 0.S(cfg.outputWidth.W) else psumRegs(r - 1)(c)
      actRegs(r)(c) := actHere
      psumRegs(r)(c) := psumHere + (actHere.asSInt * weightRegs(r)(c).asSInt)
    }
  }

  // Raw bottom-row outputs arrive at different cycles (column c lags column 0 by c cycles)
  private val deskewed = VecInit(
    Seq.tabulate(cols)(c => ShiftRegister(psumRegs(rows - 1)(c), cols - 1 - c, 0.S(cfg.outputWidth.W), true.B))
  )
  io.resultOut.bits := deskewed
  io.resultOut.valid := ShiftRegister(io.operandIn.valid, rows + cols - 1, false.B, true.B)
}
