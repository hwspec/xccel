package xccel

import chisel3._
import chisel3.util._

class SystolicArray(cfg: ComputeUnitConfig) extends ComputeUnit(cfg) with HasStationaryWeights {
  private def rows = cfg.numInputs
  private def cols = cfg.numOutputs

  def weightBeatElems = cols
  def weightBeatWidth = cfg.weightWidth
  def weightTotalBeats = rows

  private val weightRegs = Reg(Vec(2, Vec(rows, Vec(cols, UInt(cfg.weightWidth.W)))))
  when(weightWriteEn) {
    weightRegs(weightWriteSlot.asUInt)(weightWriteAddr) := weightWriteData
  }

  private val rowInputs =
    VecInit(Seq.tabulate(rows)(r => ShiftRegister(io.operandIn.bits(r), r, 0.U(cfg.inputWidth.W), true.B)))

  // weight slot select, skewed and passed along exactly like the activation it belongs to
  private val rowSelects =
    VecInit(Seq.tabulate(rows)(r => ShiftRegister(weightSelect, r, false.B, true.B)))

  private val actRegs = Reg(Vec(rows, Vec(cols, UInt(cfg.inputWidth.W))))
  private val selRegs = Reg(Vec(rows, Vec(cols, Bool())))
  private val psumRegs = Reg(Vec(rows, Vec(cols, SInt(cfg.outputWidth.W))))

  for (r <- 0 until rows) {
    for (c <- 0 until cols) {
      val actHere = if (c == 0) rowInputs(r) else actRegs(r)(c - 1)
      val selHere = if (c == 0) rowSelects(r) else selRegs(r)(c - 1)
      val psumHere = if (r == 0) 0.S(cfg.outputWidth.W) else psumRegs(r - 1)(c)
      actRegs(r)(c) := actHere
      selRegs(r)(c) := selHere
      // slot as a 2:1 mux on static indices (a dynamic index would copy the
      // whole weight type per PE during elaboration)
      val weight = Mux(selHere, weightRegs(1)(r)(c), weightRegs(0)(r)(c))
      psumRegs(r)(c) := psumHere + (actHere.asSInt * weight.asSInt)
    }
  }

  // Raw bottom-row outputs arrive at different cycles (column c lags column 0 by c cycles)
  private val deskewed = VecInit(
    Seq.tabulate(cols)(c => ShiftRegister(psumRegs(rows - 1)(c), cols - 1 - c, 0.S(cfg.outputWidth.W), true.B))
  )
  io.resultOut.bits := deskewed
  io.resultOut.valid := ShiftRegister(io.operandIn.valid, rows + cols - 1, false.B, true.B)
}
