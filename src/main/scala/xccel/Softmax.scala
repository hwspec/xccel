package xccel

import chisel3._
import chisel3.util._

// Integer softmax (I-BERT style, as in the quantized BraggNN reference's
// softmax_on_cpu) over rows that arrive as several numLanes-wide beats, e.g.
// Accumulator drains of an attention-logit row through the Requantizer.
// A row is processed in three passes over its beats, driven per beat by the
// program:
//
//   Max     max[stat] = max over the row's valid lanes
//   Sum     sum[stat] = sum of iexp(x - max[stat]) over the valid lanes
//   Divide  recip[stat] = K / sum[stat], an iterative divider that is ready
//           divLatency cycles after the Divide op
//   Out     out = clamp(roundShift(iexp(x - max[stat]) * recip[stat], outShift))
//
// `first` starts a row's Max or Sum pass; `lanes` is the number of valid
// lanes of this beat (the last beat of a row may be partial). With any other
// op the unit passes `in` through unchanged, so it sits on the Requantizer
// output for every consumer. `stat` picks one of numStats rows in flight.
//
// iexp follows the reference with int32 semantics (C division by 2^16 of a
// non-negative value, 32-bit wrap):
//   neg = max - x; z = neg * qln2Inv / 2^16; qp = z * qln2 - neg
//   q = (qp + qb)^2 + qc;  q >>= min(z, maxShift)
// The reference then scales by 1 / (sum * out_scale) in floating point;
// here the host supplies K = 2^outShift / out_scale and the unit uses the
// integer K / sum, so results can differ from the float reference in the
// last bit.
//
// Row buffer (rowBeats > 0): the unit keeps up to rowBeats beats of each
// stat's row, so a row is drained from Accumulator once. Three stages, each
// with its own control, can then work on different rows in the same cycle:
//
//   input  op = Max with buf.store: max as above, and the beat is kept in
//          buffer[stat][buf.beat]
//   sum    buf.sumOp = Sum: sum[sumStat] += iexp over buffer[sumStat][sumBeat]
//          (sumFirst starts the row); Divide starts sumStat's divider
//   out    buf.outEn: the output is the normalized buffer[outStat][outBeat]
//          (instead of `in`), on outLanes lanes
//
// A stat's buffer and registers belong to one row until its last Out; the
// program keeps the stages in order per row (Sum after the row's last Max,
// Out from Divide + divLatency).
//
// Combinational except for the per-stat registers, buffers and dividers.
object SoftmaxOp {
  val width: Int = 3
  val None: Int = 0
  val Max: Int = 1
  val Sum: Int = 2
  val Divide: Int = 3
  val Out: Int = 4
}

// the row-buffer sum stage's op
object SoftmaxSumOp {
  val width: Int = 2
  val None: Int = 0
  val Sum: Int = 1
  val Divide: Int = 2
}

case class SoftmaxConfig(numLanes: Int, inWidth: Int, numStats: Int = 1, kWidth: Int = 32, rowBeats: Int = 0) {
  require(numLanes > 0, "numLanes must be positive")
  require(inWidth > 0, "inWidth must be positive")
  require(numStats > 0, "numStats must be positive")
  require(kWidth > 0 && kWidth <= 32, "kWidth must be in 1..32")

  require(rowBeats >= 0, "rowBeats must be non-negative")
  val statWidth: Int = math.max(1, log2Ceil(numStats))
  val lanesWidth: Int = log2Ceil(numLanes + 1)
  val beatWidth: Int = math.max(1, log2Ceil(math.max(rowBeats, 1)))
  val shiftWidth: Int = 6
  // a Divide op at cycle t lets Out ops use the reciprocal from t + divLatency on
  val divLatency: Int = kWidth + 1
  val outMin: BigInt = -(BigInt(1) << (inWidth - 1))
  val outMax: BigInt = (BigInt(1) << (inWidth - 1)) - 1
}

class SoftmaxParams(cfg: SoftmaxConfig) extends Bundle {
  val qln2 = SInt(32.W)
  val qln2Inv = SInt(32.W)
  val qb = SInt(32.W)
  val qc = SInt(32.W)
  val maxShift = UInt(cfg.shiftWidth.W)
  val k = UInt(cfg.kWidth.W)
  val outShift = UInt(cfg.shiftWidth.W)
}

class SoftmaxCtrl(cfg: SoftmaxConfig) extends Bundle {
  val op = UInt(SoftmaxOp.width.W)
  val first = Bool()
  val lanes = UInt(cfg.lanesWidth.W)
  val stat = UInt(cfg.statWidth.W)
  val buf = if (cfg.rowBeats > 0) Some(new SoftmaxBufCtrl(cfg.lanesWidth, cfg.statWidth, cfg.beatWidth)) else None
}

// row-buffer stages (rowBeats > 0), see Softmax
class SoftmaxBufCtrl(lanesWidth: Int, statWidth: Int, beatWidth: Int) extends Bundle {
  val store = Bool()
  val beat = UInt(beatWidth.W)
  val sumOp = UInt(SoftmaxSumOp.width.W)
  val sumStat = UInt(statWidth.W)
  val sumBeat = UInt(beatWidth.W)
  val sumFirst = Bool()
  val sumLanes = UInt(lanesWidth.W)
  val outEn = Bool()
  val outStat = UInt(statWidth.W)
  val outBeat = UInt(beatWidth.W)
  val outLanes = UInt(lanesWidth.W)
}

class SoftmaxIO(cfg: SoftmaxConfig) extends Bundle {
  val in = Flipped(Valid(Vec(cfg.numLanes, UInt(cfg.inWidth.W))))
  val ctrl = Input(new SoftmaxCtrl(cfg))
  val out = Valid(Vec(cfg.numLanes, UInt(cfg.inWidth.W)))

  // parameter load, host-driven before start
  val paramWriteEn = Input(Bool())
  val paramWriteData = Input(new SoftmaxParams(cfg))
}

class Softmax(cfg: SoftmaxConfig) extends Module {
  val io = IO(new SoftmaxIO(cfg))

  private val params = Reg(new SoftmaxParams(cfg))
  when(io.paramWriteEn) {
    params := io.paramWriteData
  }

  // int32 wrap: sign-extend narrow values, keep the low 32 bits of wide ones
  private def i32(x: SInt): SInt = x.pad(32)(31, 0).asSInt

  private val maxReg = Reg(Vec(cfg.numStats, SInt(cfg.inWidth.W)))
  private val sumReg = Reg(Vec(cfg.numStats, SInt(32.W)))

  private val ctrl = io.ctrl
  private val x = io.in.bits.map(_.asSInt)
  private def laneMask(lanes: UInt): Seq[Bool] = Seq.tabulate(cfg.numLanes)(i => i.U < lanes)
  private val valid = laneMask(ctrl.lanes)
  private val rowMax = maxReg(ctrl.stat)

  private def iexp(v: SInt, max: SInt): SInt = {
    val neg = i32(max -& v)
    val z = i32((neg * params.qln2Inv) >> 16)
    val qp = i32(i32(z * params.qln2) - neg)
    val qpb = i32(qp + params.qb)
    val q = i32(i32(qpb * qpb) + params.qc)
    val shift = Mux(z > params.maxShift.zext, params.maxShift, z.asUInt(cfg.shiftWidth - 1, 0))
    q >> shift
  }
  private def maskedSum(exps: Seq[SInt], ok: Seq[Bool]): SInt =
    exps.zip(ok).map { case (e, v) => Mux(v, e, 0.S) }.reduce((a, b) => i32(a + b))

  private val minIn = cfg.outMin.S(cfg.inWidth.W)
  private val beatMax = x.zip(valid).map { case (v, ok) => Mux(ok, v, minIn) }.reduce((a, b) => Mux(a > b, a, b))
  private val exps = x.map(iexp(_, rowMax))
  private val beatSum = maskedSum(exps, valid)

  when(ctrl.op === SoftmaxOp.Max.U) {
    maxReg(ctrl.stat) := Mux(ctrl.first || beatMax > rowMax, beatMax, rowMax)
  }
  when(ctrl.op === SoftmaxOp.Sum.U) {
    sumReg(ctrl.stat) := Mux(ctrl.first, beatSum, i32(sumReg(ctrl.stat) + beatSum))
  }

  // row buffer: input stage stores, sum stage accumulates from it
  private val buf = ctrl.buf
  private val rowBuffer = buf.map(_ => Reg(Vec(cfg.numStats, Vec(cfg.rowBeats, Vec(cfg.numLanes, UInt(cfg.inWidth.W))))))
  for (b <- buf; rb <- rowBuffer) {
    when(ctrl.op === SoftmaxOp.Max.U && b.store) {
      rb(ctrl.stat)(b.beat) := io.in.bits
    }
    val sumExps = rb(b.sumStat)(b.sumBeat).map(v => iexp(v.asSInt, maxReg(b.sumStat)))
    val sumBeat = maskedSum(sumExps, laneMask(b.sumLanes))
    when(b.sumOp === SoftmaxSumOp.Sum.U) {
      sumReg(b.sumStat) := Mux(b.sumFirst, sumBeat, i32(sumReg(b.sumStat) + sumBeat))
    }
  }
  private def divideStarts(s: Int): Bool =
    (ctrl.op === SoftmaxOp.Divide.U && ctrl.stat === s.U) ||
      buf.map(b => b.sumOp === SoftmaxSumOp.Divide.U && b.sumStat === s.U).getOrElse(false.B)

  // one restoring divider per stat: recip = K / sum, one quotient bit per cycle
  private val recip = Seq.tabulate(cfg.numStats) { s =>
    val remainder = Reg(UInt((cfg.kWidth + 1).W))
    val quotient = Reg(UInt(cfg.kWidth.W))
    val dividend = Reg(UInt(cfg.kWidth.W))
    val divisor = Reg(UInt(32.W))
    val left = RegInit(0.U(log2Ceil(cfg.kWidth + 1).W))
    when(divideStarts(s)) {
      remainder := 0.U
      quotient := 0.U
      dividend := params.k
      divisor := sumReg(s).asUInt
      left := cfg.kWidth.U
    }.elsewhen(left =/= 0.U) {
      val shifted = Cat(remainder(cfg.kWidth - 1, 0), dividend(cfg.kWidth - 1))
      val fits = shifted >= divisor
      remainder := Mux(fits, shifted - divisor, shifted)
      quotient := Cat(quotient(cfg.kWidth - 2, 0), fits)
      dividend := dividend << 1
      left := left - 1.U
    }
    quotient
  }

  private def normalize(exps: Seq[SInt], ok: Seq[Bool], r: UInt): Vec[UInt] = VecInit(exps.zip(ok).map { case (e, v) =>
    val product = e * r.zext
    val half = (1.S(2.W) << (params.outShift - 1.U)).asSInt
    val rounded = Mux(params.outShift === 0.U, product, (product + half) >> params.outShift) // product >= 0
    val clamped = Mux(rounded > cfg.outMax.S, cfg.outMax.S, Mux(rounded < cfg.outMin.S, cfg.outMin.S, rounded))
    Mux(v, clamped.asUInt(cfg.inWidth - 1, 0), 0.U(cfg.inWidth.W))
  })

  private val normalized = normalize(exps, valid, VecInit(recip)(ctrl.stat))
  private val passOrOut = Mux(ctrl.op === SoftmaxOp.Out.U, normalized, io.in.bits)

  // out stage: the normalized buffered beat replaces `in` on the output
  private val bufOut = for (b <- buf; rb <- rowBuffer) yield {
    val outExps = rb(b.outStat)(b.outBeat).map(v => iexp(v.asSInt, maxReg(b.outStat)))
    (b.outEn, normalize(outExps, laneMask(b.outLanes), VecInit(recip)(b.outStat)))
  }
  io.out.bits := bufOut.map { case (en, v) => Mux(en, v, passOrOut) }.getOrElse(passOrOut)
  io.out.valid := io.in.valid || bufOut.map(_._1).getOrElse(false.B)
}
