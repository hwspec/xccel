package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

// Integer model of the Softmax unit, following the quantized BraggNN
// reference (chia-braggnn-workspace exo/braggnn_reference.py softmax_on_cpu)
// with int32 semantics, except that the final scaling uses the integer
// K / sum instead of a float multiply.
object SoftmaxRef {
  def i32(x: BigInt): BigInt = {
    val m = x & 0xffffffffL
    if (m >= (BigInt(1) << 31)) m - (BigInt(1) << 32) else m
  }

  case class Consts(qln2: Int, qln2Inv: Int, qb: Int, qc: Int, maxShift: Int, k: Long, outShift: Int)

  // same derivation as braggnn_final.c softmax_81 / braggnn_reference.py
  def consts(inScale: Double, outScale: Double, outShift: Int): Consts = {
    val qln2 = (0.693147 / inScale).toInt
    val qln2Inv = 65536 / qln2
    Consts(
      qln2 = qln2,
      qln2Inv = qln2Inv,
      qb = (1.353 / inScale).toInt,
      qc = (0.344 / (0.3585 * inScale * inScale)).toInt,
      maxShift = (255 * qln2Inv) / 65536 + 1, // max_softmax_shift
      k = math.round(math.pow(2, outShift) / outScale),
      outShift = outShift,
    )
  }

  def iexp(c: Consts, x: Int, max: Int): BigInt = {
    val neg = BigInt(max - x)
    val z = i32((neg * c.qln2Inv) >> 16)
    val qp = i32(i32(z * c.qln2) - neg)
    val qpb = i32(qp + c.qb)
    val q = i32(i32(qpb * qpb) + c.qc)
    q >> z.min(c.maxShift).toInt
  }

  // one row -> int8 probabilities scaled by 1 / out_scale
  def row(c: Consts, xs: Seq[Int]): Seq[Int] = {
    val max = xs.max
    val exps = xs.map(iexp(c, _, max))
    val sum = exps.foldLeft(BigInt(0))((a, b) => i32(a + b))
    val recip = BigInt(c.k) / sum
    exps.map { e =>
      val r = (e * recip + (BigInt(1) << (c.outShift - 1))) >> c.outShift
      r.max(-128).min(127).toInt
    }
  }

  // BraggNN's scales (braggnn_data.h SOFTMAX_INPUT_SCALE / SOFTMAX_OUTPUT_SCALE)
  val braggnn: Consts = consts(0.028342675417661667, 0.0010103695094585419, outShift = 16)

  def poke(p: SoftmaxParams, c: Consts): Unit = {
    p.qln2.poke(c.qln2.S)
    p.qln2Inv.poke(c.qln2Inv.S)
    p.qb.poke(c.qb.S)
    p.qc.poke(c.qc.S)
    p.maxShift.poke(c.maxShift.U)
    p.k.poke(c.k.U)
    p.outShift.poke(c.outShift.U)
  }
}

class SoftmaxTest extends AnyFlatSpec with ChiselScalatestTester {

  def toU(v: Int, width: Int): UInt = (BigInt(v) & ((BigInt(1) << width) - 1)).U(width.W)

  "Softmax" should "match the integer reference over rows split into partial beats, two rows in flight" in {
    val cfg = SoftmaxConfig(numLanes = 4, inWidth = 8, numStats = 2)
    val c = SoftmaxRef.braggnn
    val rnd = new scala.util.Random(5)
    val rowLen = 6 // beats of 4 + 2 valid lanes
    val rows = Seq.fill(2, rowLen)(rnd.nextInt(256) - 128)
    val expected = rows.map(SoftmaxRef.row(c, _))
    def beats(r: Int) = rows(r).grouped(cfg.numLanes).toSeq

    test(new Softmax(cfg)) { dut =>
      dut.io.paramWriteEn.poke(true.B)
      SoftmaxRef.poke(dut.io.paramWriteData, c)
      dut.clock.step()
      dut.io.paramWriteEn.poke(false.B)
      dut.io.in.valid.poke(true.B)

      def beat(op: Int, stat: Int, t: Int, values: Seq[Int]): Unit = {
        dut.io.ctrl.op.poke(op.U)
        dut.io.ctrl.stat.poke(stat.U)
        dut.io.ctrl.first.poke((t == 0).B)
        dut.io.ctrl.lanes.poke(values.length.U)
        for (i <- 0 until cfg.numLanes) dut.io.in.bits(i).poke(toU(values.lift(i).getOrElse(0x55), cfg.inWidth))
      }
      // Max and Sum passes, the two rows' beats interleaved
      for (op <- Seq(SoftmaxOp.Max, SoftmaxOp.Sum); t <- beats(0).indices; r <- rows.indices) {
        beat(op, r, t, beats(r)(t))
        dut.clock.step()
      }
      for (r <- rows.indices) {
        beat(SoftmaxOp.Divide, r, 0, Nil)
        dut.clock.step()
      }
      beat(SoftmaxOp.None, 0, 0, Nil)
      dut.clock.step(cfg.divLatency)
      // Out pass: normalized values on valid lanes, 0 on the others
      for (r <- rows.indices; t <- beats(r).indices) {
        beat(SoftmaxOp.Out, r, t, beats(r)(t))
        val exp = expected(r).slice(t * cfg.numLanes, (t + 1) * cfg.numLanes)
        for (i <- 0 until cfg.numLanes) dut.io.out.bits(i).expect(toU(exp.lift(i).getOrElse(0), cfg.inWidth))
        dut.clock.step()
      }
      // any other op passes the input through
      beat(SoftmaxOp.None, 0, 0, Seq(-3, 7, 100, -128))
      for ((v, i) <- Seq(-3, 7, 100, -128).zipWithIndex) dut.io.out.bits(i).expect(toU(v, cfg.inWidth))
    }
  }

  it should "drain each row once with a row buffer, its input, sum and out stages working on different rows in the same cycles" in {
    val cfg = SoftmaxConfig(numLanes = 4, inWidth = 8, numStats = 2, rowBeats = 2)
    val c = SoftmaxRef.braggnn
    val rnd = new scala.util.Random(7)
    val rows = Seq.fill(3, 6)(rnd.nextInt(256) - 128) // beats of 4 + 2 valid lanes
    val expected = rows.map(SoftmaxRef.row(c, _))
    def beats(r: Int) = rows(r).grouped(cfg.numLanes).toSeq

    test(new Softmax(cfg)) { dut =>
      dut.io.paramWriteEn.poke(true.B)
      SoftmaxRef.poke(dut.io.paramWriteData, c)
      dut.clock.step()
      dut.io.paramWriteEn.poke(false.B)
      val b = dut.io.ctrl.buf.get

      // per cycle: (input row, beat), (sum row, beat or Divide), (out row, beat); row r uses stat r % 2
      case class Cycle(in: Option[(Int, Int)] = None, sum: Option[(Int, Int)] = None, div: Option[Int] = None,
                       out: Option[(Int, Int)] = None)
      val d = cfg.divLatency
      // row 0: in 0-1, sum 2-3, divide 4, out 4+d..; row 1: in 2-3 (with row 0's sum), sum 5-6, divide 7;
      // row 2 reuses stat 0 after row 0's last out
      val plan = scala.collection.mutable.Map.empty[Int, Cycle].withDefaultValue(Cycle())
      def at(t: Int)(f: Cycle => Cycle): Unit = plan(t) = f(plan(t))
      at(0)(_.copy(in = Some((0, 0)))); at(1)(_.copy(in = Some((0, 1))))
      at(2)(_.copy(in = Some((1, 0)), sum = Some((0, 0)))); at(3)(_.copy(in = Some((1, 1)), sum = Some((0, 1))))
      at(4)(_.copy(div = Some(0)))
      at(5)(_.copy(sum = Some((1, 0)))); at(6)(_.copy(sum = Some((1, 1)))); at(7)(_.copy(div = Some(1)))
      at(4 + d)(_.copy(out = Some((0, 0)))); at(5 + d)(_.copy(out = Some((0, 1))))
      at(6 + d)(_.copy(in = Some((2, 0)))); at(7 + d)(_.copy(in = Some((2, 1)), out = Some((1, 0))))
      at(8 + d)(_.copy(out = Some((1, 1)), sum = Some((2, 0)))); at(9 + d)(_.copy(sum = Some((2, 1))))
      at(10 + d)(_.copy(div = Some(2)))
      at(10 + 2 * d)(_.copy(out = Some((2, 0)))); at(11 + 2 * d)(_.copy(out = Some((2, 1))))

      for (t <- 0 to plan.keys.max) {
        val cy = plan(t)
        dut.io.in.valid.poke(true.B)
        dut.io.ctrl.op.poke((if (cy.in.nonEmpty) SoftmaxOp.Max else SoftmaxOp.None).U)
        b.store.poke(cy.in.nonEmpty.B)
        for ((r, i) <- cy.in) {
          dut.io.ctrl.stat.poke((r % 2).U)
          dut.io.ctrl.first.poke((i == 0).B)
          dut.io.ctrl.lanes.poke(beats(r)(i).length.U)
          b.beat.poke(i.U)
          for (l <- 0 until cfg.numLanes) dut.io.in.bits(l).poke(toU(beats(r)(i).lift(l).getOrElse(0x55), 8))
        }
        b.sumOp.poke((if (cy.sum.nonEmpty) SoftmaxSumOp.Sum else if (cy.div.nonEmpty) SoftmaxSumOp.Divide
                      else SoftmaxSumOp.None).U)
        for ((r, i) <- cy.sum) {
          b.sumStat.poke((r % 2).U); b.sumBeat.poke(i.U); b.sumFirst.poke((i == 0).B)
          b.sumLanes.poke(beats(r)(i).length.U)
        }
        for (r <- cy.div) b.sumStat.poke((r % 2).U)
        b.outEn.poke(cy.out.nonEmpty.B)
        for ((r, i) <- cy.out) {
          b.outStat.poke((r % 2).U); b.outBeat.poke(i.U); b.outLanes.poke(beats(r)(i).length.U)
          val exp = expected(r).slice(i * cfg.numLanes, (i + 1) * cfg.numLanes)
          for (l <- 0 until cfg.numLanes) dut.io.out.bits(l).expect(toU(exp.lift(l).getOrElse(0), 8))
        }
        dut.clock.step()
      }
    }
  }

  it should "sum to about 1 / out_scale per row (reference sanity check)" in {
    val c = SoftmaxRef.braggnn
    val rnd = new scala.util.Random(9)
    for (_ <- 0 until 20) {
      val probs = SoftmaxRef.row(c, Seq.fill(81)(rnd.nextInt(256) - 128))
      assert(probs.forall(p => p >= 0 && p <= 127))
      // 81 values each rounded, so the total is within 81 * 0.5 of 1 / out_scale ~ 990
      assert(math.abs(probs.sum - 1 / 0.0010103695094585419) <= 41, s"row total ${probs.sum}")
    }
  }
}
