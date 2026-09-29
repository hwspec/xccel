package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class RequantizerTest extends AnyFlatSpec with ChiselScalatestTester {

  def toU(v: BigInt, width: Int): UInt = (v & ((BigInt(1) << width) - 1)).U(width.W)

  // reference: clamp(roundShift((x + bias) * mult, shift), lo, hi), rounding half away from zero
  def reference(x: BigInt, mult: Int, shift: Int, lo: BigInt, hi: BigInt, bias: BigInt = 0): BigInt = {
    val p = (x + bias) * mult
    val r =
      if (shift == 0) p
      else {
        val half = BigInt(1) << (shift - 1)
        (if (p < 0) p + half - 1 else p + half) >> shift // BigInt >> floors, like SInt >>
      }
    r.max(lo).min(hi)
  }

  def load(
    c: Requantizer,
    cfg: RequantizerConfig,
    set: Int,
    mult: Seq[Int],
    shift: Seq[Int],
    lo: BigInt,
    hi: BigInt,
    bias: Seq[BigInt] = Nil,
  ): Unit = {
    c.io.paramWriteEn.poke(true.B)
    c.io.paramWriteSet.poke(set.U)
    for (i <- 0 until cfg.numLanes) {
      c.io.paramWriteData.bias(i).poke(bias.lift(i).getOrElse(BigInt(0)).S(cfg.inWidth.W))
      c.io.paramWriteData.mult(i).poke(mult(i).U)
      c.io.paramWriteData.shift(i).poke(shift(i).U)
    }
    c.io.paramWriteData.lo.poke(lo.S)
    c.io.paramWriteData.hi.poke(hi.S)
    c.clock.step()
    c.io.paramWriteEn.poke(false.B)
  }

  def check(c: Requantizer, cfg: RequantizerConfig, set: Int, in: Seq[BigInt], expected: Seq[BigInt]): Unit = {
    c.io.paramSet.poke(set.U)
    c.io.in.valid.poke(true.B)
    for ((v, i) <- in.zipWithIndex) c.io.in.bits(i).poke(toU(v, cfg.inWidth))
    c.io.out.valid.expect(true.B)
    for ((v, i) <- expected.zipWithIndex) c.io.out.bits(i).expect(toU(v, cfg.outWidth))
  }

  "Requantizer" should "pass values through unchanged after reset (identity parameters) when outWidth == inWidth" in {
    val cfg = RequantizerConfig(numLanes = 2, inWidth = 8, outWidth = 8)
    test(new Requantizer(cfg)) { c =>
      c.io.paramWriteEn.poke(false.B)
      check(c, cfg, 0, Seq(7, -7), Seq(7, -7))
    }
  }

  it should "scale each lane by its own mult / 2^shift, rounding half away from zero" in {
    val cfg = RequantizerConfig(numLanes = 4, inWidth = 20, outWidth = 8)
    test(new Requantizer(cfg)) { c =>
      // lane scales: 3/4, 1/2, 1/16, 5/1
      val mult = Seq(3, 1, 1, 5)
      val shift = Seq(2, 1, 4, 0)
      load(c, cfg, 0, mult, shift, cfg.outMin, cfg.outMax)
      // ties in both signs: 2 * 3/4 = 1.5 -> 2, -2 * 3/4 = -1.5 -> -2,
      // 5/2 = 2.5 -> 3, -5/2 = -2.5 -> -3; 40/16 = 2.5 -> 3; -24/16 = -1.5 -> -2
      for (in <- Seq(Seq(2, 5, 40, 7), Seq(-2, -5, -24, -7), Seq(10, -3, 7, 0))) {
        val xs = in.map(BigInt(_))
        check(c, cfg, 0, xs, xs.indices.map(i => reference(xs(i), mult(i), shift(i), cfg.outMin, cfg.outMax)))
      }
      check(c, cfg, 0, Seq(2, 5, 40, 7).map(BigInt(_)), Seq(2, 3, 3, 35).map(BigInt(_)))
      check(c, cfg, 0, Seq(-2, -5, -24, -7).map(BigInt(_)), Seq(-2, -3, -2, -35).map(BigInt(_)))
    }
  }

  it should "saturate to [lo, hi], and give ReLU with lo = 0" in {
    val cfg = RequantizerConfig(numLanes = 3, inWidth = 32, outWidth = 8, numParamSets = 2)
    test(new Requantizer(cfg)) { c =>
      // set 0: full INT8 range; set 1: ReLU (lo = 0)
      load(c, cfg, 0, Seq(1, 1, 1), Seq(0, 0, 0), -128, 127)
      load(c, cfg, 1, Seq(1, 1, 1), Seq(0, 0, 0), 0, 127)
      val in = Seq(BigInt(1000), BigInt(-1000), BigInt(-5))
      check(c, cfg, 0, in, Seq(127, -128, -5).map(BigInt(_)))
      check(c, cfg, 1, in, Seq(127, 0, 0).map(BigInt(_)))
    }
  }

  it should "select the parameter set per beat, independently of the others' contents" in {
    val cfg = RequantizerConfig(numLanes = 2, inWidth = 20, outWidth = 8, numParamSets = 3)
    test(new Requantizer(cfg)) { c =>
      load(c, cfg, 1, Seq(1, 1), Seq(1, 1), -128, 127) // / 2
      load(c, cfg, 2, Seq(3, 3), Seq(0, 0), -128, 127) // * 3
      val in = Seq(BigInt(9), BigInt(-9))
      check(c, cfg, 0, in, Seq(9, -9).map(BigInt(_))) // set 0 left at identity
      check(c, cfg, 1, in, Seq(5, -5).map(BigInt(_))) // 4.5 -> 5, -4.5 -> -5
      check(c, cfg, 2, in, Seq(27, -27).map(BigInt(_)))
    }
  }

  it should "add the per-lane bias before scaling, in the accumulator's scale" in {
    val cfg = RequantizerConfig(numLanes = 3, inWidth = 32, outWidth = 8)
    test(new Requantizer(cfg)) { c =>
      val bias = Seq(BigInt(100), BigInt(-100), BigInt(-2000000000)) // lane 2: near the INT32 limit
      val mult = Seq(1, 3, 1)
      val shift = Seq(3, 2, 0)
      load(c, cfg, 0, mult, shift, cfg.outMin, cfg.outMax, bias)
      // lane 0: (4 + 100) / 8 = 13; lane 1: (-6 - 100) * 3/4 = -79.5 -> -80;
      // lane 2: in + bias below INT32 min must not wrap before saturating to -128
      val in = Seq(BigInt(4), BigInt(-6), BigInt(-2000000000))
      check(c, cfg, 0, in, Seq(13, -80, -128).map(BigInt(_)))
      check(c, cfg, 0, in, in.indices.map(i => reference(in(i), mult(i), shift(i), cfg.outMin, cfg.outMax, bias(i))))
    }
  }
}
