package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class RequantizerTest extends AnyFlatSpec with ChiselScalatestTester {

  def toU(v: Int, width: Int): UInt = (BigInt(v) & ((BigInt(1) << width) - 1)).U(width.W)

  "Requantizer" should "arithmetic-right-shift each lane to the narrower width, preserving sign and flooring" in {
    val cfg = RequantizerConfig(numLanes = 2, inWidth = 8, outWidth = 4)
    test(new Requantizer(cfg)) { c =>
      // 48 / 16 = 3 exactly; -50 / 16 = -3.125, floors to -4 (arithmetic
      // shift, not truncation-toward-zero)
      val in = Seq(48, -50)
      val expected = Seq(3, -4)

      c.io.in.valid.poke(true.B)
      for ((v, i) <- in.zipWithIndex) c.io.in.bits(i).poke(toU(v, cfg.inWidth))

      c.io.out.valid.expect(true.B)
      for ((v, i) <- expected.zipWithIndex) c.io.out.bits(i).expect(toU(v, cfg.outWidth))
    }
  }

  it should "pass through unchanged when outWidth == inWidth (shiftAmt = 0)" in {
    val cfg = RequantizerConfig(numLanes = 2, inWidth = 8, outWidth = 8)
    test(new Requantizer(cfg)) { c =>
      val in = Seq(7, -7)

      c.io.in.valid.poke(true.B)
      for ((v, i) <- in.zipWithIndex) c.io.in.bits(i).poke(toU(v, cfg.inWidth))

      c.io.out.valid.expect(true.B)
      for ((v, i) <- in.zipWithIndex) c.io.out.bits(i).expect(toU(v, cfg.outWidth))
    }
  }

  it should "requantize INT32 accumulator values down to INT8 (accumulator-scale example)" in {
    // inWidth=32/outWidth=8 is exactly the INT8-activations -> INT32-
    // accumulator -> INT8-next-layer-input scenario discussed for
    // Accumulator: shiftAmt = 32-8 = 24, i.e. divide by 2^24
    val cfg = RequantizerConfig(numLanes = 4, inWidth = 32, outWidth = 8)
    test(new Requantizer(cfg)) { c =>
      val scale = 1 << cfg.shiftAmt // 2^24
      val in = Seq(
        50 * scale, // clean multiple of 2^shiftAmt -> 50
        -70 * scale, // clean negative multiple -> -70
        Int.MaxValue, // 2^31-1 = 127*scale + (scale-1) -> floors to 127 (INT8 max)
        Int.MinValue, // -2^31 = -128*scale exactly -> -128 (INT8 min)
      )
      val expected = Seq(50, -70, 127, -128)

      c.io.in.valid.poke(true.B)
      for ((v, i) <- in.zipWithIndex) c.io.in.bits(i).poke(toU(v, cfg.inWidth))

      c.io.out.valid.expect(true.B)
      for ((v, i) <- expected.zipWithIndex) c.io.out.bits(i).expect(toU(v, cfg.outWidth))
    }
  }
}
