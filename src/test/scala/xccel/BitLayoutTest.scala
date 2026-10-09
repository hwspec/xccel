package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

// Exposes a MicroWord's asUInt so its bits can be compared with BitLayout.
class MicroWordPacker(cfg: DecoderConfig) extends Module {
  val io = IO(new Bundle {
    val in = Input(new MicroWord(cfg))
    val out = Output(UInt(new MicroWord(cfg).getWidth.W))
  })
  io.out := io.in.asUInt
}

class BitLayoutTest extends AnyFlatSpec with ChiselScalatestTester {
  "BitLayout" should "place every MicroWord field where asUInt puts it" in {
    val cfg = XccelModuleParams.default().xccelConfig.decoder
    test(new MicroWordPacker(cfg)) { c =>
      val layout = BitLayout.of(new MicroWord(cfg))
      assert(layout.map(_._3).sum == new MicroWord(cfg).getWidth, "fields must tile the word")
      def leaves(d: Data): Seq[Data] = d match {
        case r: Record => r.elements.values.toSeq.flatMap(leaves)
        case v: Vec[_] => v.flatMap(leaves)
        case e if e.getWidth == 0 => Seq() // not in the layout, and not pokeable
        case e => Seq(e)
      }
      def byPath(path: String): Data = path.split('.').foldLeft(c.io.in: Data) {
        case (r: Record, name) => r.elements(name)
        case (v: Vec[_], idx) => v(idx.toInt)
        case (d, _) => d
      }
      leaves(c.io.in).foreach {
        case b: Bool => b.poke(false.B)
        case u: UInt => u.poke(0.U)
        case _ =>
      }
      // set each field alone to all ones and check the packed position
      for ((path, lsb, width) <- layout) {
        val field = byPath(path)
        val ones = (BigInt(1) << width) - 1
        field match {
          case b: Bool => b.poke(true.B)
          case u: UInt => u.poke(ones.U(width.W))
        }
        assert(c.io.out.peek().litValue == (ones << lsb), s"field $path")
        field match {
          case b: Bool => b.poke(false.B)
          case u: UInt => u.poke(0.U)
        }
      }
    }
  }
}
