package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SystolicArrayTest extends AnyFlatSpec with ChiselScalatestTester {

  // two's complement encoding of a (possibly negative) Scala Int into an
  // unsigned literal of the given width, for poking signed values into UInt ports
  def toU(v: Int, width: Int): UInt = (BigInt(v) & ((BigInt(1) << width) - 1)).U(width.W)

  "SystolicArray" should "compute a small matmul with narrower weights than activations (asymmetric quantization)" in {
    // inputWidth=8 (signed activations, incl. a negative value) vs
    // weightWidth=4 (small positive weights only need a few bits) -- proves
    // the compute unit doesn't require operand/weight widths to match
    val cfg = ComputeUnitConfig(numInputs = 3, numOutputs = 2, inputWidth = 8, weightWidth = 4, outputWidth = 20)
    test(new SystolicArray(cfg)) { c =>
      // W (rows x cols) = [[1,2],[3,4],[5,6]], written directly by address --
      // no handshake, the test plays the role of the decoder here
      val weightRows = Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6))
      c.weightWriteEn.poke(true.B)
      for ((row, r) <- weightRows.zipWithIndex) {
        c.weightWriteAddr.poke(r.U)
        for ((v, i) <- row.zipWithIndex) c.weightWriteData(i).poke(toU(v, cfg.weightWidth))
        c.clock.step()
      }
      c.weightWriteEn.poke(false.B)

      // two samples streamed back to back, A (samples x rows) = [[1,1,1],[2,0,-1]]
      val samples = Seq(Seq(1, 1, 1), Seq(2, 0, -1))
      // A . W
      val expected = Seq(Seq(9, 12), Seq(-3, -2))

      c.io.operandIn.valid.poke(true.B)
      for (s <- samples) {
        for ((v, i) <- s.zipWithIndex) c.io.operandIn.bits(i).poke(toU(v, cfg.inputWidth))
        c.clock.step()
      }
      c.io.operandIn.valid.poke(false.B)

      // the first fired sample's result is valid `latency` cycles after it
      // was presented; we already advanced `samples.length` cycles above
      val latency = cfg.numInputs + cfg.numOutputs - 1
      c.clock.step(latency - samples.length)

      for (exp <- expected) {
        c.io.resultOut.valid.expect(true.B)
        for ((v, i) <- exp.zipWithIndex) c.io.resultOut.bits(i).expect(v.S)
        c.clock.step()
      }
    }
  }
}
