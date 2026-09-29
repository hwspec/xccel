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

  it should "use the weight slot selected per operand, and accept writes to the other slot while computing" in {
    val cfg = ComputeUnitConfig(numInputs = 2, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    test(new SystolicArray(cfg)) { c =>
      def loadRow(slot: Int, r: Int, row: Seq[Int]): Unit = {
        c.weightWriteEn.poke(true.B)
        c.weightWriteSlot.poke((slot == 1).B)
        c.weightWriteAddr.poke(r.U)
        for ((v, i) <- row.zipWithIndex) c.weightWriteData(i).poke(toU(v, cfg.weightWidth))
      }

      val w0 = Seq(Seq(1, 2), Seq(3, 4))
      val w1 = Seq(Seq(-1, 0), Seq(0, -1))
      for ((row, r) <- w0.zipWithIndex) { loadRow(0, r, row); c.clock.step() }
      c.weightWriteEn.poke(false.B)

      // stream 4 operands: [slot0, slot0, slot1, slot1]; slot 1's rows are
      // written during the first two, i.e. while slot 0 is computing
      val a = Seq(Seq(1, 1), Seq(2, -1), Seq(5, 7), Seq(-3, 2))
      val sel = Seq(0, 0, 1, 1)
      // results start coming out while operands are still streaming in
      // (latency 3 < 4 operands), so collect them every cycle
      val results = scala.collection.mutable.ArrayBuffer.empty[Seq[BigInt]]
      def collect(): Unit =
        if (c.io.resultOut.valid.peek().litToBoolean)
          results += (0 until cfg.numOutputs).map(i => c.io.resultOut.bits(i).peek().litValue)

      c.io.operandIn.valid.poke(true.B)
      for (((s, b), t) <- a.zip(sel).zipWithIndex) {
        for ((v, i) <- s.zipWithIndex) c.io.operandIn.bits(i).poke(toU(v, cfg.inputWidth))
        c.weightSelect.poke((b == 1).B)
        if (t < 2) loadRow(1, t, w1(t)) else c.weightWriteEn.poke(false.B)
        collect()
        c.clock.step()
      }
      c.io.operandIn.valid.poke(false.B)
      c.weightWriteEn.poke(false.B)
      for (_ <- 0 until cfg.numInputs + cfg.numOutputs) { collect(); c.clock.step() }

      def matvec(v: Seq[Int], w: Seq[Seq[Int]]) = w.head.indices.map(j => v.indices.map(i => v(i) * w(i)(j)).sum)
      val expected = a.zip(sel).map { case (v, b) => matvec(v, if (b == 0) w0 else w1).map(BigInt(_)) }
      assert(results.toSeq == expected)
    }
  }
}
