package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class ReductionTreeTest extends AnyFlatSpec with ChiselScalatestTester {

  def toU(v: Int, width: Int): UInt = (BigInt(v) & ((BigInt(1) << width) - 1)).U(width.W)
  def matvec(v: Seq[Int], w: Seq[Seq[Int]]) = w.head.indices.map(j => v.indices.map(i => v(i) * w(i)(j)).sum)

  "ReductionTree" should "compute a matmul like SystolicArray, one result per operand, ReductionTree.latency cycles later" in {
    // 5 rows: an odd count, so one adder level carries a lone value
    val cfg = ComputeUnitConfig(numInputs = 5, numOutputs = 3, inputWidth = 8, weightWidth = 8, outputWidth = 24)
    val rnd = new scala.util.Random(21)
    val w = Seq.fill(cfg.numInputs, cfg.numOutputs)(rnd.nextInt(256) - 128)
    val a = Seq.fill(6)(Seq.fill(cfg.numInputs)(rnd.nextInt(256) - 128))
    test(new ReductionTree(cfg)) { c =>
      c.weightWriteEn.poke(true.B)
      c.weightWriteSlot.poke(false.B)
      for (r <- 0 until cfg.numInputs) {
        c.weightWriteAddr.poke(r.U)
        for (j <- 0 until cfg.numOutputs) c.weightWriteData(j).poke(toU(w(r)(j), 8))
        c.clock.step()
      }
      c.weightWriteEn.poke(false.B)
      c.weightSelect.poke(false.B)

      val latency = ReductionTree.latency(cfg.numInputs)
      val got = scala.collection.mutable.Map.empty[Int, Seq[BigInt]]
      for (t <- 0 until a.length + latency + 2) {
        c.io.operandIn.valid.poke((t < a.length).B)
        if (t < a.length) for ((v, i) <- a(t).zipWithIndex) c.io.operandIn.bits(i).poke(toU(v, 8))
        if (c.io.resultOut.valid.peek().litToBoolean)
          got(t) = (0 until cfg.numOutputs).map(j => c.io.resultOut.bits(j).peek().litValue)
        c.clock.step()
      }
      assert(got.keys.toSeq.sorted == a.indices.map(_ + latency))
      for (t <- a.indices) assert(got(t + latency) == matvec(a(t), w).map(BigInt(_)), s"operand $t")
    }
  }

  it should "use the slot selected per operand, and take a slot's new weights from the cycle after its last operand" in {
    val cfg = ComputeUnitConfig(numInputs = 2, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    test(new ReductionTree(cfg)) { c =>
      def loadRow(slot: Int, r: Int, row: Seq[Int]): Unit = {
        c.weightWriteEn.poke(true.B)
        c.weightWriteSlot.poke((slot == 1).B)
        c.weightWriteAddr.poke(r.U)
        for ((v, i) <- row.zipWithIndex) c.weightWriteData(i).poke(toU(v, 8))
      }
      val w0 = Seq(Seq(1, 2), Seq(3, 4))
      val w1 = Seq(Seq(-1, 0), Seq(0, -1))
      val w0b = Seq(Seq(5, 6), Seq(7, 8))
      for ((row, r) <- w0.zipWithIndex) { loadRow(0, r, row); c.clock.step() }
      for ((row, r) <- w1.zipWithIndex) { loadRow(1, r, row); c.clock.step() }
      c.weightWriteEn.poke(false.B)

      // operands: slot 0, slot 1, then slot 0 again after slot 0 is rewritten
      // with w0b starting the cycle the first operand is presented + 1
      val a = Seq(Seq(1, 1), Seq(2, -1), Seq(5, 7))
      val sel = Seq(0, 1, 0)
      val expected = Seq(matvec(a(0), w0), matvec(a(1), w1), matvec(a(2), w0b)).map(_.map(BigInt(_)))
      val results = scala.collection.mutable.ArrayBuffer.empty[Seq[BigInt]]
      for (t <- 0 until 3 + ReductionTree.latency(2) + 1) {
        c.io.operandIn.valid.poke((t < 3).B)
        if (t < 3) {
          for ((v, i) <- a(t).zipWithIndex) c.io.operandIn.bits(i).poke(toU(v, 8))
          c.weightSelect.poke((sel(t) == 1).B)
        }
        // operand 0 multiplies at cycle 1; rewrite slot 0 at cycles 1 and 2
        if (t == 1 || t == 2) loadRow(0, t - 1, w0b(t - 1)) else c.weightWriteEn.poke(false.B)
        if (c.io.resultOut.valid.peek().litToBoolean)
          results += (0 until 2).map(j => c.io.resultOut.bits(j).peek().litValue)
        c.clock.step()
      }
      assert(results.toSeq == expected)
    }
  }

  "ReductionTreeN" should "sum a 2 x 2 window like SystolicArrayN" in {
    val cfg = ComputeUnitConfig(numInputs = 2, numOutputs = 3, inputWidth = 8, weightWidth = 8, outputWidth = 24)
    val (kc, pitch, n) = (2, 3, 4)
    val rnd = new scala.util.Random(13)
    val w = Seq.fill(n, cfg.numInputs, cfg.numOutputs)(rnd.nextInt(256) - 128)
    val beats = Seq.fill(12)(Seq.fill(cfg.numInputs)(rnd.nextInt(256) - 128))
    def delay(a: Int) = (1 - a / kc) * pitch + (kc - 1 - a % kc)
    test(new ReductionTreeN(cfg, n, kernelCols = kc, rowPitch = pitch, wideWeights = true)) { c =>
      c.weightWriteEn.poke(true.B)
      c.weightWriteSlot.poke(false.B)
      for (r <- 0 until cfg.numInputs) {
        c.weightWriteAddr.poke(r.U)
        for (a <- 0 until n; j <- 0 until cfg.numOutputs)
          c.weightWriteData(a * cfg.numOutputs + j).poke(toU(w(a)(r)(j), 8))
        c.clock.step()
      }
      c.weightWriteEn.poke(false.B)
      c.weightSelect.poke(false.B)
      val results = scala.collection.mutable.ArrayBuffer.empty[Seq[BigInt]]
      for (t <- 0 until beats.length + ReductionTree.latency(2) + 1) {
        c.io.operandIn.valid.poke((t < beats.length).B)
        if (t < beats.length) for ((v, i) <- beats(t).zipWithIndex) c.io.operandIn.bits(i).poke(toU(v, 8))
        if (c.io.resultOut.valid.peek().litToBoolean)
          results += (0 until cfg.numOutputs).map(j => c.io.resultOut.bits(j).peek().litValue)
        c.clock.step()
      }
      assert(results.length == beats.length)
      for (t <- delay(0) until beats.length) {
        val exp = (0 until n).map(a => matvec(beats(t - delay(a)), w(a))).reduce((x, y) => x.zip(y).map(p => p._1 + p._2))
        assert(results(t) == exp.map(BigInt(_)), s"beat $t")
      }
    }
  }
}
