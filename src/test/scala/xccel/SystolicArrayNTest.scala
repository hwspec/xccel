package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SystolicArrayNTest extends AnyFlatSpec with ChiselScalatestTester {

  def toU(v: Int, width: Int): UInt = (BigInt(v) & ((BigInt(1) << width) - 1)).U(width.W)

  "SystolicArrayN" should "sum n arrays over n consecutive operands (a kernel row per pass), with the latency of one array" in {
    val cfg = ComputeUnitConfig(numInputs = 4, numOutputs = 3, inputWidth = 8, weightWidth = 8, outputWidth = 24)
    val n = 3
    val rnd = new scala.util.Random(3)
    val w = Seq.fill(n, cfg.numInputs, cfg.numOutputs)(rnd.nextInt(256) - 128)
    val beats = Seq.fill(9)(Seq.fill(cfg.numInputs)(rnd.nextInt(256) - 128))

    test(new SystolicArrayN(cfg, n, kernelCols = n)) { c =>
      c.weightWriteEn.poke(true.B)
      c.weightWriteSlot.poke(false.B)
      for (k <- 0 until n; r <- 0 until cfg.numInputs) {
        c.weightWriteAddr.poke((k * cfg.numInputs + r).U)
        for (j <- 0 until cfg.numOutputs) c.weightWriteData(j).poke(toU(w(k)(r)(j), cfg.weightWidth))
        c.clock.step()
      }
      c.weightWriteEn.poke(false.B)
      c.weightSelect.poke(false.B)

      val results = scala.collection.mutable.ArrayBuffer.empty[Seq[BigInt]]
      def collect(): Unit =
        if (c.io.resultOut.valid.peek().litToBoolean)
          results += (0 until cfg.numOutputs).map(j => c.io.resultOut.bits(j).peek().litValue)

      c.io.operandIn.valid.poke(true.B)
      for ((b, t) <- beats.zipWithIndex) {
        for ((v, i) <- b.zipWithIndex) c.io.operandIn.bits(i).poke(toU(v, cfg.inputWidth))
        collect()
        c.clock.step()
      }
      c.io.operandIn.valid.poke(false.B)
      for (_ <- 0 until cfg.numInputs + cfg.numOutputs) { collect(); c.clock.step() }

      // one result per beat; from beat n - 1 on it is sum_k W_k . b_{t - (n - 1) + k}
      assert(results.length == beats.length)
      def matvec(v: Seq[Int], m: Seq[Seq[Int]]) = m.head.indices.map(j => v.indices.map(i => v(i) * m(i)(j)).sum)
      for (t <- n - 1 until beats.length) {
        val exp = (0 until n).map(k => matvec(beats(t - (n - 1) + k), w(k))).reduce((a, b) => a.zip(b).map(p => p._1 + p._2))
        assert(results(t) == exp.map(BigInt(_)), s"beat $t")
      }
    }
  }

  it should "compute a plain matmul with only the last array loaded (1x1 layers)" in {
    val cfg = ComputeUnitConfig(numInputs = 2, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    test(new SystolicArrayN(cfg, 3, kernelCols = 3)) { c =>
      val wLast = Seq(Seq(1, 2), Seq(3, 4))
      c.weightWriteEn.poke(true.B)
      for (k <- 0 until 3; r <- 0 until 2) {
        c.weightWriteAddr.poke((k * 2 + r).U)
        for (j <- 0 until 2) c.weightWriteData(j).poke(toU(if (k == 2) wLast(r)(j) else 0, 8))
        c.clock.step()
      }
      c.weightWriteEn.poke(false.B)
      val a = Seq(Seq(1, 1), Seq(2, -1), Seq(5, 7))
      c.io.operandIn.valid.poke(true.B)
      for (s <- a) {
        for ((v, i) <- s.zipWithIndex) c.io.operandIn.bits(i).poke(toU(v, 8))
        c.clock.step()
      }
      c.io.operandIn.valid.poke(false.B)
      c.clock.step(2 + 2 - 1 - a.length)
      for (s <- a) {
        c.io.resultOut.valid.expect(true.B)
        for (j <- 0 until 2) c.io.resultOut.bits(j).expect((s(0) * wLast(0)(j) + s(1) * wLast(1)(j)).S)
        c.clock.step()
      }
    }
  }

  it should "sum a 2 x 2 window of arrays over input rows of rowPitch beats, loading row r of every array per wide weight beat" in {
    val cfg = ComputeUnitConfig(numInputs = 2, numOutputs = 3, inputWidth = 8, weightWidth = 8, outputWidth = 24)
    val (kr, kc, pitch) = (2, 2, 3)
    val n = kr * kc
    val rnd = new scala.util.Random(11)
    val w = Seq.fill(n, cfg.numInputs, cfg.numOutputs)(rnd.nextInt(256) - 128)
    val beats = Seq.fill(12)(Seq.fill(cfg.numInputs)(rnd.nextInt(256) - 128))
    def delay(a: Int) = (kr - 1 - a / kc) * pitch + (kc - 1 - a % kc)

    test(new SystolicArrayN(cfg, n, kernelCols = kc, rowPitch = pitch, wideWeights = true)) { c =>
      c.weightWriteEn.poke(true.B)
      c.weightWriteSlot.poke(false.B)
      for (r <- 0 until cfg.numInputs) {
        c.weightWriteAddr.poke(r.U)
        for (a <- 0 until n; j <- 0 until cfg.numOutputs)
          c.weightWriteData(a * cfg.numOutputs + j).poke(toU(w(a)(r)(j), cfg.weightWidth))
        c.clock.step()
      }
      c.weightWriteEn.poke(false.B)
      c.weightSelect.poke(false.B)

      val results = scala.collection.mutable.ArrayBuffer.empty[Seq[BigInt]]
      def collect(): Unit =
        if (c.io.resultOut.valid.peek().litToBoolean)
          results += (0 until cfg.numOutputs).map(j => c.io.resultOut.bits(j).peek().litValue)
      c.io.operandIn.valid.poke(true.B)
      for (b <- beats) {
        for ((v, i) <- b.zipWithIndex) c.io.operandIn.bits(i).poke(toU(v, cfg.inputWidth))
        collect()
        c.clock.step()
      }
      c.io.operandIn.valid.poke(false.B)
      for (_ <- 0 until cfg.numInputs + cfg.numOutputs) { collect(); c.clock.step() }

      assert(results.length == beats.length)
      def matvec(v: Seq[Int], m: Seq[Seq[Int]]) = m.head.indices.map(j => v.indices.map(i => v(i) * m(i)(j)).sum)
      for (t <- delay(0) until beats.length) {
        val exp = (0 until n).map(a => matvec(beats(t - delay(a)), w(a))).reduce((x, y) => x.zip(y).map(p => p._1 + p._2))
        assert(results(t) == exp.map(BigInt(_)), s"beat $t")
      }
    }
  }
}
