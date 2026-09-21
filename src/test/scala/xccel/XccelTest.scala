package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class XccelTest extends AnyFlatSpec with ChiselScalatestTester {

  def toU(v: Int, width: Int): UInt = (BigInt(v) & ((BigInt(1) << width) - 1)).U(width.W)

  "Xccel" should "run one weight-stationary tile end to end: load, compute, and drain the same matmul as the sub-module tests" in {
    // same W (4x2) / samples (2x4) / expected as ActivationBufferTest's
    // integration case: [1,1,1,1].W=[16,20]; [2,0,-1,1].W=[4,6]
    val cu = ComputeUnitConfig(numInputs = 4, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val wbCfg = WeightBufferConfig(cu, numBanks = cu.numOutputs, wordsPerBank = 1, tiles = 1,
      readLatency = 2, writeLatency = 2)
    val abCfg = ActivationBufferConfig(cu, numBanks = cu.numInputs, wordsPerBank = 1, depth = 2,
      readLatency = 1, writeLatency = 1)
    val accCfg = AccumulatorConfig(cu, numBanks = cu.numOutputs, wordsPerBank = 1, depth = 2,
      readLatency = 2, writeLatency = 2)
    // identity requantization (requantizedWidth == outputWidth) so this
    // test isolates wiring/timing correctness from Requantizer's own
    // (separately tested) arithmetic
    val cfg = XccelConfig(cu, wbCfg, abCfg, accCfg, requantizedWidth = cu.outputWidth)

    test(new Xccel(cfg)) { c =>
      val weightRows = Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6), Seq(7, 8))
      val samples = Seq(Seq(1, 1, 1, 1), Seq(2, 0, -1, 1))
      val expected = Seq(Seq(16, 20), Seq(4, 6))

      c.io.start.poke(false.B)
      c.io.stop.poke(false.B)
      c.io.wbWriteEn.poke(false.B)
      c.io.abWriteEn.poke(false.B)
      c.io.accReadEn.poke(false.B)

      // host-load weights (before start -- WeightBuffer's shared bus
      // defaults to this port whenever the schedule isn't using it)
      c.io.wbWriteEn.poke(true.B)
      for ((row, r) <- weightRows.zipWithIndex) {
        c.io.wbWriteAddr.poke(r.U)
        for ((v, i) <- row.zipWithIndex) c.io.wbWriteData(i).poke(toU(v, cu.weightWidth))
        c.clock.step()
      }
      c.io.wbWriteEn.poke(false.B)
      c.clock.step(wbCfg.writeLatency)

      // host-load activations (before start)
      c.io.abWriteEn.poke(true.B)
      for ((s, r) <- samples.zipWithIndex) {
        c.io.abWriteAddr.poke(r.U)
        for ((v, i) <- s.zipWithIndex) c.io.abWriteData(i).poke(toU(v, cu.inputWidth))
        c.clock.step()
      }
      c.io.abWriteEn.poke(false.B)
      c.clock.step(abCfg.writeLatency)

      // run the schedule: pc sweeps the weight-load window then the
      // activation window on its own, no further host intervention
      c.io.start.poke(true.B)
      c.clock.step()
      c.io.start.poke(false.B)
      c.io.running.expect(true.B)

      // generous margin for the whole pipeline (weight load + activation
      // sweep + WeightBuffer/ActivationBuffer read latency + systolic
      // latency) to fully drain into Accumulator
      val latency = cu.numInputs + cu.numOutputs - 1
      val margin = cu.numInputs + abCfg.depth + wbCfg.readLatency + abCfg.readLatency + latency + 4
      c.clock.step(margin)

      // drain: always available, independent of the (now-idle) schedule
      for ((exp, r) <- expected.zipWithIndex) {
        c.io.accReadAddr.poke(r.U)
        c.io.accReadEn.poke(true.B)
        c.clock.step(accCfg.readLatency)
        c.io.accumOut.valid.expect(true.B)
        for ((v, i) <- exp.zipWithIndex) c.io.accumOut.bits(i).expect(toU(v, cu.outputWidth))
      }
      c.io.accReadEn.poke(false.B)
    }
  }
}
