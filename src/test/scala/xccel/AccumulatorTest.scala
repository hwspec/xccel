package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class AccumulatorTest extends AnyFlatSpec with ChiselScalatestTester {

  def toU(v: Int, width: Int): UInt = (BigInt(v) & ((BigInt(1) << width) - 1)).U(width.W)

  "Accumulator" should "read back a written result beat after readLatency cycles (write-then-read, unicast)" in {
    val cuCfg = ComputeUnitConfig(numInputs = 1, numOutputs = 4, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val cfg = AccumulatorConfig(cuCfg, numBanks = 4, wordsPerBank = 1, depth = 2, readLatency = 2, writeLatency = 2)
    test(new Accumulator(cfg)) { c =>
      val beat = Seq(1, -2, 3, -4)

      c.io.resultIn.valid.poke(true.B)
      c.io.writeAddr.poke(0.U)
      for ((v, i) <- beat.zipWithIndex) c.io.resultIn.bits(i).poke(v.S(cfg.cu.outputWidth.W))
      c.clock.step()
      c.io.resultIn.valid.poke(false.B)
      c.clock.step(cfg.writeLatency)

      c.io.readAddr.poke(0.U)
      c.io.readEn.poke(true.B)
      c.io.shiftAmt.poke(0.U)
      c.io.groupShift.poke(0.U) // unicast
      c.clock.step(cfg.readLatency)

      c.io.accOut.valid.expect(true.B)
      for ((v, i) <- beat.zipWithIndex) c.io.accOut.bits(i).expect(toU(v, cfg.cu.outputWidth))
      c.io.readEn.poke(false.B)
    }
  }

  it should "let a drain read and a fresh write proceed concurrently on independent addresses (OneR1W)" in {
    val cuCfg = ComputeUnitConfig(numInputs = 1, numOutputs = 4, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val cfg = AccumulatorConfig(cuCfg, numBanks = 4, wordsPerBank = 1, depth = 2, readLatency = 2, writeLatency = 2)
    test(new Accumulator(cfg)) { c =>
      val bufA = Seq(10, 20, 30, 40)
      val bufB = Seq(50, 60, 70, 80)

      // prefill buffer A (addr 0), as if a previous pass already finished
      c.io.resultIn.valid.poke(true.B)
      c.io.writeAddr.poke(0.U)
      for ((v, i) <- bufA.zipWithIndex) c.io.resultIn.bits(i).poke(v.S(cfg.cu.outputWidth.W))
      c.clock.step()
      c.io.resultIn.valid.poke(false.B)
      c.clock.step(cfg.writeLatency)

      // now drain A (readAddr=0) while simultaneously writing B (addr 1) --
      // the defining OneR1W behavior: the drain must see A's untouched
      // data, unaffected by the concurrent write to a different address
      c.io.readAddr.poke(0.U)
      c.io.readEn.poke(true.B)
      c.io.shiftAmt.poke(0.U)
      c.io.groupShift.poke(0.U)

      c.io.resultIn.valid.poke(true.B)
      c.io.writeAddr.poke(1.U)
      for ((v, i) <- bufB.zipWithIndex) c.io.resultIn.bits(i).poke(v.S(cfg.cu.outputWidth.W))

      c.clock.step(cfg.readLatency)

      c.io.accOut.valid.expect(true.B)
      for ((v, i) <- bufA.zipWithIndex) c.io.accOut.bits(i).expect(toU(v, cfg.cu.outputWidth))

      c.io.readEn.poke(false.B)
      c.io.resultIn.valid.poke(false.B)
      c.clock.step(cfg.writeLatency)

      // B is now resident at addr 1 -- read it back to confirm the
      // concurrent write actually landed correctly
      c.io.readAddr.poke(1.U)
      c.io.readEn.poke(true.B)
      c.clock.step(cfg.readLatency)

      c.io.accOut.valid.expect(true.B)
      for ((v, i) <- bufB.zipWithIndex) c.io.accOut.bits(i).expect(toU(v, cfg.cu.outputWidth))
      c.io.readEn.poke(false.B)
    }
  }
}
