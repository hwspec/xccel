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

      c.io.accumulate.poke(false.B)
      c.io.resultIn.valid.poke(true.B)
      c.io.writeAddr.poke(0.U)
      for ((v, i) <- beat.zipWithIndex) c.io.resultIn.bits(i).poke(v.S(cfg.cu.outputWidth.W))
      c.clock.step()
      c.io.resultIn.valid.poke(false.B)
      c.clock.step(cfg.commitLatency)

      c.io.readAddr.poke(0.U)
      c.io.readEn.poke(true.B)
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(0.U) // unicast
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
      c.io.accumulate.poke(false.B)
      c.io.resultIn.valid.poke(true.B)
      c.io.writeAddr.poke(0.U)
      for ((v, i) <- bufA.zipWithIndex) c.io.resultIn.bits(i).poke(v.S(cfg.cu.outputWidth.W))
      c.clock.step()
      c.io.resultIn.valid.poke(false.B)
      c.clock.step(cfg.commitLatency)

      // now drain A (readAddr=0) while simultaneously writing B (addr 1) --
      // the defining OneR1W behavior: the drain must see A's untouched
      // data, unaffected by the concurrent write to a different address
      c.io.readAddr.poke(0.U)
      c.io.readEn.poke(true.B)
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(0.U)

      c.io.resultIn.valid.poke(true.B)
      c.io.writeAddr.poke(1.U)
      for ((v, i) <- bufB.zipWithIndex) c.io.resultIn.bits(i).poke(v.S(cfg.cu.outputWidth.W))

      c.clock.step(cfg.readLatency)

      c.io.accOut.valid.expect(true.B)
      for ((v, i) <- bufA.zipWithIndex) c.io.accOut.bits(i).expect(toU(v, cfg.cu.outputWidth))

      c.io.readEn.poke(false.B)
      c.io.resultIn.valid.poke(false.B)
      c.clock.step(cfg.commitLatency)

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

  it should "add accumulating beats to a row's current value (read-modify-write), streaming back to back" in {
    val cuCfg = ComputeUnitConfig(numInputs = 1, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val cfg = AccumulatorConfig(cuCfg, numBanks = 2, wordsPerBank = 1, depth = 4, readLatency = 2, writeLatency = 2)
    test(new Accumulator(cfg)) { c =>
      c.io.readEn.poke(false.B)
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(0.U)

      // one beat per cycle to rows 0..2, each pass on consecutive cycles
      def pass(values: Seq[Seq[Int]], accumulate: Boolean): Unit = {
        c.io.accumulate.poke(accumulate.B)
        c.io.resultIn.valid.poke(true.B)
        for ((beat, r) <- values.zipWithIndex) {
          c.io.writeAddr.poke(r.U)
          for ((v, i) <- beat.zipWithIndex) c.io.resultIn.bits(i).poke(v.S(cfg.cu.outputWidth.W))
          c.clock.step()
        }
        c.io.resultIn.valid.poke(false.B)
        c.clock.step(cfg.commitLatency)
      }

      val p0 = Seq(Seq(1, -2), Seq(3, 4), Seq(-5, 6))
      val p1 = Seq(Seq(10, 20), Seq(-30, 40), Seq(50, 60))
      val p2 = Seq(Seq(100, 100), Seq(100, 100), Seq(100, 100))
      pass(p0, accumulate = false) // overwrite: whatever was there before is irrelevant
      pass(p1, accumulate = true)
      pass(p2, accumulate = true)

      val expected = (0 until 3).map(r => (0 until 2).map(i => p0(r)(i) + p1(r)(i) + p2(r)(i)))
      c.io.readEn.poke(true.B)
      for ((exp, r) <- expected.zipWithIndex) {
        c.io.readAddr.poke(r.U)
        c.clock.step(cfg.readLatency)
        c.io.accOut.valid.expect(true.B)
        for ((v, i) <- exp.zipWithIndex) c.io.accOut.bits(i).expect(toU(v, cfg.cu.outputWidth))
      }
      c.io.readEn.poke(false.B)
    }
  }

  it should "accumulate into one row bank while draining the other in the same cycles (rowBanks = 2)" in {
    val cuCfg = ComputeUnitConfig(numInputs = 1, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val cfg = AccumulatorConfig(cuCfg, numBanks = 2, wordsPerBank = 1, depth = 8, readLatency = 2, writeLatency = 2,
      rowBanks = 2) // rows 0..3 in bank 0, rows 4..7 in bank 1
    test(new Accumulator(cfg)) { c =>
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(0.U)
      c.io.readEn.poke(false.B)
      def beat(row: Int, v: Seq[Int], acc: Boolean): Unit = {
        c.io.resultIn.valid.poke(true.B)
        c.io.accumulate.poke(acc.B)
        c.io.writeAddr.poke(row.U)
        for ((x, i) <- v.zipWithIndex) c.io.resultIn.bits(i).poke(x.S(cfg.cu.outputWidth.W))
      }
      // initial values: rows 0..3 (bank 0) and 4..7 (bank 1)
      for (r <- 0 until 8) { beat(r, Seq(r, -r), acc = false); c.clock.step() }
      c.io.resultIn.valid.poke(false.B)
      c.clock.step(cfg.commitLatency)

      // cycles 0..3: accumulate +100 into rows 0..3 while draining rows 4..7
      val drained = scala.collection.mutable.ArrayBuffer.empty[Seq[BigInt]]
      for (k <- 0 until 4 + cfg.readLatency) {
        if (k < 4) {
          beat(k, Seq(100, 100), acc = true)
          c.io.readAddr.poke((4 + k).U)
          c.io.readEn.poke(true.B)
        } else {
          c.io.resultIn.valid.poke(false.B)
          c.io.readEn.poke(false.B)
        }
        if (c.io.accOut.valid.peek().litToBoolean)
          drained += (0 until 2).map(i => c.io.accOut.bits(i).peek().litValue)
        c.clock.step()
      }
      def toU(v: Int) = (BigInt(v) & ((BigInt(1) << cfg.cu.outputWidth) - 1))
      assert(drained.toSeq == (4 until 8).map(r => Seq(toU(r), toU(-r))), s"drained $drained")
      c.clock.step(cfg.commitLatency)

      // bank 0 rows now hold r + 100, -r + 100
      c.io.readEn.poke(true.B)
      for (r <- 0 until 4) {
        c.io.readAddr.poke(r.U)
        c.clock.step(cfg.readLatency)
        c.io.accOut.valid.expect(true.B)
        c.io.accOut.bits(0).expect(toU(r + 100).U)
        c.io.accOut.bits(1).expect(toU(100 - r).U)
      }
    }
  }

  "Accumulator with address groups" should "write, accumulate and drain each group's row in its own row bank" in {
    val cuCfg = ComputeUnitConfig(numInputs = 1, numOutputs = 4, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    // rows 0-1 in row bank 0, rows 2-3 in row bank 1; columns 0-1 are
    // address group 0, columns 2-3 group 1
    val cfg = AccumulatorConfig(cuCfg, numBanks = 4, wordsPerBank = 1, depth = 4, readLatency = 2, writeLatency = 2,
      rowBanks = 2, addrGroups = 2)
    test(new Accumulator(cfg)) { c =>
      val wg = c.io.writeGroupAddr.get(0)
      val rg = c.io.readGroupAddr.get(0)
      c.io.readEn.poke(false.B)
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(0.U)

      def beat(a0: Int, a1: Int, accumulate: Boolean, s: Seq[Int]): Unit = {
        c.io.writeAddr.poke(a0.U)
        wg.poke(a1.U)
        c.io.accumulate.poke(accumulate.B)
        c.io.resultIn.valid.poke(true.B)
        for ((v, i) <- s.zipWithIndex) c.io.resultIn.bits(i).poke(v.S(cfg.cu.outputWidth.W))
        c.clock.step()
        c.io.resultIn.valid.poke(false.B)
        c.clock.step(cfg.commitLatency)
      }
      def drain(a0: Int, a1: Int, expected: Seq[Int]): Unit = {
        c.io.readAddr.poke(a0.U)
        rg.poke(a1.U)
        c.io.readEn.poke(true.B)
        c.clock.step()
        c.io.readEn.poke(false.B)
        c.clock.step(cfg.readLatency - 1)
        c.io.accOut.valid.expect(true.B)
        for ((v, i) <- expected.zipWithIndex) c.io.accOut.bits(i).expect(toU(v, cfg.cu.outputWidth))
      }

      // group 0 -> row 0 (row bank 0), group 1 -> row 3 (row bank 1), and
      // the other way round
      beat(0, 3, accumulate = false, Seq(1, 2, 3, 4))
      beat(3, 0, accumulate = false, Seq(5, 6, 7, 8))
      // each group's read-modify-write reads its own row in its own row bank
      beat(0, 3, accumulate = true, Seq(10, 20, 30, -40))
      drain(0, 3, Seq(11, 22, 33, -36))
      drain(3, 0, Seq(5, 6, 7, 8))
    }
  }

  "Accumulator with a write switch" should "place beats anywhere in a wider row, writing only the words they land on" in {
    // rows of 4 words (2 banks x 2 words), beats of 2 lanes
    val cuCfg = ComputeUnitConfig(numInputs = 1, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val cfg = AccumulatorConfig(cuCfg, numBanks = 2, wordsPerBank = 2, depth = 2, readLatency = 2, writeLatency = 2)
    test(new Accumulator(cfg)) { c =>
      c.io.readEn.poke(false.B)
      c.io.fanout.poke(0.U)
      c.io.interleave.poke(0.U)

      def beat(at: Int, accumulate: Boolean, s: Seq[Int], fanout: Int = 0, interleave: Int = 0): Unit = {
        c.io.writeAddr.poke(0.U)
        c.io.writeRotate.poke(((cfg.totalWordsPerRow - at) % cfg.totalWordsPerRow).U) // lane 0 -> word `at`
        c.io.writeFanout.poke(fanout.U)
        c.io.writeInterleave.poke(interleave.U)
        c.io.accumulate.poke(accumulate.B)
        c.io.resultIn.valid.poke(true.B)
        for ((v, i) <- s.zipWithIndex) c.io.resultIn.bits(i).poke(v.S(cfg.cu.outputWidth.W))
        c.clock.step()
        c.io.resultIn.valid.poke(false.B)
        c.clock.step(cfg.commitLatency)
      }
      // the whole row, through the drain switch (rotate = first word)
      def row(expected: Seq[Int]): Unit =
        for (at <- Seq(0, 2)) {
          c.io.readAddr.poke(0.U)
          c.io.rotate.poke(at.U)
          c.io.readEn.poke(true.B)
          c.clock.step()
          c.io.readEn.poke(false.B)
          c.clock.step(cfg.readLatency - 1)
          c.io.accOut.valid.expect(true.B)
          for (i <- 0 until 2) c.io.accOut.bits(i).expect(toU(expected(at + i), cfg.cu.outputWidth))
        }

      beat(0, accumulate = false, Seq(1, 2))
      beat(2, accumulate = false, Seq(3, 4))
      row(Seq(1, 2, 3, 4))
      // accumulate into words 2-3 only; words 0-1 keep their value
      beat(2, accumulate = true, Seq(10, -20))
      row(Seq(1, 2, 13, -16))
      // across the row end: lanes 0, 1 -> words 3, 0
      beat(3, accumulate = false, Seq(7, 8))
      row(Seq(8, 2, 13, 7))
      // multicast over the row: grouped (r0 r0 r1 r1), interleaved (r0 r1 r0 r1)
      beat(0, accumulate = false, Seq(5, 6), fanout = 1)
      row(Seq(5, 5, 6, 6))
      beat(0, accumulate = false, Seq(5, 6), fanout = 1, interleave = 1)
      row(Seq(5, 6, 5, 6))
    }
  }
}
