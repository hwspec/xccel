package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class DecoderTest extends AnyFlatSpec with ChiselScalatestTester {

  val cuCfg = ComputeUnitConfig(numInputs = 4, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
  val wbCfg = WeightBufferConfig(cuCfg, numBanks = cuCfg.numOutputs, wordsPerBank = 1, tiles = 1,
    readLatency = 2, writeLatency = 2)
  val abCfg = ActivationBufferConfig(cuCfg, numBanks = cuCfg.numInputs, wordsPerBank = 1, depth = 8,
    readLatency = 1, writeLatency = 1)
  val accCfg = AccumulatorConfig(cuCfg, numBanks = cuCfg.numOutputs, wordsPerBank = 1, depth = 4,
    readLatency = 2, writeLatency = 2)

  def loader(c: Decoder) =
    new MicroProgramLoader(c.clock, c.io.progWriteEn, c.io.progWriteAddr, c.io.progWriteData)

  def pulseStart(c: Decoder): Unit = {
    c.io.start.poke(true.B)
    c.clock.step()
    c.io.start.poke(false.B)
  }

  "Decoder" should "hold each word for its length, apply strides per cycle, drive slots in parallel," +
    " and self-terminate on halt" in {
      val cfg = DecoderConfig(wbCfg, abCfg, accCfg, programLength = 5, maxRunLength = 2)

      test(new Decoder(cfg)) { c =>
        val p = loader(c)
        // w0: WeightBuffer read, addr 0,1, while writing WeightBuffer addr
        // 3,2 and ActivationBuffer addr 4,6
        p.word(length = 2) { w =>
          w.wbRead.en.poke(true.B)
          w.wbRead.addrStride.poke(1.U)
          w.wbWrite.en.poke(true.B)
          w.wbWrite.addr.poke(3.U)
          w.wbWrite.addrStride.poke(MicroProgramLoader.minusOne(wbCfg.sram.addrWidth))
          w.wbWrite.transposed.poke(true.B)
          w.wbWrite.column.poke(0.U)
          w.wbWrite.columnStride.poke(1.U)
          w.transposer.capture.poke(true.B)
          w.transposer.rowStride.poke(1.U)
          w.softmax.op.poke(SoftmaxOp.Sum.U)
          w.softmax.first.poke(true.B)
          w.softmax.lanes.poke(2.U)
          w.residual.en.poke(true.B)
          w.residual.paramSet.poke(1.U)
          w.abWrite.en.poke(true.B)
          w.abWrite.addr.poke(4.U)
          w.abWrite.addrStride.poke(2.U)
          w.abWrite.fromRequantizer.poke(true.B)
        }
        // w1: ActivationBuffer sweeping shiftAmt 0,1 at a fixed addr, in
        // parallel with the weight port writing slot 1, addr 0,1
        p.word(length = 2) { w =>
          w.abRead.en.poke(true.B)
          w.abRead.addr.poke(5.U)
          w.abRead.shiftStride.poke(1.U)
          w.weightPort.en.poke(true.B)
          w.weightPort.addrStride.poke(1.U)
          w.weightPort.slot.poke(true.B)
        }
        // w2: one more ActivationBuffer read, compute selecting weight slot 1
        p.word(length = 1) { w =>
          w.abRead.en.poke(true.B)
          w.abRead.addr.poke(5.U)
          w.abRead.shiftAmt.poke(2.U)
          w.compute.weightSelect.poke(true.B)
        }
        // w3: Accumulator drain counting down (addr 3,2) with groupShift=1,
        // accumulating writes counting up (1,2)
        p.word(length = 2) { w =>
          w.accRead.en.poke(true.B)
          w.accRead.addr.poke(3.U)
          w.accRead.addrStride.poke(MicroProgramLoader.minusOne(accCfg.addrWidth))
          w.accRead.groupShift.poke(1.U)
          w.accWrite.addr.poke(1.U)
          w.accWrite.addrStride.poke(1.U)
          w.accWrite.accumulate.poke(true.B)
        }
        // w4: idle, halt
        p.word(length = 1, halt = true)()
        p.done()

        pulseStart(c)
        c.io.running.expect(true.B)

        // t0,1: w0
        for ((a, i) <- Seq(0, 1).zipWithIndex) {
          c.io.wbRead.en.expect(true.B)
          c.io.wbRead.addr.expect(a.U)
          c.io.wbWrite.en.expect(true.B)
          c.io.wbWrite.addr.expect((3 - i).U)
          c.io.wbWrite.transposed.expect(true.B)
          c.io.wbWrite.column.expect(i.U)
          c.io.transposer.capture.expect(true.B)
          c.io.transposer.row.expect(i.U)
          c.io.softmax.op.expect(SoftmaxOp.Sum.U)
          c.io.softmax.lanes.expect(2.U)
          c.io.residual.en.expect(true.B)
          c.io.residual.paramSet.expect(1.U)
          c.io.abWrite.en.expect(true.B)
          c.io.abWrite.addr.expect((4 + 2 * i).U)
          c.io.abWrite.fromRequantizer.expect(true.B)
          c.io.abRead.en.expect(false.B)
          c.io.weightPort.en.expect(false.B)
          c.io.accRead.en.expect(false.B)
          c.io.weightSelect.expect(false.B)
          c.io.accWrite.accumulate.expect(false.B)
          c.clock.step()
        }

        // t2,3: w1, both slots active in the same cycles
        for (s <- Seq(0, 1)) {
          c.io.wbRead.en.expect(false.B)
          c.io.abRead.en.expect(true.B)
          c.io.abRead.addr.expect(5.U)
          c.io.abRead.shiftAmt.expect(s.U)
          c.io.weightPort.en.expect(true.B)
          c.io.weightPort.addr.expect(s.U)
          c.io.weightPort.slot.expect(true.B)
          c.io.weightSelect.expect(false.B)
          c.clock.step()
        }

        // t4: w2
        c.io.abRead.en.expect(true.B)
        c.io.abRead.addr.expect(5.U)
        c.io.abRead.shiftAmt.expect(2.U)
        c.io.weightSelect.expect(true.B)
        c.io.weightPort.en.expect(false.B)
        c.clock.step()

        // t5,6: w3
        for ((r, wr) <- Seq((3, 1), (2, 2))) {
          c.io.abRead.en.expect(false.B)
          c.io.accRead.en.expect(true.B)
          c.io.accRead.addr.expect(r.U)
          c.io.accRead.groupShift.expect(1.U)
          c.io.accWrite.addr.expect(wr.U)
          c.io.accWrite.accumulate.expect(true.B)
          c.io.weightSelect.expect(false.B)
          c.clock.step()
        }

        // t7: w4 (halt word) -- nothing enabled, still running this cycle
        c.io.wbRead.en.expect(false.B)
        c.io.wbWrite.en.expect(false.B)
        c.io.transposer.capture.expect(false.B)
        c.io.softmax.op.expect(SoftmaxOp.None.U)
        c.io.residual.en.expect(false.B)
        c.io.abWrite.en.expect(false.B)
        c.io.abRead.en.expect(false.B)
        c.io.weightPort.en.expect(false.B)
        c.io.accRead.en.expect(false.B)
        c.io.running.expect(true.B)
        c.clock.step()

        c.io.running.expect(false.B)
      }
    }

  it should "execute the halt word itself, and restart from word 0 with fresh strides" in {
    val cfg = DecoderConfig(wbCfg, abCfg, accCfg, programLength = 1, maxRunLength = 3)

    test(new Decoder(cfg)) { c =>
      val p = loader(c)
      p.word(length = 3, halt = true) { w =>
        w.wbRead.en.poke(true.B)
        w.wbRead.addr.poke(1.U)
        w.wbRead.addrStride.poke(1.U)
      }
      p.done()

      for (_ <- 0 until 2) {
        pulseStart(c)
        for (a <- Seq(1, 2, 3)) {
          c.io.running.expect(true.B)
          c.io.wbRead.en.expect(true.B)
          c.io.wbRead.addr.expect(a.U)
          c.clock.step()
        }
        c.io.running.expect(false.B)
        c.io.wbRead.en.expect(false.B)
        c.clock.step()
      }
    }
  }
}
