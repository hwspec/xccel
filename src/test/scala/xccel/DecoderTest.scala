package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class DecoderTest extends AnyFlatSpec with ChiselScalatestTester {

  def loadInstr(
    c: Decoder,
    idx: Int,
    source: DecoderSource.Type,
    op: DecoderOp.Type,
    baseAddr: Int,
    baseShiftAmt: Int,
    groupShift: Int,
    length: Int,
  ): Unit = {
    c.io.progWriteAddr.poke(idx.U)
    c.io.progWriteData.source.poke(source)
    c.io.progWriteData.op.poke(op)
    c.io.progWriteData.baseAddr.poke(baseAddr.U)
    c.io.progWriteData.baseShiftAmt.poke(baseShiftAmt.U)
    c.io.progWriteData.groupShift.poke(groupShift.U)
    c.io.progWriteData.length.poke(length.U)
    c.io.progWriteEn.poke(true.B)
    c.clock.step()
  }

  "Decoder" should "sequence WeightBufferRead(ADVANCE), ActivationBufferRead(ADVANCE then RESELECT)," +
    " AccumulatorRead(ADVANCE), then self-terminate on Halt" in {
      val cuCfg = ComputeUnitConfig(numInputs = 4, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
      val wbCfg = WeightBufferConfig(cuCfg, numBanks = cuCfg.numOutputs, wordsPerBank = 1, tiles = 1,
        readLatency = 2, writeLatency = 2)
      val abCfg = ActivationBufferConfig(cuCfg, numBanks = cuCfg.numInputs, wordsPerBank = 1, depth = 8,
        readLatency = 1, writeLatency = 1)
      val accCfg = AccumulatorConfig(cuCfg, numBanks = cuCfg.numOutputs, wordsPerBank = 1, depth = 4,
        readLatency = 2, writeLatency = 2)
      val cfg = DecoderConfig(wbCfg, abCfg, accCfg, programLength = 5, maxRunLength = 2)

      test(new Decoder(cfg)) { c =>
        // instr0: WeightBufferRead ADVANCE, addr 0..1 (2 steps)
        loadInstr(c, 0, DecoderSource.WeightBufferRead, DecoderOp.Advance, baseAddr = 0, baseShiftAmt = 0,
          groupShift = 0, length = 2)
        // instr1: ActivationBufferRead ADVANCE, addr=0 (1 step)
        loadInstr(c, 1, DecoderSource.ActivationBufferRead, DecoderOp.Advance, baseAddr = 0, baseShiftAmt = 0,
          groupShift = 0, length = 1)
        // instr2: ActivationBufferRead RESELECT, addr fixed=5, shiftAmt 0..1 (2 steps)
        loadInstr(c, 2, DecoderSource.ActivationBufferRead, DecoderOp.Reselect, baseAddr = 5, baseShiftAmt = 0,
          groupShift = 0, length = 2)
        // instr3: AccumulatorRead ADVANCE, addr=2 (1 step)
        loadInstr(c, 3, DecoderSource.AccumulatorRead, DecoderOp.Advance, baseAddr = 2, baseShiftAmt = 0,
          groupShift = 1, length = 1)
        // instr4: Halt
        loadInstr(c, 4, DecoderSource.Halt, DecoderOp.Advance, baseAddr = 0, baseShiftAmt = 0, groupShift = 0,
          length = 1)
        c.io.progWriteEn.poke(false.B)

        c.io.start.poke(true.B)
        c.clock.step()
        c.io.start.poke(false.B)
        c.io.running.expect(true.B)

        // cycle0,1: instr0 (WeightBufferRead ADVANCE), addr = baseAddr+offset = 0,1
        for (expectedAddr <- Seq(0, 1)) {
          c.io.wbEn.expect(true.B)
          c.io.wbAddr.expect(expectedAddr.U)
          c.io.abEn.expect(false.B)
          c.io.accEn.expect(false.B)
          c.clock.step()
        }

        // cycle2: instr1 (ActivationBufferRead ADVANCE), single step, shiftIn always true on ADVANCE
        c.io.abEn.expect(true.B)
        c.io.abAddr.expect(0.U)
        c.io.abShiftAmt.expect(0.U)
        c.io.abShiftIn.expect(true.B)
        c.io.wbEn.expect(false.B)
        c.clock.step()

        // cycle3: instr2 step0 (RESELECT), addr fixed at baseAddr=5, shiftAmt=0, NOT the last step -> shiftIn=false
        c.io.abEn.expect(true.B)
        c.io.abAddr.expect(5.U)
        c.io.abShiftAmt.expect(0.U)
        c.io.abShiftIn.expect(false.B)
        c.clock.step()

        // cycle4: instr2 step1 (RESELECT), addr STILL 5 (unchanged), shiftAmt=1, last step -> shiftIn=true
        c.io.abEn.expect(true.B)
        c.io.abAddr.expect(5.U)
        c.io.abShiftAmt.expect(1.U)
        c.io.abShiftIn.expect(true.B)
        c.clock.step()

        // cycle5: instr3 (AccumulatorRead ADVANCE), addr=2, groupShift=1
        c.io.accEn.expect(true.B)
        c.io.accAddr.expect(2.U)
        c.io.accGroupShift.expect(1.U)
        c.io.abEn.expect(false.B)
        c.clock.step()

        // cycle6: instr4 (Halt) -- nothing active, running still true this cycle
        c.io.wbEn.expect(false.B)
        c.io.abEn.expect(false.B)
        c.io.accEn.expect(false.B)
        c.io.running.expect(true.B)
        c.clock.step()

        // running deasserts once the Halt instruction has been processed
        c.io.running.expect(false.B)
      }
    }

  it should "shadow wbAddr by weightPortDelay cycles onto weightWriteAddr" in {
    val cuCfg = ComputeUnitConfig(numInputs = 2, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val wbCfg = WeightBufferConfig(cuCfg, numBanks = cuCfg.numOutputs, wordsPerBank = 1, tiles = 1,
      readLatency = 2, writeLatency = 2)
    val abCfg = ActivationBufferConfig(cuCfg, numBanks = cuCfg.numInputs, wordsPerBank = 1, depth = 4,
      readLatency = 1, writeLatency = 1)
    val accCfg = AccumulatorConfig(cuCfg, numBanks = cuCfg.numOutputs, wordsPerBank = 1, depth = 4,
      readLatency = 2, writeLatency = 2)
    val cfg = DecoderConfig(wbCfg, abCfg, accCfg, programLength = 2, maxRunLength = 2)

    test(new Decoder(cfg)) { c =>
      loadInstr(c, 0, DecoderSource.WeightBufferRead, DecoderOp.Advance, baseAddr = 0, baseShiftAmt = 0,
        groupShift = 0, length = 2)
      loadInstr(c, 1, DecoderSource.Halt, DecoderOp.Advance, baseAddr = 0, baseShiftAmt = 0, groupShift = 0,
        length = 1)
      c.io.progWriteEn.poke(false.B)

      c.io.start.poke(true.B)
      c.clock.step()
      c.io.start.poke(false.B)

      // wbAddr = 0, 1 on the two ADVANCE steps
      for (expectedAddr <- Seq(0, 1)) {
        c.io.wbAddr.expect(expectedAddr.U)
        c.clock.step()
      }

      // weightWriteAddr echoes wbAddr, delayed by weightPortDelay (=2) cycles:
      // wbAddr was 0 at cycle0, so weightWriteAddr shows 0 at cycle0+2=cycle2,
      // which is exactly where the loop above already left us (2 steps done)
      c.io.weightWriteAddr.expect(0.U)
      c.clock.step()
      c.io.weightWriteAddr.expect(1.U)
    }
  }
}
