package xccel

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

// Wires a WeightBuffer into a SystolicArray's address-based weight port,
// the way the decoder eventually will, plus passthroughs for the compute
// unit's own operandIn/resultOut so the whole pipeline is testable end to
// end. Since SystolicArray no longer tracks its own load progress
// (HasStationaryWeights is address-based), this harness plays the
// decoder's role: it re-derives which row a given weightOut beat belongs
// to by delaying its own copy of `wbAddr` by the buffer's readLatency,
// the same delay WeightBuffer applies internally to produce that beat.
// (Only valid when wbAddr already equals the row-within-tile, i.e.
// tiles=1, as in the test below.)
class WeightBufferSystolicHarness(
  cuCfg: ComputeUnitConfig,
  numBanks: Int,
  wordsPerBank: Int,
  tiles: Int,
  wbReadLatency: Int,
  wbWriteLatency: Int,
) extends Module {
  val wbCfg = WeightBufferConfig(cuCfg, numBanks, wordsPerBank, tiles, wbReadLatency, wbWriteLatency)

  val io = IO(new Bundle {
    val wbAddr = Input(UInt(wbCfg.sram.addrWidth.W))
    val wbEn = Input(Bool())
    val wbWen = Input(Bool())
    val wbWdata = Input(Vec(wbCfg.totalWordsPerRow, UInt(cuCfg.weightWidth.W)))
    val wbShiftAmt = Input(UInt(wbCfg.laneSwitch.shiftWidth.W))
    val wbGroupShift = Input(UInt(wbCfg.laneSwitch.groupShiftWidth.W))

    val operandIn = Flipped(Valid(Vec(cuCfg.numInputs, UInt(cuCfg.inputWidth.W))))
    val resultOut = Valid(Vec(cuCfg.numOutputs, SInt(cuCfg.outputWidth.W)))
  })

  val weightBuffer = Module(new WeightBuffer(wbCfg))
  val compute = Module(new SystolicArray(cuCfg))

  // the harness keeps a single shared bus: wen splits it into the read/write streams
  weightBuffer.io.readAddr := io.wbAddr
  weightBuffer.io.readEn := io.wbEn && !io.wbWen
  weightBuffer.io.writeAddr := io.wbAddr
  weightBuffer.io.writeEn := io.wbEn && io.wbWen
  weightBuffer.io.wdata := io.wbWdata
  weightBuffer.io.shiftAmt := io.wbShiftAmt
  weightBuffer.io.groupShift := io.wbGroupShift

  compute.weightWriteEn := weightBuffer.io.weightOut.valid
  compute.weightWriteData := weightBuffer.io.weightOut.bits
  compute.weightWriteSlot := false.B // single weight set: slot 0 only
  compute.weightSelect := false.B
  compute.weightWriteAddr := ShiftRegister(io.wbAddr, wbReadLatency, 0.U(wbCfg.sram.addrWidth.W), true.B)

  compute.io.operandIn <> io.operandIn
  io.resultOut <> compute.io.resultOut
}

class WeightBufferTest extends AnyFlatSpec with ChiselScalatestTester {

  "WeightBuffer" should "read back a written row after readLatency cycles (one bank per column, unicast)" in {
    val cuCfg = ComputeUnitConfig(numInputs = 3, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val cfg = WeightBufferConfig(
      cuCfg,
      numBanks = cuCfg.numOutputs,
      wordsPerBank = 1,
      tiles = 1,
      readLatency = 2,
      writeLatency = 2,
    )
    test(new WeightBuffer(cfg)) { c =>
      val rows = Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6))
      c.io.shiftAmt.poke(0.U)
      c.io.groupShift.poke(0.U) // unicast

      for ((row, r) <- rows.zipWithIndex) {
        c.io.writeAddr.poke(r.U)
        c.io.writeEn.poke(true.B)
        for ((v, col) <- row.zipWithIndex) c.io.wdata(col).poke(v.U)
        c.clock.step()
      }
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
      c.clock.step(cfg.writeLatency)

      for ((row, r) <- rows.zipWithIndex) {
        c.io.readAddr.poke(r.U)
        c.io.readEn.poke(true.B)
        c.clock.step(cfg.readLatency)
        c.io.weightOut.valid.expect(true.B)
        for ((v, col) <- row.zipWithIndex) c.io.weightOut.bits(col).expect(v.U)
      }
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
    }
  }

  it should "deliver a whole row from a single wide bank (one bank, several columns, unicast)" in {
    val cuCfg = ComputeUnitConfig(numInputs = 3, numOutputs = 4, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val cfg = WeightBufferConfig(
      cuCfg,
      numBanks = 1,
      wordsPerBank = cuCfg.numOutputs,
      tiles = 1,
      readLatency = 2,
      writeLatency = 2,
    )
    test(new WeightBuffer(cfg)) { c =>
      val rows = Seq(Seq(1, 2, 3, 4), Seq(5, 6, 7, 8), Seq(9, 10, 11, 12))
      c.io.shiftAmt.poke(0.U)
      c.io.groupShift.poke(0.U) // unicast

      for ((row, r) <- rows.zipWithIndex) {
        c.io.writeAddr.poke(r.U)
        c.io.writeEn.poke(true.B)
        for ((v, i) <- row.zipWithIndex) c.io.wdata(i).poke(v.U)
        c.clock.step()
      }
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
      c.clock.step(cfg.writeLatency)

      for ((row, r) <- rows.zipWithIndex) {
        c.io.readAddr.poke(r.U)
        c.io.readEn.poke(true.B)
        c.clock.step(cfg.readLatency)
        c.io.weightOut.valid.expect(true.B)
        for ((v, i) <- row.zipWithIndex) c.io.weightOut.bits(i).expect(v.U)
      }
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
    }
  }

  it should "select the right cols-wide window via shiftAmt (several banks, several columns, several resident tiles, unicast)" in {
    val cuCfg = ComputeUnitConfig(numInputs = 3, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    // 3 tiles resident side by side across the bank dimension: A at banks
    // [0,1], B at [2,3], C at [4,5]
    val cfg =
      WeightBufferConfig(cuCfg, numBanks = 6, wordsPerBank = 1, tiles = 1, readLatency = 2, writeLatency = 2)
    test(new WeightBuffer(cfg)) { c =>
      val tileA = Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6))
      val tileB = Seq(Seq(10, 20), Seq(30, 40), Seq(50, 60))
      val tileC = Seq(Seq(100, 101), Seq(102, 103), Seq(104, 105))
      c.io.groupShift.poke(0.U) // unicast

      for (r <- 0 until cuCfg.numInputs) {
        c.io.writeAddr.poke(r.U)
        c.io.writeEn.poke(true.B)
        val packed = tileA(r) ++ tileB(r) ++ tileC(r)
        for ((v, word) <- packed.zipWithIndex) c.io.wdata(word).poke(v.U)
        c.clock.step()
      }
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
      c.clock.step(cfg.writeLatency)

      for ((tile, shamt) <- Seq((tileA, 0), (tileB, 2), (tileC, 4))) {
        for ((row, r) <- tile.zipWithIndex) {
          c.io.readAddr.poke(r.U)
          c.io.readEn.poke(true.B)
          c.io.shiftAmt.poke(shamt.U)
          c.clock.step(cfg.readLatency)
          c.io.weightOut.valid.expect(true.B)
          for ((v, col) <- row.zipWithIndex) c.io.weightOut.bits(col).expect(v.U)
        }
      }
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
    }
  }

  it should "broadcast one input lane to every output lane when groupShift = log2(cols)" in {
    val cuCfg = ComputeUnitConfig(numInputs = 2, numOutputs = 4, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val cfg =
      WeightBufferConfig(cuCfg, numBanks = 4, wordsPerBank = 1, tiles = 1, readLatency = 1, writeLatency = 1)
    test(new WeightBuffer(cfg)) { c =>
      val row = Seq(1, 2, 3, 4)

      c.io.writeAddr.poke(0.U)
      c.io.writeEn.poke(true.B)
      for ((v, i) <- row.zipWithIndex) c.io.wdata(i).poke(v.U)
      c.clock.step()
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
      c.clock.step(cfg.writeLatency)

      c.io.readAddr.poke(0.U)
      c.io.readEn.poke(true.B)
      c.io.groupShift.poke(2.U) // log2(4) = broadcast

      for (shamt <- row.indices) {
        c.io.shiftAmt.poke(shamt.U)
        c.clock.step(cfg.readLatency)
        c.io.weightOut.valid.expect(true.B)
        for (i <- row.indices) c.io.weightOut.bits(i).expect(row(shamt).U)
      }
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
    }
  }

  it should "multicast in groups of 2 when groupShift = 1 (between unicast and broadcast)" in {
    val cuCfg = ComputeUnitConfig(numInputs = 2, numOutputs = 4, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val cfg =
      WeightBufferConfig(cuCfg, numBanks = 4, wordsPerBank = 1, tiles = 1, readLatency = 1, writeLatency = 1)
    test(new WeightBuffer(cfg)) { c =>
      val row = Seq(1, 2, 3, 4)

      c.io.writeAddr.poke(0.U)
      c.io.writeEn.poke(true.B)
      for ((v, i) <- row.zipWithIndex) c.io.wdata(i).poke(v.U)
      c.clock.step()
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
      c.clock.step(cfg.writeLatency)

      c.io.readAddr.poke(0.U)
      c.io.readEn.poke(true.B)
      c.io.shiftAmt.poke(0.U)
      c.io.groupShift.poke(1.U) // groupSize = 2
      c.clock.step(cfg.readLatency)

      c.io.weightOut.valid.expect(true.B)
      // output lanes 0,1 share input lane 0; 2,3 share input lane 1
      val expected = Seq(row(0), row(0), row(1), row(1))
      for (i <- row.indices) c.io.weightOut.bits(i).expect(expected(i).U)
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
    }
  }

  "WeightBuffer -> SystolicArray wiring" should "produce the same matmul as poking weightIn directly" in {
    val cuCfg = ComputeUnitConfig(numInputs = 3, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    test(
      new WeightBufferSystolicHarness(
        cuCfg,
        numBanks = cuCfg.numOutputs,
        wordsPerBank = 1,
        tiles = 1,
        wbReadLatency = 2,
        wbWriteLatency = 2,
      )
    ) { c =>
      def toU(v: Int, width: Int): UInt = (BigInt(v) & ((BigInt(1) << width) - 1)).U(width.W)

      c.io.wbShiftAmt.poke(0.U)
      c.io.wbGroupShift.poke(0.U) // unicast

      val weightRows = Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6))
      for ((row, r) <- weightRows.zipWithIndex) {
        c.io.wbAddr.poke(r.U)
        c.io.wbEn.poke(true.B)
        c.io.wbWen.poke(true.B)
        for ((v, col) <- row.zipWithIndex) c.io.wbWdata(col).poke(toU(v, cuCfg.weightWidth))
        c.clock.step()
      }
      c.io.wbEn.poke(false.B)
      c.io.wbWen.poke(false.B)
      c.clock.step(2) // wbWriteLatency

      // stream every row out to load the systolic array, then a margin for
      // the last row's weightOut beat to actually arrive
      for (r <- weightRows.indices) {
        c.io.wbAddr.poke(r.U)
        c.io.wbEn.poke(true.B)
        c.clock.step()
      }
      c.io.wbEn.poke(false.B)
      c.clock.step(2) // wbReadLatency margin

      val samples = Seq(Seq(1, 1, 1), Seq(2, 0, -1))
      val expected = Seq(Seq(9, 12), Seq(-3, -2))

      c.io.operandIn.valid.poke(true.B)
      for (s <- samples) {
        for ((v, i) <- s.zipWithIndex) c.io.operandIn.bits(i).poke(toU(v, cuCfg.inputWidth))
        c.clock.step()
      }
      c.io.operandIn.valid.poke(false.B)

      val latency = cuCfg.numInputs + cuCfg.numOutputs - 1
      c.clock.step(latency - samples.length)

      for (exp <- expected) {
        c.io.resultOut.valid.expect(true.B)
        for ((v, i) <- exp.zipWithIndex) c.io.resultOut.bits(i).expect(v.S)
        c.clock.step()
      }
    }
  }

  for (portType <- Seq(BankPortType.OneRW, BankPortType.OneR1W, BankPortType.TwoRW))
  it should s"handle a read and a write in the same cycle per its port type ($portType)" in {
    val cuCfg = ComputeUnitConfig(numInputs = 2, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val cfg = WeightBufferConfig(cuCfg, numBanks = cuCfg.numOutputs, wordsPerBank = 1, tiles = 1,
      readLatency = 2, writeLatency = 2, portType = portType)
    test(new WeightBuffer(cfg)) { c =>
      c.io.shiftAmt.poke(0.U)
      c.io.groupShift.poke(0.U)
      def write(r: Int, row: Seq[Int]): Unit = {
        c.io.writeAddr.poke(r.U)
        c.io.writeEn.poke(true.B)
        for ((v, i) <- row.zipWithIndex) c.io.wdata(i).poke(v.U)
      }
      def read(r: Int): Seq[BigInt] = {
        c.io.readAddr.poke(r.U)
        c.io.readEn.poke(true.B)
        c.clock.step(cfg.readLatency)
        c.io.readEn.poke(false.B)
        c.io.weightOut.valid.expect(true.B)
        (0 until cuCfg.numOutputs).map(i => c.io.weightOut.bits(i).peek().litValue)
      }

      write(0, Seq(1, 2))
      c.clock.step()
      c.io.writeEn.poke(false.B)
      c.clock.step(cfg.writeLatency)

      // same cycle: read row 0, write row 1
      c.io.readAddr.poke(0.U)
      c.io.readEn.poke(true.B)
      write(1, Seq(3, 4))
      c.clock.step()
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
      c.clock.step(cfg.readLatency - 1)
      c.io.weightOut.valid.expect(true.B)
      for ((v, i) <- Seq(1, 2).zipWithIndex) c.io.weightOut.bits(i).expect(v.U)
      c.clock.step(cfg.writeLatency)

      // separate ports land the write; OneRW's single port gives the cycle
      // to the read and drops the write
      val row1 = read(1)
      if (portType == BankPortType.OneRW) assert(row1 != Seq(BigInt(3), BigInt(4)))
      else assert(row1 == Seq(BigInt(3), BigInt(4)))
    }
  }
}
