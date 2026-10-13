package xccel

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class ActivationBufferSystolicHarness(
  cuCfg: ComputeUnitConfig,
  numBanks: Int,
  wordsPerBank: Int,
  depth: Int,
  abReadLatency: Int,
  abWriteLatency: Int,
) extends Module {
  val abCfg = ActivationBufferConfig(cuCfg, numBanks, wordsPerBank, depth, abReadLatency, abWriteLatency)
  private val weightAddrWidth = math.max(1, log2Ceil(cuCfg.numInputs))

  val io = IO(new Bundle {
    val abAddr = Input(UInt(abCfg.sram.addrWidth.W))
    val abEn = Input(Bool())
    val abWen = Input(Bool())
    val abWdata = Input(Vec(abCfg.totalWordsPerRow, UInt(cuCfg.inputWidth.W)))
    val abRotate = Input(UInt(abCfg.laneSwitch.rotateWidth.W))
    val abFanout = Input(UInt(abCfg.laneSwitch.fanoutWidth.W))
    // val abShiftIn = Input(Bool())

    val weightWriteAddr = Input(UInt(weightAddrWidth.W))
    val weightWriteEn = Input(Bool())
    val weightWriteData = Input(Vec(cuCfg.numOutputs, UInt(cuCfg.weightWidth.W)))

    val resultOut = Valid(Vec(cuCfg.numOutputs, SInt(cuCfg.outputWidth.W)))
  })

  val activationBuffer = Module(new ActivationBuffer(abCfg))
  val compute = Module(new SystolicArray(cuCfg))

  // the harness keeps a single shared bus: wen splits it into the read/write streams
  activationBuffer.io.readAddr := io.abAddr
  activationBuffer.io.readEn := io.abEn && !io.abWen
  activationBuffer.io.writeAddr := io.abAddr
  activationBuffer.io.writeEn := io.abEn && io.abWen
  activationBuffer.io.wdata := io.abWdata
  // whole rows, unmoved
  activationBuffer.io.wmask.foreach(_ := true.B)
  activationBuffer.io.writeRotate := 0.U
  activationBuffer.io.writeFanout := 0.U
  activationBuffer.io.writeInterleave := 0.U
  activationBuffer.io.rotate := io.abRotate
  activationBuffer.io.fanout := io.abFanout
  activationBuffer.io.interleave := 0.U // grouped multicast only
  // activationBuffer.io.shiftIn := io.abShiftIn

  compute.weightWriteAddr := io.weightWriteAddr
  compute.weightWriteEn := io.weightWriteEn
  compute.weightWriteData := io.weightWriteData
  compute.weightWriteSlot := false.B // single weight set: slot 0 only
  compute.weightSelect := false.B

  compute.io.operandIn <> activationBuffer.io.operandOut
  io.resultOut <> compute.io.resultOut
}

class ActivationBufferTest extends AnyFlatSpec with ChiselScalatestTester {

  def toU(v: Int, width: Int): UInt = (BigInt(v) & ((BigInt(1) << width) - 1)).U(width.W)

  "ActivationBuffer" should "read back a written sample after readLatency cycles (one bank per input, unicast)" in {
    // numInputs is a power of two here: it is the LaneSwitch's numOut
    // (any numOut works; fanout then has a partial last group)
    val cuCfg = ComputeUnitConfig(numInputs = 4, numOutputs = 1, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val cfg = ActivationBufferConfig(cuCfg, numBanks = cuCfg.numInputs, wordsPerBank = 1, depth = 3,
      readLatency = 2, writeLatency = 2)
    test(new ActivationBuffer(cfg)) { c =>
      val samples = Seq(Seq(11, 12, 13, 14), Seq(21, 22, 23, 24), Seq(31, 32, 33, 34))
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(0.U) // unicast
      // c.io.shiftIn.poke(true.B)

      for ((s, r) <- samples.zipWithIndex) {
        c.io.writeAddr.poke(r.U)
        c.io.writeEn.poke(true.B)
        c.io.wmask.foreach(_.poke(true.B))
        for ((v, i) <- s.zipWithIndex) c.io.wdata(i).poke(v.U)
        c.clock.step()
      }
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
      c.clock.step(cfg.writeLatency)

      for ((s, r) <- samples.zipWithIndex) {
        c.io.readAddr.poke(r.U)
        c.io.readEn.poke(true.B)
        c.clock.step(cfg.readLatency)
        c.io.operandOut.valid.expect(true.B)
        for ((v, i) <- s.zipWithIndex) c.io.operandOut.bits(i).expect(v.U)
      }
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
    }
  }

  it should "deliver a whole sample from a single wide bank (one bank, several inputs, unicast)" in {
    val cuCfg = ComputeUnitConfig(numInputs = 4, numOutputs = 1, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val cfg = ActivationBufferConfig(cuCfg, numBanks = 1, wordsPerBank = cuCfg.numInputs, depth = 3,
      readLatency = 2, writeLatency = 2)
    test(new ActivationBuffer(cfg)) { c =>
      val samples = Seq(Seq(1, 2, 3, 4), Seq(5, 6, 7, 8), Seq(9, 10, 11, 12))
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(0.U) // unicast
      // c.io.shiftIn.poke(true.B)

      for ((s, r) <- samples.zipWithIndex) {
        c.io.writeAddr.poke(r.U)
        c.io.writeEn.poke(true.B)
        c.io.wmask.foreach(_.poke(true.B))
        for ((v, i) <- s.zipWithIndex) c.io.wdata(i).poke(v.U)
        c.clock.step()
      }
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
      c.clock.step(cfg.writeLatency)

      for ((s, r) <- samples.zipWithIndex) {
        c.io.readAddr.poke(r.U)
        c.io.readEn.poke(true.B)
        c.clock.step(cfg.readLatency)
        c.io.operandOut.valid.expect(true.B)
        for ((v, i) <- s.zipWithIndex) c.io.operandOut.bits(i).expect(v.U)
      }
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
    }
  }

  it should "select the right window via rotate (several banks, several sample groups packed side by side, unicast)" in {
    val cuCfg = ComputeUnitConfig(numInputs = 2, numOutputs = 1, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    // 3 independent sample streams packed side by side across the bank
    // dimension at the same addresses: A at banks [0,1], B at [2,3], C at [4,5]
    val cfg = ActivationBufferConfig(cuCfg, numBanks = 6, wordsPerBank = 1, depth = 3,
      readLatency = 2, writeLatency = 2)
    test(new ActivationBuffer(cfg)) { c =>
      val groupA = Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6))
      val groupB = Seq(Seq(10, 20), Seq(30, 40), Seq(50, 60))
      val groupC = Seq(Seq(100, 101), Seq(102, 103), Seq(104, 105))
      c.io.fanout.poke(0.U) // unicast
      // c.io.shiftIn.poke(true.B)

      for (r <- 0 until cfg.depth) {
        c.io.writeAddr.poke(r.U)
        c.io.writeEn.poke(true.B)
        c.io.wmask.foreach(_.poke(true.B))
        val packed = groupA(r) ++ groupB(r) ++ groupC(r)
        for ((v, word) <- packed.zipWithIndex) c.io.wdata(word).poke(v.U)
        c.clock.step()
      }
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
      c.clock.step(cfg.writeLatency)

      for ((group, shamt) <- Seq((groupA, 0), (groupB, 2), (groupC, 4))) {
        for ((s, r) <- group.zipWithIndex) {
          c.io.readAddr.poke(r.U)
          c.io.readEn.poke(true.B)
          c.io.rotate.poke(shamt.U)
          c.clock.step(cfg.readLatency)
          c.io.operandOut.valid.expect(true.B)
          for ((v, i) <- s.zipWithIndex) c.io.operandOut.bits(i).expect(v.U)
        }
      }
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
    }
  }

  it should "broadcast one input lane to every output lane when fanout = log2(numInputs)" in {
    val cuCfg = ComputeUnitConfig(numInputs = 4, numOutputs = 1, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    // depth=2 (not 1): depth=1 would make sram.addrWidth = log2Ceil(1) = 0,
    // a zero-width port chiseltest can't poke -- same bug class as the
    // earlier rows=1 ComputeUnitConfig case, worked around the same way
    val cfg = ActivationBufferConfig(cuCfg, numBanks = 4, wordsPerBank = 1, depth = 2,
      readLatency = 1, writeLatency = 1)
    test(new ActivationBuffer(cfg)) { c =>
      val sample = Seq(1, 2, 3, 4)
      // c.io.shiftIn.poke(true.B)

      c.io.writeAddr.poke(0.U)
      c.io.writeEn.poke(true.B)
      c.io.wmask.foreach(_.poke(true.B))
      for ((v, i) <- sample.zipWithIndex) c.io.wdata(i).poke(v.U)
      c.clock.step()
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
      c.clock.step(cfg.writeLatency)

      c.io.readAddr.poke(0.U)
      c.io.readEn.poke(true.B)
      c.io.fanout.poke(2.U) // log2(4) = broadcast

      for (shamt <- sample.indices) {
        c.io.rotate.poke(shamt.U)
        c.clock.step(cfg.readLatency)
        c.io.operandOut.valid.expect(true.B)
        for (i <- sample.indices) c.io.operandOut.bits(i).expect(sample(shamt).U)
      }
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
    }
  }

  it should "multicast in groups of 2 when fanout = 1 (between unicast and broadcast)" in {
    val cuCfg = ComputeUnitConfig(numInputs = 4, numOutputs = 1, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    // depth=2, not 1 -- see the broadcast test above for why
    val cfg = ActivationBufferConfig(cuCfg, numBanks = 4, wordsPerBank = 1, depth = 2,
      readLatency = 1, writeLatency = 1)
    test(new ActivationBuffer(cfg)) { c =>
      val sample = Seq(1, 2, 3, 4)
      // c.io.shiftIn.poke(true.B)

      c.io.writeAddr.poke(0.U)
      c.io.writeEn.poke(true.B)
      c.io.wmask.foreach(_.poke(true.B))
      for ((v, i) <- sample.zipWithIndex) c.io.wdata(i).poke(v.U)
      c.clock.step()
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
      c.clock.step(cfg.writeLatency)

      c.io.readAddr.poke(0.U)
      c.io.readEn.poke(true.B)
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(1.U) // groupSize = 2
      c.clock.step(cfg.readLatency)

      c.io.operandOut.valid.expect(true.B)
      // output lanes 0,1 share input lane 0; 2,3 share input lane 1
      val expected = Seq(sample(0), sample(0), sample(1), sample(1))
      for (i <- sample.indices) c.io.operandOut.bits(i).expect(expected(i).U)
      c.io.readEn.poke(false.B)
      c.io.writeEn.poke(false.B)
    }
  }

  "ActivationBuffer -> SystolicArray wiring" should "produce the same matmul as poking operandIn directly" in {
    // numInputs must be a power of two here too (feeds ActivationBuffer's
    // LaneSwitch), unlike SystolicArrayTest's standalone numInputs=3
    val cuCfg = ComputeUnitConfig(numInputs = 4, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val abReadLatency = 1
    test(
      new ActivationBufferSystolicHarness(
        cuCfg,
        numBanks = cuCfg.numInputs,
        wordsPerBank = 1,
        depth = 2,
        abReadLatency = abReadLatency,
        abWriteLatency = 1,
      )
    ) { c =>
      // weights are loaded directly through the compute unit's own
      // address-based port -- no handshake, the test plays the decoder
      // W (4x2) = [[1,2],[3,4],[5,6],[7,8]]
      val weightRows = Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6), Seq(7, 8))
      c.io.weightWriteEn.poke(true.B)
      for ((row, r) <- weightRows.zipWithIndex) {
        c.io.weightWriteAddr.poke(r.U)
        for ((v, i) <- row.zipWithIndex) c.io.weightWriteData(i).poke(toU(v, cuCfg.weightWidth))
        c.clock.step()
      }
      c.io.weightWriteEn.poke(false.B)

      // samples (2x4) . W: [1,1,1,1].W = [16,20]; [2,0,-1,1].W = [4,6]
      val samples = Seq(Seq(1, 1, 1, 1), Seq(2, 0, -1, 1))
      val expected = Seq(Seq(16, 20), Seq(4, 6))

      for ((s, r) <- samples.zipWithIndex) {
        c.io.abAddr.poke(r.U)
        c.io.abEn.poke(true.B)
        c.io.abWen.poke(true.B)
        for ((v, i) <- s.zipWithIndex) c.io.abWdata(i).poke(toU(v, cuCfg.inputWidth))
        c.clock.step()
      }
      c.io.abEn.poke(false.B)
      c.io.abWen.poke(false.B)
      c.clock.step(1) // abWriteLatency

      c.io.abRotate.poke(0.U)
      c.io.abFanout.poke(0.U) // unicast
      // c.io.abShiftIn.poke(true.B)

      for (r <- samples.indices) {
        c.io.abAddr.poke(r.U)
        c.io.abEn.poke(true.B)
        c.clock.step()
      }
      c.io.abEn.poke(false.B)

      // each sample's operandOut beat (and therefore operandIn.valid) lands
      // abReadLatency cycles after its request was presented, so the
      // fixed systolic latency is measured from there, not from the
      // request loop itself
      val latency = cuCfg.numInputs + cuCfg.numOutputs - 1
      c.clock.step(abReadLatency + latency - samples.length)

      for (exp <- expected) {
        c.io.resultOut.valid.expect(true.B)
        for ((v, i) <- exp.zipWithIndex) c.io.resultOut.bits(i).expect(v.S)
        c.clock.step()
      }
    }
  }

  // Line buffer tests disabled together with ActivationBuffer's line buffer
  // (im2col is now done by accumulating kn2row). Kept for reference.
  // "ActivationBuffer" should "hold windowDepth-1 resident rows and present a sliding window across consecutive reads (line buffer)" in {
  //   // 3-row window (like a 3x3 conv kernel's row dimension), 2 elements
  //   // per row -- rowWidth = numInputs(6) / windowDepth(3) = 2
  //   val cuCfg = ComputeUnitConfig(numInputs = 6, numOutputs = 1, inputWidth = 8, weightWidth = 8, outputWidth = 20)
  //   val cfg = ActivationBufferConfig(cuCfg, numBanks = 2, wordsPerBank = 1, depth = 5,
  //     readLatency = 2, writeLatency = 2, windowDepth = 3)
  //   test(new ActivationBuffer(cfg)) { c =>
  //     val rows = Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6), Seq(7, 8), Seq(9, 10))
  //     c.io.rotate.poke(0.U)
  //     c.io.fanout.poke(0.U) // unicast
  //     c.io.shiftIn.poke(true.B) // ADVANCE: every read is a new row
//
  //     for ((row, r) <- rows.zipWithIndex) {
  //       c.io.addr.poke(r.U)
  //       c.io.en.poke(true.B)
  //       c.io.wen.poke(true.B)
  //       for ((v, i) <- row.zipWithIndex) c.io.wdata(i).poke(v.U)
  //       c.clock.step()
  //     }
  //     c.io.en.poke(false.B)
  //     c.io.wen.poke(false.B)
  //     c.clock.step(cfg.writeLatency)
//
  //     // issue exactly one read request (en high for a single cycle) and
  //     // wait out the rest of readLatency -- holding en high across the
  //     // whole step(readLatency) would issue a duplicate request every
  //     // cycle instead of one, which corrupts the line buffer's shift
  //     // sequence (harmless in the other, stateless tests above, but not here)
  //     def issueRead(addr: Int): Unit = {
  //       c.io.addr.poke(addr.U)
  //       c.io.en.poke(true.B)
  //       c.io.wen.poke(false.B)
  //       c.clock.step()
  //       c.io.en.poke(false.B)
  //       c.clock.step(cfg.readLatency - 1)
  //     }
//
  //     // read rows 0,1 to warm up the history -- their windows still have
  //     // zero-padded not-yet-filled slots, so not checked
  //     issueRead(0)
  //     issueRead(1)
//
  //     // from row 2 onward every window is 3 real, consecutive rows
  //     val expectedWindows = Seq(
  //       rows(0) ++ rows(1) ++ rows(2),
  //       rows(1) ++ rows(2) ++ rows(3),
  //       rows(2) ++ rows(3) ++ rows(4),
  //     )
  //     for ((exp, i) <- expectedWindows.zipWithIndex) {
  //       issueRead(i + 2)
  //       c.io.operandOut.valid.expect(true.B)
  //       for ((v, j) <- exp.zipWithIndex) c.io.operandOut.bits(j).expect(v.U)
  //     }
  //   }
  // }
//
  // it should "let shiftIn=false re-select a different rotate window without disturbing history (RESELECT)" in {
  //   val cuCfg = ComputeUnitConfig(numInputs = 4, numOutputs = 1, inputWidth = 8, weightWidth = 8, outputWidth = 20)
  //   val cfg = ActivationBufferConfig(cuCfg, numBanks = 4, wordsPerBank = 1, depth = 3,
  //     readLatency = 2, writeLatency = 2, windowDepth = 2)
  //   test(new ActivationBuffer(cfg)) { c =>
  //     val row0 = Seq(1, 2, 3, 4)
  //     val row1 = Seq(10, 20, 30, 40)
//
  //     for ((row, r) <- Seq(row0, row1).zipWithIndex) {
  //       c.io.addr.poke(r.U)
  //       c.io.en.poke(true.B)
  //       c.io.wen.poke(true.B)
  //       for ((v, i) <- row.zipWithIndex) c.io.wdata(i).poke(v.U)
  //       c.clock.step()
  //     }
  //     c.io.en.poke(false.B)
  //     c.io.wen.poke(false.B)
  //     c.clock.step(cfg.writeLatency)
//
  //     def issueRead(addr: Int, rotate: Int, shiftIn: Boolean): Unit = {
  //       c.io.addr.poke(addr.U)
  //       c.io.en.poke(true.B)
  //       c.io.wen.poke(false.B)
  //       c.io.rotate.poke(rotate.U)
  //       c.io.fanout.poke(0.U)
  //       c.io.shiftIn.poke(shiftIn.B)
  //       c.clock.step()
  //       c.io.en.poke(false.B)
  //       c.clock.step(cfg.readLatency - 1)
  //     }
//
  //     // ADVANCE: load row0 (warm-up, not checked)
  //     issueRead(0, rotate = 0, shiftIn = true)
//
  //     // two reads at the SAME address (row1), sweeping rotate. The
  //     // first is RESELECT (shiftIn=false): history must stay [row0] so
  //     // both reads see the same window [row0, row1], just different
  //     // columns. The second (last of the group) commits the shift.
  //     issueRead(1, rotate = 0, shiftIn = false)
  //     c.io.operandOut.valid.expect(true.B)
  //     Seq(1, 2, 10, 20).zipWithIndex.foreach { case (v, i) => c.io.operandOut.bits(i).expect(v.U) }
//
  //     issueRead(1, rotate = 2, shiftIn = true)
  //     c.io.operandOut.valid.expect(true.B)
  //     Seq(3, 4, 30, 40).zipWithIndex.foreach { case (v, i) => c.io.operandOut.bits(i).expect(v.U) }
  //   }
  // }

  "ActivationBuffer with address groups" should "gather a beat from, and scatter a write to, a different row per group" in {
    val cuCfg = ComputeUnitConfig(numInputs = 4, numOutputs = 1, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    // banks 0-1 are group 0 (readAddr / writeAddr), banks 2-3 group 1
    val cfg = ActivationBufferConfig(cuCfg, numBanks = 4, wordsPerBank = 1, depth = 4,
      readLatency = 2, writeLatency = 2, portType = BankPortType.OneR1W, addrGroups = 2)
    test(new ActivationBuffer(cfg)) { c =>
      val rg = c.io.readGroupAddr.get(0)
      val wg = c.io.writeGroupAddr.get(0)
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(0.U)
      c.io.readEn.poke(false.B)

      def write(a0: Int, a1: Int, s: Seq[Int]): Unit = {
        c.io.writeAddr.poke(a0.U)
        wg.poke(a1.U)
        c.io.writeEn.poke(true.B)
        c.io.wmask.foreach(_.poke(true.B))
        for ((v, i) <- s.zipWithIndex) c.io.wdata(i).poke(v.U)
        c.clock.step()
        c.io.writeEn.poke(false.B)
      }
      def read(a0: Int, a1: Int, expected: Seq[Int]): Unit = {
        c.io.readAddr.poke(a0.U)
        rg.poke(a1.U)
        c.io.readEn.poke(true.B)
        c.clock.step()
        c.io.readEn.poke(false.B)
        c.clock.step(cfg.readLatency - 1)
        c.io.operandOut.valid.expect(true.B)
        for ((v, i) <- expected.zipWithIndex) c.io.operandOut.bits(i).expect(v.U)
      }

      // whole rows: both groups at the same row
      for ((s, r) <- Seq(Seq(10, 11, 12, 13), Seq(20, 21, 22, 23), Seq(30, 31, 32, 33)).zipWithIndex) write(r, r, s)
      c.clock.step(cfg.writeLatency)
      read(0, 2, Seq(10, 11, 32, 33))
      read(2, 1, Seq(30, 31, 22, 23))

      // scatter: banks 0-1 to row 3, banks 2-3 to row 0
      write(3, 0, Seq(40, 41, 42, 43))
      c.clock.step(cfg.writeLatency)
      read(0, 0, Seq(10, 11, 42, 43))
      read(3, 1, Seq(40, 41, 22, 23))
    }
  }

  "ActivationBuffer" should "interleave a multicast, the select delayed with the read" in {
    val cuCfg = ComputeUnitConfig(numInputs = 4, numOutputs = 1, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val cfg = ActivationBufferConfig(cuCfg, numBanks = 4, wordsPerBank = 1, depth = 2, readLatency = 2, writeLatency = 2)
    test(new ActivationBuffer(cfg)) { c =>
      c.io.writeAddr.poke(0.U)
      c.io.writeEn.poke(true.B)
      c.io.wmask.foreach(_.poke(true.B))
      for (i <- 0 until 4) c.io.wdata(i).poke((10 + i).U)
      c.clock.step()
      c.io.writeEn.poke(false.B)
      c.clock.step(cfg.writeLatency)

      c.io.readAddr.poke(0.U)
      c.io.readEn.poke(true.B)
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(1.U)
      c.io.interleave.poke(1.U)
      c.clock.step()
      // a later select must not affect the beat in flight
      c.io.readEn.poke(false.B)
      c.io.interleave.poke(0.U)
      c.clock.step(cfg.readLatency - 1)
      c.io.operandOut.valid.expect(true.B)
      for ((v, j) <- Seq(10, 11, 10, 11).zipWithIndex) c.io.operandOut.bits(j).expect(v.U)
    }
  }

  "ActivationBuffer with a write switch" should "place a narrow write anywhere in the row, writing only its words" in {
    val cuCfg = ComputeUnitConfig(numInputs = 4, numOutputs = 1, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val cfg = ActivationBufferConfig(cuCfg, numBanks = 4, wordsPerBank = 1, depth = 2, readLatency = 2, writeLatency = 2,
      portType = BankPortType.OneR1W)
    test(new ActivationBuffer(cfg)) { c =>
      c.io.readEn.poke(false.B)
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(0.U)
      c.io.interleave.poke(0.U)

      // a 2-word write (words 0-1 of wdata carry data) at word `at`
      def write(s: Seq[Int], mask: Seq[Boolean], at: Int, fanout: Int = 0, interleave: Int = 0): Unit = {
        c.io.writeAddr.poke(0.U)
        c.io.writeEn.poke(true.B)
        for (i <- 0 until 4) {
          c.io.wdata(i).poke(s(i).U)
          c.io.wmask(i).poke(mask(i).B)
        }
        c.io.writeRotate.poke(((4 - at) % 4).U)
        c.io.writeFanout.poke(fanout.U)
        c.io.writeInterleave.poke(interleave.U)
        c.clock.step()
        c.io.writeEn.poke(false.B)
        c.clock.step(cfg.writeLatency)
      }
      def row(expected: Seq[Int]): Unit = {
        c.io.readAddr.poke(0.U)
        c.io.readEn.poke(true.B)
        c.clock.step()
        c.io.readEn.poke(false.B)
        c.clock.step(cfg.readLatency - 1)
        for ((v, i) <- expected.zipWithIndex) c.io.operandOut.bits(i).expect(v.U)
      }
      val beat = Seq(true, true, false, false)

      write(Seq(1, 2, 3, 4), Seq.fill(4)(true), at = 0)
      row(Seq(1, 2, 3, 4))
      write(Seq(9, 8, 0, 0), beat, at = 2)
      row(Seq(1, 2, 9, 8))
      // across the row end: words 3, 0
      write(Seq(7, 6, 0, 0), beat, at = 3)
      row(Seq(6, 2, 9, 7))
      // a 2-word tile replicated over the row
      write(Seq(5, 4, 0, 0), beat, at = 0, fanout = 1, interleave = 1)
      row(Seq(5, 4, 5, 4))
      // without a mask the zero padding is written too
      write(Seq(3, 3, 0, 0), Seq.fill(4)(true), at = 1)
      row(Seq(0, 3, 3, 0))
    }
  }
}
