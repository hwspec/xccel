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
    val cfg = XccelConfig(cu, wbCfg, abCfg, accCfg, requantizedWidth = cu.outputWidth,
      programLength = 5, maxRunLength = 4)

    test(new Xccel(cfg)) { c =>
      val weightRows = Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6), Seq(7, 8))
      val samples = Seq(Seq(1, 1, 1, 1), Seq(2, 0, -1, 1))
      val expected = Seq(Seq(16, 20), Seq(4, 6))

      c.io.start.poke(false.B)
      c.io.progWriteEn.poke(false.B)
      c.io.requantParamWriteEn.poke(false.B)
      c.io.accReadParamSet.poke(0.U)
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

      // microcode, hand-scheduled from the fixed latencies:
      // WeightBuffer read at t -> weightOut at t+2 (wb readLatency)
      // ActivationBuffer read at t -> resultOut at t+1+5 (ab readLatency +
      // systolic latency numInputs+numOutputs-1)
      val p = new MicroProgramLoader(c.clock, c.io.progWriteEn, c.io.progWriteAddr, c.io.progWriteData)
      // t0-1: read weight rows 0,1
      p.word(length = 2) { w =>
        w.wbRead.en.poke(true.B)
        w.wbRead.addrStride.poke(1.U)
      }
      // t2-3: read weight rows 2,3; rows 0,1 land on the weight port
      p.word(length = 2) { w =>
        w.wbRead.en.poke(true.B)
        w.wbRead.addr.poke(2.U)
        w.wbRead.addrStride.poke(1.U)
        w.weightPort.en.poke(true.B)
        w.weightPort.addrStride.poke(1.U)
      }
      // t4-5: stream samples 0,1; rows 2,3 land on the weight port
      p.word(length = 2) { w =>
        w.abRead.en.poke(true.B)
        w.abRead.addrStride.poke(1.U)
        w.weightPort.en.poke(true.B)
        w.weightPort.addr.poke(2.U)
        w.weightPort.addrStride.poke(1.U)
      }
      // t6-9: pipeline fill
      p.word(length = 4)()
      // t10-11: results for samples 0,1 land in Accumulator rows 0,1
      // (overwrite, accumulate = false)
      p.word(length = 2, halt = true) { w =>
        w.accWrite.addrStride.poke(1.U)
      }
      p.done()

      c.io.start.poke(true.B)
      c.clock.step()
      c.io.start.poke(false.B)
      c.io.running.expect(true.B)

      // the program self-terminates on its halt word
      var cycles = 0
      while (c.io.running.peek().litToBoolean) {
        c.clock.step()
        cycles += 1
        assert(cycles < 64, "program never halted")
      }
      assert(cycles == 12)
      c.clock.step(accCfg.commitLatency)

      // host drain, now that the program's accRead slot is idle
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

  // ---- 3x3 conv, three ways --------------------------------------------
  // A stride-1, unpadded 3x3 conv (one C_in tile, one C_out tile) run on the
  // same hardware as accumulating kn2row (full input width or valid outputs
  // only) and as im2col; only the buffer layouts and the microcode's address
  // sequences differ. Schedules follow the conv-cycles.py rules, and each
  // test checks the measured cycles against its prediction.

  // One pass: operands from ActivationBuffer abAddrs, weight tile at
  // WeightBuffer rows wbBase until wbBase+R-1, results to Accumulator accAddrs.
  case class ConvPass(abAddrs: Seq[Int], wbBase: Int, accAddrs: Seq[Int], accumulate: Boolean)

  case class ConvPlan(
    abRows: Seq[Seq[Int]], // R values per row
    wbRows: Seq[Seq[Int]], // C values per row
    passes: Seq[ConvPass],
    outputs: Seq[(Int, Seq[Int])], // (Accumulator row, expected C lanes) of every valid output
  )

  sealed trait ConvMode
  case object FullWidth extends ConvMode // kn2row, OH*W outputs per pass, the x >= OW ones discarded
  case object ValidOnly extends ConvMode // kn2row, only the OH*OW valid outputs per pass
  case object Im2col extends ConvMode // host-materialized im2col matrix, K*K*C_in cut into R-wide chunks

  def convPlan(mode: ConvMode, h: Int, w: Int, cin: Int, cout: Int, k: Int, rows: Int, cols: Int, seed: Int): ConvPlan = {
    require(cout <= cols, "one C_out tile only")
    val oh = h - k + 1
    val ow = w - k + 1
    val rnd = new scala.util.Random(seed)
    val input = Seq.fill(h, w, cin)(rnd.nextInt(7) - 3)
    val weights = Seq.fill(k, k, cin, cout)(rnd.nextInt(7) - 3) // [kh][kw][ci][co]
    def ref(y: Int, x: Int)(co: Int): Int =
      (for (kh <- 0 until k; kw <- 0 until k; ci <- 0 until cin)
        yield input(y + kh)(x + kw)(ci) * weights(kh)(kw)(ci)(co)).sum
    def lanes(f: Int => Int): Seq[Int] = Seq.tabulate(cols)(co => if (co < cout) f(co) else 0)
    val positions = for (kh <- 0 until k; kw <- 0 until k) yield (kh, kw)

    mode match {
      case FullWidth | ValidOnly =>
        require(cin <= rows, "one C_in tile only")
        // one row per pixel (addr y*W + x), plus K-1 rows of margin that the
        // full-width passes' discarded outputs read
        val abRows = Seq.tabulate(h * w + k - 1)(p =>
          Seq.tabulate(rows)(ci => if (p < h * w && ci < cin) input(p / w)(p % w)(ci) else 0))
        // one R-row 1x1 weight tile per kernel position
        val wbRows = for ((kh, kw) <- positions; ci <- 0 until rows)
          yield lanes(co => if (ci < cin) weights(kh)(kw)(ci)(co) else 0)
        val full = mode == FullWidth
        def acc(y: Int, x: Int) = if (full) y * w + x else y * ow + x
        val outs = for (y <- 0 until oh; x <- 0 until (if (full) w else ow)) yield (y, x)
        val passes = for (((kh, kw), pos) <- positions.zipWithIndex) yield ConvPass(
          abAddrs = outs.map { case (y, x) => (y + kh) * w + x + kw }, // input-side shift
          wbBase = pos * rows,
          accAddrs = outs.map { case (y, x) => acc(y, x) },
          accumulate = pos > 0,
        )
        ConvPlan(abRows, wbRows, passes, for (y <- 0 until oh; x <- 0 until ow) yield acc(y, x) -> lanes(ref(y, x)))

      case Im2col =>
        // im2col column kk = (kh*K + kw)*C_in + ci, cut into R-wide chunks;
        // a chunk may span several kernel positions
        val kred = k * k * cin
        val chunks = (kred + rows - 1) / rows
        val npix = oh * ow
        def decode(kk: Int) = (kk / cin / k, kk / cin % k, kk % cin)
        val abRows = for (ch <- 0 until chunks; p <- 0 until npix) yield Seq.tabulate(rows) { i =>
          val kk = ch * rows + i
          if (kk < kred) { val (kh, kw, ci) = decode(kk); input(p / ow + kh)(p % ow + kw)(ci) } else 0
        }
        // the host's reordering of the kernel into W' (K*K*C_in x C_out), zero-padded to whole chunks
        val wbRows = for (kk <- 0 until chunks * rows)
          yield lanes(co => if (kk < kred) { val (kh, kw, ci) = decode(kk); weights(kh)(kw)(ci)(co) } else 0)
        val passes = for (ch <- 0 until chunks) yield ConvPass(
          abAddrs = Seq.tabulate(npix)(p => ch * npix + p),
          wbBase = ch * rows,
          accAddrs = Seq.tabulate(npix)(p => p),
          accumulate = ch > 0,
        )
        ConvPlan(abRows, wbRows, passes, for (y <- 0 until oh; x <- 0 until ow) yield (y * ow + x) -> lanes(ref(y, x)))
    }
  }

  // Per-cycle control and cycles from start to halt, by the conv-cycles.py
  // rules: passes back to back with period P, each loading its R weight rows
  // into the idle weight slot while the previous pass streams.
  def convSchedule(plan: ConvPlan, rows: Int, cols: Int, wbRL: Int, abRL: Int, commitLatency: Int): (Seq[CycleControl], Int) = {
    val stream = plan.passes.head.abAddrs.length
    require(plan.passes.forall(_.abAddrs.length == stream), "all passes must stream the same number of operands")
    val latency = rows + cols - 1
    val period = Seq(stream, rows, commitLatency, (stream + cols) / 2).max // (stream+cols)/2 = ceil((stream+cols-1)/2)
    val s0 = math.max(0, wbRL + 1 - abRL)
    def S(t: Int) = s0 + t * period
    val total = S(plan.passes.length - 1) + stream + abRL + latency
    val cycles = Array.fill(total)(CycleControl())
    for ((pass, t) <- plan.passes.zipWithIndex) {
      val slot = t % 2 == 1
      val wld = S(t) + abRL - wbRL - 1 // row r lands just before the first operand reaches array row r
      for (r <- 0 until rows) {
        val wp = wld + wbRL + r
        require(!cycles(wld + r).wbEn && !cycles(wp).wpEn, s"weight loads overlap in pass $t")
        cycles(wld + r) = cycles(wld + r).copy(wbEn = true, wbAddr = pass.wbBase + r)
        cycles(wp) = cycles(wp).copy(wpEn = true, wpAddr = r, wpSlot = slot)
      }
      for (i <- 0 until stream) {
        cycles(S(t) + i) = cycles(S(t) + i).copy(abEn = true, abAddr = pass.abAddrs(i))
        val arrive = S(t) + abRL + i
        cycles(arrive) = cycles(arrive).copy(weightSelect = slot)
        val result = arrive + latency
        cycles(result) = cycles(result).copy(accAddr = pass.accAddrs(i), accumulate = pass.accumulate)
      }
    }
    (cycles.toSeq, total)
  }

  def convTest(
    desc: String,
    mode: ConvMode,
    cin: Int,
    rows: Int,
    predicted: Int,
    portType: BankPortType = BankPortType.OneRW,
  ): Unit = it should s"compute a 3x3 conv by $desc in $predicted cycles (conv-cycles.py prediction)" in {
    val (h, w, k, cout, cols) = (4, 4, 3, 2, 2)
    val plan = convPlan(mode, h, w, cin, cout, k, rows, cols, seed = 7)
    val cu = ComputeUnitConfig(numInputs = rows, numOutputs = cols, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val wbCfg = WeightBufferConfig(cu, numBanks = cols, wordsPerBank = 1, tiles = plan.wbRows.length / rows,
      readLatency = 2, writeLatency = 2, portType = portType)
    val abCfg = ActivationBufferConfig(cu, numBanks = rows, wordsPerBank = 1, depth = plan.abRows.length,
      readLatency = 1, writeLatency = 1, portType = portType)
    val accCfg = AccumulatorConfig(cu, numBanks = cols, wordsPerBank = 1,
      depth = plan.passes.flatMap(_.accAddrs).max + 1, readLatency = 2, writeLatency = 2)
    val (cycles, total) = convSchedule(plan, rows, cols, wbCfg.readLatency, abCfg.readLatency, accCfg.commitLatency)
    assert(total == predicted, s"schedule gives $total cycles, conv-cycles.py predicts $predicted")
    val maxRunLength = 16
    val words = CycleProgram.compress(cycles, maxRunLength)
    val cfg = XccelConfig(cu, wbCfg, abCfg, accCfg, requantizedWidth = cu.outputWidth,
      programLength = words.length, maxRunLength = maxRunLength)

    test(new Xccel(cfg)) { c =>
      c.io.start.poke(false.B)
      c.io.progWriteEn.poke(false.B)
      c.io.requantParamWriteEn.poke(false.B)
      c.io.accReadParamSet.poke(0.U)
      c.io.accReadEn.poke(false.B)
      c.io.abWriteEn.poke(false.B)

      c.io.wbWriteEn.poke(true.B)
      for ((row, a) <- plan.wbRows.zipWithIndex) {
        c.io.wbWriteAddr.poke(a.U)
        for ((v, i) <- row.zipWithIndex) c.io.wbWriteData(i).poke(toU(v, cu.weightWidth))
        c.clock.step()
      }
      c.io.wbWriteEn.poke(false.B)
      c.io.abWriteEn.poke(true.B)
      for ((row, a) <- plan.abRows.zipWithIndex) {
        c.io.abWriteAddr.poke(a.U)
        for ((v, i) <- row.zipWithIndex) c.io.abWriteData(i).poke(toU(v, cu.inputWidth))
        c.clock.step()
      }
      c.io.abWriteEn.poke(false.B)
      c.clock.step(math.max(wbCfg.writeLatency, abCfg.writeLatency))

      CycleProgram.load(new MicroProgramLoader(c.clock, c.io.progWriteEn, c.io.progWriteAddr, c.io.progWriteData), words)

      c.io.start.poke(true.B)
      c.clock.step()
      c.io.start.poke(false.B)
      var n = 0
      while (c.io.running.peek().litToBoolean) {
        c.clock.step()
        n += 1
        assert(n <= total, "program never halted")
      }
      assert(n == total)
      c.clock.step(accCfg.commitLatency)

      for ((addr, exp) <- plan.outputs) {
        c.io.accReadAddr.poke(addr.U)
        c.io.accReadEn.poke(true.B)
        c.clock.step(accCfg.readLatency)
        c.io.accumOut.valid.expect(true.B)
        for ((v, i) <- exp.zipWithIndex) c.io.accumOut.bits(i).expect(toU(v, cu.outputWidth))
      }
      c.io.accReadEn.poke(false.B)
    }
  }

  // 4x4 input, C_out = 2 -> 2x2 output. Predictions:
  //   python3 conv-cycles.py --preset example                    (C_in = 2, 2x2 array)
  //   python3 conv-cycles.py --conv 4 4 1 2 3 --rows 4 --cols 2  (C_in = 1, 4x2 array)
  for (portType <- Seq(BankPortType.OneRW, BankPortType.OneR1W, BankPortType.TwoRW))
    convTest(s"accumulating kn2row, full input width (C_in = 2, 2x2 array, WB/AB $portType)",
      FullWidth, cin = 2, rows = 2, predicted = 78, portType)
  convTest("accumulating kn2row, valid outputs only (C_in = 2, 2x2 array)", ValidOnly, cin = 2, rows = 2, predicted = 42)
  convTest("im2col (C_in = 2, 2x2 array)", Im2col, cin = 2, rows = 2, predicted = 42)
  convTest("accumulating kn2row, valid outputs only (C_in = 1, 4x2 array)", ValidOnly, cin = 1, rows = 4, predicted = 44)
  convTest("im2col, chunks spanning kernel positions (C_in = 1, 4x2 array)", Im2col, cin = 1, rows = 4, predicted = 20)

  for (portType <- Seq(BankPortType.OneRW, BankPortType.OneR1W))
  it should s"write weights and activations during the run from program-scheduled writes, the host supplying only data (WB/AB $portType)" in {
    // same matmul as the first test, but nothing is preloaded: the program
    // writes W's rows and the samples into WB/AB itself (addr/en from its
    // wbWrite/abWrite slots) while the host puts each cycle's data on
    // wbWriteData/abWriteData. With OneR1W a row is read back as soon as it
    // is committed, overlapping later writes; with OneRW (one port) all
    // writes of a buffer finish before its reads start.
    val cu = ComputeUnitConfig(numInputs = 4, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val wbCfg = WeightBufferConfig(cu, numBanks = cu.numOutputs, wordsPerBank = 1, tiles = 1,
      readLatency = 2, writeLatency = 2, portType = portType)
    val abCfg = ActivationBufferConfig(cu, numBanks = cu.numInputs, wordsPerBank = 1, depth = 2,
      readLatency = 1, writeLatency = 1, portType = portType)
    val accCfg = AccumulatorConfig(cu, numBanks = cu.numOutputs, wordsPerBank = 1, depth = 2,
      readLatency = 2, writeLatency = 2)
    val latency = cu.numInputs + cu.numOutputs - 1

    val weightRows = Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6), Seq(7, 8))
    val samples = Seq(Seq(1, 1, 1, 1), Seq(2, 0, -1, 1))
    val expected = Seq(Seq(16, 20), Seq(4, 6))

    val sharedPort = portType == BankPortType.OneRW
    // a row written in cycle t is readable from t + writeLatency + 1
    val wbWrite = (r: Int) => r
    val wbRead = (r: Int) => wbWrite(r) + wbCfg.writeLatency + 1 + (if (sharedPort) weightRows.length else 0)
    val weightPort = (r: Int) => wbRead(r) + wbCfg.readLatency
    // operand s reaches array row r at abRead(s) + abRL + r, after that row's weight is in
    val abRead = (s: Int) => weightPort(0) + 1 - abCfg.readLatency + s
    val abWrite = (s: Int) => abRead(0) - abCfg.writeLatency - 1 - (samples.length - 1) + s
    val result = (s: Int) => abRead(s) + abCfg.readLatency + latency
    assert(!sharedPort || abWrite(samples.length - 1) < abRead(0))

    val total = result(samples.length - 1) + 1
    val cycles = Array.fill(total)(CycleControl())
    val wbData = scala.collection.mutable.Map.empty[Int, Seq[Int]]
    val abData = scala.collection.mutable.Map.empty[Int, Seq[Int]]
    for ((row, r) <- weightRows.zipWithIndex) {
      cycles(wbWrite(r)) = cycles(wbWrite(r)).copy(wbwEn = true, wbwAddr = r)
      wbData(wbWrite(r)) = row
      cycles(wbRead(r)) = cycles(wbRead(r)).copy(wbEn = true, wbAddr = r)
      cycles(weightPort(r)) = cycles(weightPort(r)).copy(wpEn = true, wpAddr = r)
    }
    for ((smp, s) <- samples.zipWithIndex) {
      cycles(abWrite(s)) = cycles(abWrite(s)).copy(abwEn = true, abwAddr = s)
      abData(abWrite(s)) = smp
      cycles(abRead(s)) = cycles(abRead(s)).copy(abEn = true, abAddr = s)
      cycles(result(s)) = cycles(result(s)).copy(accAddr = s)
    }
    val words = CycleProgram.compress(cycles.toSeq, maxRunLength = 8)
    val cfg = XccelConfig(cu, wbCfg, abCfg, accCfg, requantizedWidth = cu.outputWidth,
      programLength = words.length, maxRunLength = 8)

    test(new Xccel(cfg)) { c =>
      c.io.start.poke(false.B)
      c.io.progWriteEn.poke(false.B)
      c.io.requantParamWriteEn.poke(false.B)
      c.io.accReadParamSet.poke(0.U)
      c.io.wbWriteEn.poke(false.B)
      c.io.abWriteEn.poke(false.B)
      c.io.accReadEn.poke(false.B)
      CycleProgram.load(new MicroProgramLoader(c.clock, c.io.progWriteEn, c.io.progWriteAddr, c.io.progWriteData), words)

      c.io.start.poke(true.B)
      c.clock.step()
      c.io.start.poke(false.B)
      // cycle t of the program: the host presents that cycle's data only
      for (t <- 0 until total) {
        c.io.running.expect(true.B)
        for (row <- wbData.get(t); (v, i) <- row.zipWithIndex) c.io.wbWriteData(i).poke(toU(v, cu.weightWidth))
        for (smp <- abData.get(t); (v, i) <- smp.zipWithIndex) c.io.abWriteData(i).poke(toU(v, cu.inputWidth))
        c.clock.step()
      }
      c.io.running.expect(false.B)
      c.clock.step(accCfg.commitLatency)

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

  // ---- layer-to-layer path -----------------------------------------------
  for (portType <- Seq(BankPortType.OneRW, BankPortType.OneR1W))
  it should s"chain two layers on chip: drain layer 1 through the Requantizer into ActivationBuffer, then run layer 2 on it (AB $portType)" in {
    // Two quantized matmuls (1x1 convs) on a 4x4 array, N samples, 20-bit
    // accumulators and 8-bit activations:
    //   A1 = requant1(X.W1)   per-lane bias and scale, ReLU    (Requantizer set 1)
    //   Y2 = requant0(A1.W2)  per-lane bias and scale, INT8    (Requantizer set 0)
    // A1 never leaves the chip: the program drains layer 1 with accRead,
    // selects set 1 with its requant slot, and writes the result back with
    // abWrite.fromRequantizer. The host drains Y2 with set 0.
    val R = 4; val C = 4; val N = 3
    val cu = ComputeUnitConfig(numInputs = R, numOutputs = C, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    val wbCfg = WeightBufferConfig(cu, numBanks = C, wordsPerBank = 1, tiles = 2, readLatency = 2, writeLatency = 2)
    val abCfg = ActivationBufferConfig(cu, numBanks = R, wordsPerBank = 1, depth = 2 * N,
      readLatency = 1, writeLatency = 1, portType = portType)
    val accCfg = AccumulatorConfig(cu, numBanks = C, wordsPerBank = 1, depth = 2 * N, readLatency = 2, writeLatency = 2)
    val latency = R + C - 1

    val rnd = new scala.util.Random(3)
    val x = Seq.fill(N, R)(rnd.nextInt(15) - 7)
    val w1 = Seq.fill(R, C)(rnd.nextInt(15) - 7)
    val w2 = Seq.fill(C, C)(rnd.nextInt(15) - 7)
    def matmul(a: Seq[Seq[Int]], b: Seq[Seq[Int]]) = a.map(row => b.head.indices.map(j => row.indices.map(i => row(i) * b(i)(j)).sum))
    case class Params(bias: Seq[Int], mult: Seq[Int], shift: Seq[Int], lo: Int, hi: Int)
    val set1 = Params(bias = Seq(10, -20, 5, 0), mult = Seq(3, 5, 1, 7), shift = Seq(3, 4, 1, 5), lo = 0, hi = 127) // ReLU
    val set0 = Params(bias = Seq(-50, 0, 300, 7), mult = Seq(1, 3, 5, 1), shift = Seq(4, 6, 7, 3), lo = -128, hi = 127)
    def requant(v: Seq[Seq[Int]], p: Params) = v.map(_.zipWithIndex.map { case (a, i) =>
      val prod = BigInt(a + p.bias(i)) * p.mult(i)
      val half = if (p.shift(i) == 0) BigInt(0) else BigInt(1) << (p.shift(i) - 1)
      val r = if (p.shift(i) == 0) prod else (if (prod < 0) prod + half - 1 else prod + half) >> p.shift(i)
      r.max(p.lo).min(p.hi).toInt
    })
    val a1 = requant(matmul(x, w1), set1)
    val y2 = requant(matmul(a1, w2), set0)
    assert(a1.flatten.exists(_ > 0) && a1.flatten.contains(0), "layer 1 should exercise both ReLU sides")

    // WeightBuffer rows 0..R-1 = W1 (slot 0), R..2R-1 = W2 (slot 1), both
    // loaded at the start. ActivationBuffer rows 0..N-1 = X, N..2N-1 = Y1.
    // Accumulator rows 0..N-1 = Y1, N..2N-1 = Y2.
    val s0 = wbCfg.readLatency + 1 - abCfg.readLatency
    val lastY1 = s0 + N - 1 + abCfg.readLatency + latency // last layer-1 result beat
    val drain = lastY1 + accCfg.commitLatency // first accRead of Y1
    val lastFeedback = drain + N - 1 + accCfg.readLatency // last abWrite of Y1
    val firstFeedback = drain + accCfg.readLatency
    // a row written in cycle t is readable from t + writeLatency + 1; with
    // one shared port (OneRW) layer 2's reads also wait for the last write
    val s2 = (if (portType == BankPortType.OneRW) lastFeedback else firstFeedback) + abCfg.writeLatency + 1
    val total = s2 + N - 1 + abCfg.readLatency + latency + 1

    val cycles = Array.fill(total)(CycleControl())
    def upd(t: Int)(f: CycleControl => CycleControl): Unit = cycles(t) = f(cycles(t))
    for (r <- 0 until 2 * R) {
      upd(r)(_.copy(wbEn = true, wbAddr = r))
      upd(r + wbCfg.readLatency)(_.copy(wpEn = true, wpAddr = r % R, wpSlot = r >= R))
    }
    for (s <- 0 until N) {
      upd(s0 + s)(_.copy(abEn = true, abAddr = s))
      upd(s0 + s + abCfg.readLatency + latency)(_.copy(accAddr = s))
      upd(drain + s)(_.copy(accrEn = true, accrAddr = s))
      upd(drain + s + accCfg.readLatency)(_.copy(abwEn = true, abwAddr = N + s, abwFromRequantizer = true, requantSet = 1))
      upd(s2 + s)(_.copy(abEn = true, abAddr = N + s))
      upd(s2 + s + abCfg.readLatency)(_.copy(weightSelect = true))
      upd(s2 + s + abCfg.readLatency + latency)(_.copy(accAddr = N + s))
    }
    val words = CycleProgram.compress(cycles.toSeq, maxRunLength = 16)
    val cfg = XccelConfig(cu, wbCfg, abCfg, accCfg, requantizedWidth = cu.inputWidth,
      programLength = words.length, maxRunLength = 16, requantParamSets = 2)
    assert(cfg.feedback)

    test(new Xccel(cfg)) { c =>
      c.io.start.poke(false.B)
      c.io.progWriteEn.poke(false.B)
      c.io.requantParamWriteEn.poke(false.B)
      c.io.accReadParamSet.poke(0.U)
      c.io.accReadEn.poke(false.B)
      c.io.abWriteEn.poke(false.B)
      c.io.wbWriteEn.poke(true.B)
      for ((row, a) <- (w1 ++ w2).zipWithIndex) {
        c.io.wbWriteAddr.poke(a.U)
        for ((v, i) <- row.zipWithIndex) c.io.wbWriteData(i).poke(toU(v, cu.weightWidth))
        c.clock.step()
      }
      c.io.wbWriteEn.poke(false.B)
      c.io.abWriteEn.poke(true.B)
      for ((row, a) <- x.zipWithIndex) {
        c.io.abWriteAddr.poke(a.U)
        for ((v, i) <- row.zipWithIndex) c.io.abWriteData(i).poke(toU(v, cu.inputWidth))
        c.clock.step()
      }
      c.io.abWriteEn.poke(false.B)
      for ((p, set) <- Seq(set0 -> 0, set1 -> 1)) {
        c.io.requantParamWriteEn.poke(true.B)
        c.io.requantParamWriteSet.poke(set.U)
        for (i <- 0 until C) {
          c.io.requantParamWriteData.bias(i).poke(p.bias(i).S(cu.outputWidth.W))
          c.io.requantParamWriteData.mult(i).poke(p.mult(i).U)
          c.io.requantParamWriteData.shift(i).poke(p.shift(i).U)
        }
        c.io.requantParamWriteData.lo.poke(p.lo.S)
        c.io.requantParamWriteData.hi.poke(p.hi.S)
        c.clock.step()
      }
      c.io.requantParamWriteEn.poke(false.B)
      // garbage on the host data port during the run: layer 2 must only
      // ever see the Requantizer's values
      for (i <- 0 until R) c.io.abWriteData(i).poke(toU(99, cu.inputWidth))
      c.clock.step(2)
      CycleProgram.load(new MicroProgramLoader(c.clock, c.io.progWriteEn, c.io.progWriteAddr, c.io.progWriteData), words)

      c.io.start.poke(true.B)
      c.clock.step()
      c.io.start.poke(false.B)
      var n = 0
      while (c.io.running.peek().litToBoolean) { c.clock.step(); n += 1; assert(n <= total) }
      assert(n == total)
      c.clock.step(accCfg.commitLatency)

      c.io.accReadParamSet.poke(0.U)
      for ((exp, s) <- y2.zipWithIndex) {
        c.io.accReadAddr.poke((N + s).U)
        c.io.accReadEn.poke(true.B)
        c.clock.step(accCfg.readLatency)
        c.io.accumOut.valid.expect(true.B)
        for ((v, i) <- exp.zipWithIndex) c.io.accumOut.bits(i).expect(toU(v, cu.inputWidth))
      }
      c.io.accReadEn.poke(false.B)
    }
  }
}
