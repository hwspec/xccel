package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

// Direct probe of LaneSwitch.connect, independent of any buffer -- lets
// this test reach inPerOut > 1 (the N:1 "combine" shape), which no
// WeightBuffer/ActivationBuffer/Accumulator wiring ever exercises (all
// three use inPerOut = 1).
class LaneSwitchProbe(cfg: LaneSwitchConfig) extends Module {
  val io = IO(new Bundle {
    val in = Input(Vec(cfg.numIn, UInt(cfg.laneWidth.W)))
    val rotate = Input(UInt(cfg.rotateWidth.W))
    val fanout = Input(UInt(cfg.fanoutWidth.W))
    val out = Output(Vec(cfg.numOut, UInt(cfg.outWidth.W)))
  })
  io.out := LaneSwitch.connect(cfg, io.in, io.rotate, io.fanout)
}

class LaneSwitchTest extends AnyFlatSpec with ChiselScalatestTester {

  "LaneSwitchConfig" should "reject numIn smaller than numOut*inPerOut" in {
    intercept[IllegalArgumentException] {
      LaneSwitchConfig(numIn = 3, numOut = 4, inPerOut = 1, laneWidth = 8)
    }
  }

  it should "compute outWidth/rotateWidth/fanoutWidth correctly" in {
    val cfg = LaneSwitchConfig(numIn = 8, numOut = 4, inPerOut = 2, laneWidth = 4)
    assert(cfg.outWidth == 8) // inPerOut(2) * laneWidth(4)
    assert(cfg.rotateWidth == 3) // log2Ceil(numIn=8)
    assert(cfg.fanoutWidth == 2) // log2Ceil(log2Ceil(numOut=4)=2 + 1) = log2Ceil(3) = 2
  }

  it should "narrow fanout to maxFanout and reject one above log2Ceil(numOut)" in {
    assert(LaneSwitchConfig(numIn = 8, numOut = 8, inPerOut = 1, laneWidth = 8, Some(0)).fanoutWidth == 0)
    assert(LaneSwitchConfig(numIn = 8, numOut = 8, inPerOut = 1, laneWidth = 8, Some(2)).fanoutWidth == 2)
    intercept[IllegalArgumentException] {
      LaneSwitchConfig(numIn = 8, numOut = 4, inPerOut = 1, laneWidth = 8, Some(3))
    }
  }

  "LaneSwitch.connect" should "select a numOut-wide unicast window starting at rotate" in {
    val cfg = LaneSwitchConfig(numIn = 8, numOut = 4, inPerOut = 1, laneWidth = 8)
    test(new LaneSwitchProbe(cfg)) { c =>
      val lanes = Seq(10, 11, 12, 13, 14, 15, 16, 17)
      for ((v, i) <- lanes.zipWithIndex) c.io.in(i).poke(v.U)
      c.io.fanout.poke(0.U) // unicast

      for (shamt <- 0 until 8) {
        c.io.rotate.poke(shamt.U)
        for (j <- 0 until cfg.numOut) {
          val expected = lanes((j + shamt) % lanes.length)
          c.io.out(j).expect(expected.U)
        }
      }
    }
  }

  it should "broadcast one input lane to every output lane when fanout = log2(numOut)" in {
    val cfg = LaneSwitchConfig(numIn = 8, numOut = 4, inPerOut = 1, laneWidth = 8)
    test(new LaneSwitchProbe(cfg)) { c =>
      val lanes = Seq(10, 11, 12, 13, 14, 15, 16, 17)
      for ((v, i) <- lanes.zipWithIndex) c.io.in(i).poke(v.U)
      c.io.fanout.poke(2.U) // log2(4) = broadcast

      for (shamt <- 0 until 8) {
        c.io.rotate.poke(shamt.U)
        val expected = lanes(shamt)
        for (j <- 0 until cfg.numOut) c.io.out(j).expect(expected.U)
      }
    }
  }

  it should "multicast in groups of 2 when fanout = 1 (between unicast and broadcast)" in {
    val cfg = LaneSwitchConfig(numIn = 8, numOut = 4, inPerOut = 1, laneWidth = 8)
    test(new LaneSwitchProbe(cfg)) { c =>
      val lanes = Seq(10, 11, 12, 13, 14, 15, 16, 17)
      for ((v, i) <- lanes.zipWithIndex) c.io.in(i).poke(v.U)
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(1.U) // groupSize = 2

      // output lanes 0,1 share input lane 0; 2,3 share input lane 1
      val expected = Seq(lanes(0), lanes(0), lanes(1), lanes(1))
      for (j <- 0 until cfg.numOut) c.io.out(j).expect(expected(j).U)
    }
  }

  it should "combine inPerOut adjacent input lanes into one wider output lane (the N:1 shape)" in {
    // numIn == numOut*inPerOut exactly (no spare room) -- here numOut <
    // numIn, which no buffer builds since they all use inPerOut = 1
    val cfg = LaneSwitchConfig(numIn = 4, numOut = 2, inPerOut = 2, laneWidth = 4)
    test(new LaneSwitchProbe(cfg)) { c =>
      val lanes = Seq(1, 2, 3, 4) // 4-bit each: 0001, 0010, 0011, 0100
      for ((v, i) <- lanes.zipWithIndex) c.io.in(i).poke(v.U)
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(0.U) // unicast: output lane j gets its own distinct inPerOut-wide group

      // input lane 0 is the LSB half of its output lane (same "index 0 = LSB" convention
      // as Sram.fromChunks / the buffers' own wdata packing)
      c.io.out(0).expect("b0010_0001".U) // Cat(in1=2, in0=1)
      c.io.out(1).expect("b0100_0011".U) // Cat(in3=4, in2=3)
    }
  }

  "LaneSwitch" should "rotate, unicast and broadcast with a numOut that is not a power of two (a wide weight beat)" in {
    val cfg = LaneSwitchConfig(numIn = 6, numOut = 6, inPerOut = 1, laneWidth = 8)
    test(new LaneSwitchProbe(cfg)) { c =>
      for (i <- 0 until 6) c.io.in(i).poke((10 + i).U)
      // unicast, rotated by 2: output lane j gets input lane (j + 2) mod 6
      c.io.rotate.poke(2.U)
      c.io.fanout.poke(0.U)
      for (j <- 0 until 6) c.io.out(j).expect((10 + (j + 2) % 6).U)
      // groups of 4 output lanes: the last group (lanes 4, 5) is partial
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(2.U)
      for (j <- 0 until 6) c.io.out(j).expect((10 + j / 4).U)
      // fanout = log2Ceil(6) = 3: broadcast
      c.io.fanout.poke(3.U)
      for (j <- 0 until 6) c.io.out(j).expect(10.U)
    }
  }

  it should "build unicast only with maxFanout = 0 (no fanout input)" in {
    val cfg = LaneSwitchConfig(numIn = 8, numOut = 4, inPerOut = 1, laneWidth = 8, maxFanout = Some(0))
    test(new LaneSwitchProbe(cfg)) { c =>
      for (i <- 0 until 8) c.io.in(i).poke((10 + i).U)
      for (shamt <- 0 until 8) {
        c.io.rotate.poke(shamt.U)
        for (j <- 0 until cfg.numOut) c.io.out(j).expect((10 + (j + shamt) % 8).U)
      }
    }
  }

  it should "stop at maxFanout: a larger fanout multicasts like the largest one" in {
    // numOut = 8 could broadcast at fanout 3; capped at 2, fanout 3
    // (still representable in 2 bits) gives groups of 4 like fanout 2
    val cfg = LaneSwitchConfig(numIn = 8, numOut = 8, inPerOut = 1, laneWidth = 8, maxFanout = Some(2))
    test(new LaneSwitchProbe(cfg)) { c =>
      for (i <- 0 until 8) c.io.in(i).poke((10 + i).U)
      c.io.rotate.poke(0.U)
      c.io.fanout.poke(1.U)
      for (j <- 0 until 8) c.io.out(j).expect((10 + j / 2).U)
      for (g <- Seq(2, 3)) {
        c.io.fanout.poke(g.U)
        for (j <- 0 until 8) c.io.out(j).expect((10 + j / 4).U)
      }
    }
  }
}
