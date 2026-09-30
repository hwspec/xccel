package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

// Direct probe of PortAdapter.connect, independent of any buffer -- lets
// this test reach grainsPerPort > 1 (the N:1 "combine" shape), which no
// WeightBuffer/ActivationBuffer/Accumulator wiring ever exercises (all
// three hardcode grainsPerPort = 1, see their own comments for why).
class PortAdapterProbe(cfg: PortAdapterConfig) extends Module {
  val io = IO(new Bundle {
    val grains = Input(Vec(cfg.totalGrains, UInt(cfg.grainWidth.W)))
    val shiftAmt = Input(UInt(cfg.shiftWidth.W))
    val groupShift = Input(UInt(cfg.groupShiftWidth.W))
    val ports = Output(Vec(cfg.numPorts, UInt(cfg.portWidth.W)))
  })
  io.ports := PortAdapter.connect(cfg, io.grains, io.shiftAmt, io.groupShift)
}

class PortAdapterTest extends AnyFlatSpec with ChiselScalatestTester {

  "PortAdapterConfig" should "reject totalGrains smaller than numPorts*grainsPerPort" in {
    intercept[IllegalArgumentException] {
      PortAdapterConfig(totalGrains = 3, grainWidth = 8, numPorts = 4, grainsPerPort = 1)
    }
  }

  it should "compute portWidth/shiftWidth/groupShiftWidth correctly" in {
    val cfg = PortAdapterConfig(totalGrains = 8, grainWidth = 4, numPorts = 4, grainsPerPort = 2)
    assert(cfg.portWidth == 8) // grainsPerPort(2) * grainWidth(4)
    assert(cfg.shiftWidth == 3) // log2Ceil(totalGrains=8)
    assert(cfg.groupShiftWidth == 2) // log2Ceil(log2Ceil(numPorts=4)=2 + 1) = log2Ceil(3) = 2
  }

  "PortAdapter.connect" should "select a numPorts-wide unicast window starting at shiftAmt" in {
    val cfg = PortAdapterConfig(totalGrains = 8, grainWidth = 8, numPorts = 4, grainsPerPort = 1)
    test(new PortAdapterProbe(cfg)) { c =>
      val grains = Seq(10, 11, 12, 13, 14, 15, 16, 17)
      for ((v, i) <- grains.zipWithIndex) c.io.grains(i).poke(v.U)
      c.io.groupShift.poke(0.U) // unicast

      for (shamt <- 0 until 8) {
        c.io.shiftAmt.poke(shamt.U)
        for (j <- 0 until cfg.numPorts) {
          val expected = grains((j + shamt) % grains.length)
          c.io.ports(j).expect(expected.U)
        }
      }
    }
  }

  it should "broadcast a single grain to every port when groupShift = log2(numPorts)" in {
    val cfg = PortAdapterConfig(totalGrains = 8, grainWidth = 8, numPorts = 4, grainsPerPort = 1)
    test(new PortAdapterProbe(cfg)) { c =>
      val grains = Seq(10, 11, 12, 13, 14, 15, 16, 17)
      for ((v, i) <- grains.zipWithIndex) c.io.grains(i).poke(v.U)
      c.io.groupShift.poke(2.U) // log2(4) = broadcast

      for (shamt <- 0 until 8) {
        c.io.shiftAmt.poke(shamt.U)
        val expected = grains(shamt)
        for (j <- 0 until cfg.numPorts) c.io.ports(j).expect(expected.U)
      }
    }
  }

  it should "multicast in groups of 2 when groupShift = 1 (between unicast and broadcast)" in {
    val cfg = PortAdapterConfig(totalGrains = 8, grainWidth = 8, numPorts = 4, grainsPerPort = 1)
    test(new PortAdapterProbe(cfg)) { c =>
      val grains = Seq(10, 11, 12, 13, 14, 15, 16, 17)
      for ((v, i) <- grains.zipWithIndex) c.io.grains(i).poke(v.U)
      c.io.shiftAmt.poke(0.U)
      c.io.groupShift.poke(1.U) // groupSize = 2

      // ports 0,1 share grain 0; ports 2,3 share grain 1
      val expected = Seq(grains(0), grains(0), grains(1), grains(1))
      for (j <- 0 until cfg.numPorts) c.io.ports(j).expect(expected(j).U)
    }
  }

  it should "combine grainsPerPort adjacent grains into one wider port (the N:1 shape)" in {
    // totalGrains == numPorts*grainsPerPort exactly (no spare room), same
    // as Accumulator's own shape -- but here numPorts < totalGrains,
    // which no buffer ever builds since they all fix grainsPerPort=1
    val cfg = PortAdapterConfig(totalGrains = 4, grainWidth = 4, numPorts = 2, grainsPerPort = 2)
    test(new PortAdapterProbe(cfg)) { c =>
      val grains = Seq(1, 2, 3, 4) // 4-bit each: 0001, 0010, 0011, 0100
      for ((v, i) <- grains.zipWithIndex) c.io.grains(i).poke(v.U)
      c.io.shiftAmt.poke(0.U)
      c.io.groupShift.poke(0.U) // unicast: port j gets its own distinct grainsPerPort-wide group

      // grain 0 is the LSB half of its port (same "index 0 = LSB" convention
      // as Sram.fromChunks / the buffers' own wdata packing)
      c.io.ports(0).expect("b0010_0001".U) // Cat(grain1=2, grain0=1)
      c.io.ports(1).expect("b0100_0011".U) // Cat(grain3=4, grain2=3)
    }
  }

  "PortAdapter" should "rotate, unicast and broadcast with a numPorts that is not a power of two (a wide weight beat)" in {
    val cfg = PortAdapterConfig(totalGrains = 6, grainWidth = 8, numPorts = 6, grainsPerPort = 1)
    test(new PortAdapterProbe(cfg)) { c =>
      for (i <- 0 until 6) c.io.grains(i).poke((10 + i).U)
      // unicast, rotated by 2: port j gets grain (j + 2) mod 6
      c.io.shiftAmt.poke(2.U)
      c.io.groupShift.poke(0.U)
      for (j <- 0 until 6) c.io.ports(j).expect((10 + (j + 2) % 6).U)
      // groups of 4 ports: the last group (ports 4, 5) is partial
      c.io.shiftAmt.poke(0.U)
      c.io.groupShift.poke(2.U)
      for (j <- 0 until 6) c.io.ports(j).expect((10 + j / 4).U)
      // groupShift = log2Ceil(6) = 3: broadcast
      c.io.groupShift.poke(3.U)
      for (j <- 0 until 6) c.io.ports(j).expect(10.U)
    }
  }
}
