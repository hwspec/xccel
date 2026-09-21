package xccel

import chisel3._
import chisel3.util._

// Grains = supply-side (e.g., SRAM banks), Ports = demand-side (e.g., systolic mesh)
//   - grainsPerPort (structural, compile-time): portWidth =
//     grainsPerPort * grainWidth. 1 = the 1:1 / 1:N shapes (a port is
//     exactly one grain); >1 = the N:1 shape (several grains concatenate
//     into one wider port).
//   - shiftAmt (dynamic, decoder-driven): rotates the grain pool before
//     assignment, so the active window can start anywhere -- e.g.
//     selecting which resident tile is active.
//   - groupShift (dynamic, decoder-driven): log2 of how many consecutive
//     ports share one grain-group. 0 = unicast (every port gets a
//     distinct group), log2(numPorts) = broadcast (every port shares the
//     same group), in between = multicast.
case class PortAdapterConfig(totalGrains: Int, grainWidth: Int, numPorts: Int, grainsPerPort: Int) {
  require(totalGrains > 0, "totalGrains must be positive")
  require(grainWidth > 0, "grainWidth must be positive")
  require(numPorts > 0, "numPorts must be positive")
  require(grainsPerPort > 0, "grainsPerPort must be positive")
  require(
    (numPorts & (numPorts - 1)) == 0,
    s"numPorts ($numPorts) must be a power of two so groupShift can address every group size" +
      " (unicast..broadcast) with a plain shift, not a divider"
  )
  require(
    totalGrains >= numPorts * grainsPerPort,
    s"totalGrains ($totalGrains) must cover unicast mode's worst case," +
      s" numPorts*grainsPerPort (${numPorts * grainsPerPort})"
  )

  val portWidth: Int = grainsPerPort * grainWidth
  val shiftWidth: Int = log2Ceil(totalGrains)
  // groupShift ranges 0..log2Ceil(numPorts) inclusive (unicast..broadcast)
  val groupShiftWidth: Int = log2Ceil(log2Ceil(numPorts) + 1)
}

object PortAdapter {
  def connect(cfg: PortAdapterConfig, grains: Vec[UInt], shiftAmt: UInt, groupShift: UInt): Vec[UInt] = {
    val rotated = BarrelShifter.rotateLeft(grains, shiftAmt)
    // pin the index width explicitly rather than let it drift through the
    // shift/multiply chain -- must cover indices up to numPorts*grainsPerPort-1
    val idxWidth = math.max(1, log2Ceil(cfg.numPorts * cfg.grainsPerPort))
    VecInit((0 until cfg.numPorts).map { j =>
      val group = Wire(UInt(idxWidth.W))
      group := j.U >> groupShift
      val base = Wire(UInt(idxWidth.W))
      base := group * cfg.grainsPerPort.U
      val slices = (0 until cfg.grainsPerPort).reverse.map { i =>
        val idx = Wire(UInt(idxWidth.W))
        idx := base + i.U
        rotated(idx)
      }
      Cat(slices)
    })
  }
}
