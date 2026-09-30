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
  // groups are 2^groupShift consecutive ports (a plain shift, not a
  // divider); with a numPorts that is not a power of two the last group is
  // partial, and groupShift = log2Ceil(numPorts) is still broadcast
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
    // port j takes group j >> groupShift. groupShift only ranges over
    // 0..log2Ceil(numPorts) (any larger value is broadcast, like the last), so
    // each port selects among those few static groups instead of indexing the
    // whole grain pool (numPorts x log numPorts muxes, not numPorts x grains)
    val maxShift = log2Ceil(cfg.numPorts)
    VecInit((0 until cfg.numPorts).map { j =>
      def port(g: Int): UInt = {
        val base = (j >> g) * cfg.grainsPerPort
        Cat((0 until cfg.grainsPerPort).reverse.map(i => rotated(base + i)))
      }
      MuxLookup(groupShift, port(maxShift))((0 until maxShift).map(g => g.U -> port(g)))
    })
  }
}
