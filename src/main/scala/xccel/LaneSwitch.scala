package xccel

import chisel3._
import chisel3.util._

// Routes the numIn input lanes of one read (e.g. a buffer row's words, which
// can span several SRAM banks) to numOut output lanes (e.g. a compute unit's
// operand or weight beat, or another unit's input), all laneWidth bits wide:
//   - inPerOut (structural, compile-time): an output lane concatenates
//     inPerOut adjacent input lanes (outWidth = inPerOut * laneWidth).
//     1 = the 1:1 / 1:N shapes; > 1 = the N:1 shape.
//   - shiftAmt (dynamic, decoder-driven): rotates the input lanes before
//     assignment, so the active window can start anywhere -- e.g.
//     selecting which resident tile is active.
//   - groupShift (dynamic, decoder-driven): log2 of how many consecutive
//     output lanes share one input group. 0 = unicast (every output lane
//     gets a distinct group), log2Ceil(numOut) = broadcast (every output
//     lane shares the same group), in between = multicast.
case class LaneSwitchConfig(numIn: Int, numOut: Int, inPerOut: Int, laneWidth: Int) {
  require(numIn > 0, "numIn must be positive")
  require(numOut > 0, "numOut must be positive")
  require(inPerOut > 0, "inPerOut must be positive")
  require(laneWidth > 0, "laneWidth must be positive")
  // groups are 2^groupShift consecutive output lanes (a plain shift, not a
  // divider); with a numOut that is not a power of two the last group is
  // partial, and groupShift = log2Ceil(numOut) is still broadcast
  require(
    numIn >= numOut * inPerOut,
    s"numIn ($numIn) must cover unicast mode's worst case, numOut*inPerOut (${numOut * inPerOut})"
  )

  val outWidth: Int = inPerOut * laneWidth
  val shiftWidth: Int = log2Ceil(numIn)
  // groupShift ranges 0..log2Ceil(numOut) inclusive (unicast..broadcast)
  val groupShiftWidth: Int = log2Ceil(log2Ceil(numOut) + 1)
}

object LaneSwitch {
  def connect(cfg: LaneSwitchConfig, in: Vec[UInt], shiftAmt: UInt, groupShift: UInt): Vec[UInt] = {
    val rotated = BarrelShifter.rotateLeft(in, shiftAmt)
    // output lane j takes group j >> groupShift. groupShift only ranges over
    // 0..log2Ceil(numOut) (any larger value is broadcast, like the last), so
    // each output lane selects among those few static groups instead of
    // indexing every input lane (numOut x log numOut muxes, not numOut x numIn)
    val maxShift = log2Ceil(cfg.numOut)
    VecInit((0 until cfg.numOut).map { j =>
      def group(g: Int): UInt = {
        val base = (j >> g) * cfg.inPerOut
        Cat((0 until cfg.inPerOut).reverse.map(i => rotated(base + i)))
      }
      MuxLookup(groupShift, group(maxShift))((0 until maxShift).map(g => g.U -> group(g)))
    })
  }
}
