package xccel

import chisel3._
import chisel3.util._

// Routes the numIn input lanes of one read (e.g. a buffer row's words, which
// can span several SRAM banks) to numOut output lanes (e.g. a compute unit's
// operand or weight beat, or another unit's input), all laneWidth bits wide:
//   - inPerOut (structural, compile-time): an output lane concatenates
//     inPerOut adjacent input lanes (outWidth = inPerOut * laneWidth).
//     1 = the 1:1 / 1:N shapes; > 1 = the N:1 shape.
//   - rotate (dynamic, decoder-driven): rotates the input lanes before
//     assignment, so the active window can start anywhere -- e.g.
//     selecting which resident tile is active.
//   - fanout (dynamic, decoder-driven): log2 of how many consecutive
//     output lanes share one input group. 0 = unicast (every output lane
//     gets a distinct group), log2Ceil(numOut) = broadcast (every output
//     lane shares the same group), in between = multicast.
//   - maxFanout (structural, compile-time): the largest fanout
//     built; None = log2Ceil(numOut) (up to broadcast), Some(0) = unicast
//     only (a zero-width fanout, no group muxes).
case class LaneSwitchConfig(
  numIn: Int,
  numOut: Int,
  inPerOut: Int,
  laneWidth: Int,
  maxFanout: Option[Int] = None,
) {
  require(numIn > 0, "numIn must be positive")
  require(numOut > 0, "numOut must be positive")
  require(inPerOut > 0, "inPerOut must be positive")
  require(laneWidth > 0, "laneWidth must be positive")
  // groups are 2^fanout consecutive output lanes (a plain shift, not a
  // divider); with a numOut that is not a power of two the last group is
  // partial, and fanout = log2Ceil(numOut) is still broadcast
  require(
    numIn >= numOut * inPerOut,
    s"numIn ($numIn) must cover unicast mode's worst case, numOut*inPerOut (${numOut * inPerOut})"
  )

  val outWidth: Int = inPerOut * laneWidth
  val rotateWidth: Int = log2Ceil(numIn)
  // the largest fanout built (maxFanout resolved)
  val fanoutLimit: Int = maxFanout.getOrElse(log2Ceil(numOut))
  require(
    fanoutLimit >= 0 && fanoutLimit <= log2Ceil(numOut),
    s"maxFanout ($fanoutLimit) must be in 0..log2Ceil(numOut) (${log2Ceil(numOut)})"
  )
  // fanout ranges 0..fanoutLimit inclusive (unicast..broadcast by default)
  val fanoutWidth: Int = log2Ceil(fanoutLimit + 1)
}

object LaneSwitch {
  def connect(cfg: LaneSwitchConfig, in: Vec[UInt], rotate: UInt, fanout: UInt): Vec[UInt] = {
    val rotated = BarrelShifter.rotateLeft(in, rotate)
    // output lane j takes group j >> fanout. fanout only ranges over
    // 0..maxFanout (any larger value acts as maxFanout), so each output lane
    // selects among those few static groups instead of indexing every input
    // lane (numOut x maxFanout muxes, not numOut x numIn)
    val maxFanout = cfg.fanoutLimit
    VecInit((0 until cfg.numOut).map { j =>
      def group(g: Int): UInt = {
        val base = (j >> g) * cfg.inPerOut
        Cat((0 until cfg.inPerOut).reverse.map(i => rotated(base + i)))
      }
      if (maxFanout == 0) group(0)
      else MuxLookup(fanout, group(maxFanout))((0 until maxFanout).map(g => g.U -> group(g)))
    })
  }
}
