package xccel

import chisel3._
import chisel3.util._

// Narrows a wide accumulator value (e.g. INT32 partial sums) down to a
// narrower port width (e.g. INT8 activations for the next stage) --
// sits between Accumulator.accOut and the next stage's
// ActivationBuffer.wdata
case class RequantizerConfig(numLanes: Int, inWidth: Int, outWidth: Int) {
  require(numLanes > 0, "numLanes must be positive")
  require(inWidth > 0, "inWidth must be positive")
  require(outWidth > 0, "outWidth must be positive")
  require(
    outWidth <= inWidth,
    s"outWidth ($outWidth) must not exceed inWidth ($inWidth) -- this narrows, never widens"
  )

  val shiftAmt: Int = inWidth - outWidth
}

// `in` matches a buffer's raw UInt output (e.g. Accumulator.accOut);
// `out` matches a buffer's raw UInt write input (e.g. ActivationBuffer.
// wdata)
class RequantizerIO(cfg: RequantizerConfig) extends Bundle {
  val in = Flipped(Valid(Vec(cfg.numLanes, UInt(cfg.inWidth.W))))
  val out = Valid(Vec(cfg.numLanes, UInt(cfg.outWidth.W)))
}

class Requantizer(cfg: RequantizerConfig) extends Module {
  val io = IO(new RequantizerIO(cfg))

  // purely combinational: a static-amount SInt >> is just wiring (keeps
  // the top outWidth bits, sign-extended) -- no registers, no latency
  io.out.bits := VecInit(io.in.bits.map(lane => (lane.asSInt >> cfg.shiftAmt).asUInt))
  io.out.valid := io.in.valid
}
