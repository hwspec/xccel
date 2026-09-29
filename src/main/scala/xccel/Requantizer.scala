package xccel

import chisel3._
import chisel3.util._

// Narrows wide accumulator values (e.g. INT32 partial sums) to the next
// stage's width (e.g. INT8 activations) with a per-lane bias and
// fixed-point scaling:
//
//   out = clamp(roundShift((in + bias) * mult, shift), lo, hi)
//
// bias is per lane, in the accumulator's scale and width (like an INT32
// bias added to INT32 partial sums).
// mult/shift are per lane (output channel) and encode a real-valued scale
// of about mult / 2^shift. roundShift rounds half away from zero (like
// TFLite's RoundingDivideByPOT). lo/hi saturate; lo = 0 gives ReLU.
//
// numParamSets independent parameter sets (e.g. one per layer and C_out
// tile) are loaded by the host and selected per beat by `paramSet`, which
// must be aligned with `in`. After reset every set is the identity (bias 0,
// mult 1, shift 0, full outWidth range).
//
// Purely combinational (no latency), like the placeholder it replaces.
case class RequantizerConfig(
  numLanes: Int,
  inWidth: Int,
  outWidth: Int,
  multWidth: Int = 16,
  numParamSets: Int = 1,
) {
  require(numLanes > 0, "numLanes must be positive")
  require(inWidth > 0, "inWidth must be positive")
  require(outWidth > 0, "outWidth must be positive")
  require(
    outWidth <= inWidth,
    s"outWidth ($outWidth) must not exceed inWidth ($inWidth) -- this narrows, never widens"
  )
  require(multWidth > 0, "multWidth must be positive")
  require(numParamSets > 0, "numParamSets must be positive")

  // in + bias is inWidth + 1 bits; times the unsigned mult, the product is
  // inWidth + 1 + multWidth + 1 bits
  val productWidth: Int = inWidth + multWidth + 2
  val shiftWidth: Int = log2Ceil(productWidth)
  val setWidth: Int = math.max(1, log2Ceil(numParamSets))
  val outMin: BigInt = -(BigInt(1) << (outWidth - 1))
  val outMax: BigInt = (BigInt(1) << (outWidth - 1)) - 1
}

class RequantParams(cfg: RequantizerConfig) extends Bundle {
  val bias = Vec(cfg.numLanes, SInt(cfg.inWidth.W))
  val mult = Vec(cfg.numLanes, UInt(cfg.multWidth.W))
  val shift = Vec(cfg.numLanes, UInt(cfg.shiftWidth.W))
  val lo = SInt(cfg.outWidth.W)
  val hi = SInt(cfg.outWidth.W)
}

// `in` matches a buffer's raw UInt output (e.g. Accumulator.accOut);
// `out` matches a buffer's raw UInt write input (e.g. ActivationBuffer.
// wdata)
class RequantizerIO(cfg: RequantizerConfig) extends Bundle {
  val in = Flipped(Valid(Vec(cfg.numLanes, UInt(cfg.inWidth.W))))
  val paramSet = Input(UInt(cfg.setWidth.W))
  val out = Valid(Vec(cfg.numLanes, UInt(cfg.outWidth.W)))

  // parameter load, host-driven
  val paramWriteEn = Input(Bool())
  val paramWriteSet = Input(UInt(cfg.setWidth.W))
  val paramWriteData = Input(new RequantParams(cfg))
}

class Requantizer(cfg: RequantizerConfig) extends Module {
  val io = IO(new RequantizerIO(cfg))

  private val identity = Wire(new RequantParams(cfg))
  identity.bias := VecInit(Seq.fill(cfg.numLanes)(0.S(cfg.inWidth.W)))
  identity.mult := VecInit(Seq.fill(cfg.numLanes)(1.U(cfg.multWidth.W)))
  identity.shift := VecInit(Seq.fill(cfg.numLanes)(0.U(cfg.shiftWidth.W)))
  identity.lo := cfg.outMin.S(cfg.outWidth.W)
  identity.hi := cfg.outMax.S(cfg.outWidth.W)

  private val params = RegInit(VecInit(Seq.fill(cfg.numParamSets)(identity)))
  when(io.paramWriteEn) {
    params(io.paramWriteSet) := io.paramWriteData
  }
  private val p = params(io.paramSet)

  io.out.bits := VecInit(io.in.bits.zipWithIndex.map { case (lane, i) =>
    val product = (lane.asSInt +& p.bias(i)) * p.mult(i).zext
    val shift = p.shift(i)
    // round half away from zero: add 2^(shift-1) before the arithmetic
    // (flooring) shift, one less for negative values
    val half = (1.S(2.W) << (shift - 1.U)).asSInt
    val nudge = Mux(shift === 0.U, 0.S, Mux(product < 0.S, half - 1.S, half))
    val rounded = (product +& nudge) >> shift
    val clamped = Mux(rounded < p.lo, p.lo, Mux(rounded > p.hi, p.hi, rounded))
    clamped.asUInt(cfg.outWidth - 1, 0)
  })
  io.out.valid := io.in.valid
}
