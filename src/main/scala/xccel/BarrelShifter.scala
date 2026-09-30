package xccel

import chisel3._
import chisel3.util._

object BarrelShifter {
  def rotateLeft[T <: Data](in: Vec[T], shamt: UInt): Vec[T] = {
    val n = in.length
    val stages = log2Ceil(n)
    (0 until stages).foldLeft(in) { (cur, k) =>
      val step = 1 << k // 2^k
      val doShift = shamt(k)
      // each stage is a kept wire: otherwise firtool may inline the stages into
      // every output, a 2^stages-leaf mux tree per output instead of n muxes per stage
      val stage = Wire(Vec(n, chiselTypeOf(in.head)))
      stage := VecInit((0 until n).map(i => Mux(doShift, cur((i + step) % n), cur(i))))
      dontTouch(stage)
    }
  }
}
