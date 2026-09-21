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
      VecInit((0 until n).map(i => Mux(doShift, cur((i + step) % n), cur(i))))
    }
  }
}
