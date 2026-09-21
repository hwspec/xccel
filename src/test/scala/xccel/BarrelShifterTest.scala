package xccel

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class BarrelShifterProbe(n: Int, w: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(Vec(n, UInt(w.W)))
    val shamt = Input(UInt(log2Ceil(n).W))
    val out = Output(Vec(n, UInt(w.W)))
  })
  io.out := BarrelShifter.rotateLeft(io.in, io.shamt)
}

class BarrelShifterTest extends AnyFlatSpec with ChiselScalatestTester {
  "BarrelShifter.rotateLeft" should "rotate out(i) = in((i+shamt) mod n) for a non-power-of-two n" in {
    val n = 6
    test(new BarrelShifterProbe(n, 8)) { c =>
      val values = Seq(10, 20, 30, 40, 50, 60)
      for ((v, i) <- values.zipWithIndex) c.io.in(i).poke(v.U)

      for (shamt <- 0 until n) {
        c.io.shamt.poke(shamt.U)
        for (i <- 0 until n) {
          c.io.out(i).expect(values((i + shamt) % n).U)
        }
      }
    }
  }
}
