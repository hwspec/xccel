package xccel

import chisel3._
import chisel3.util._

// n x n register block on the Requantizer -> WeightBuffer path, used to turn
// an activation matrix into weights when the product needs its transpose
// (e.g. theta . phi^T: phi arrives as rows of one pixel's n channels, a
// weight row must hold one channel of n pixels).
//
// capture writes `in` into row `row`; `out` is column `column`, i.e.
// out(j) = row j's element `column`. Both are driven by the program;
// there is one block (no double buffering), so a block's n captures must
// all precede its n column reads.
class Transposer(n: Int, width: Int) extends Module {
  require(n > 0, "n must be positive")
  require(width > 0, "width must be positive")
  val indexWidth: Int = math.max(1, log2Ceil(n))

  val io = IO(new Bundle {
    val capture = Input(Bool())
    val row = Input(UInt(indexWidth.W))
    val in = Input(Vec(n, UInt(width.W)))
    val column = Input(UInt(indexWidth.W))
    val out = Output(Vec(n, UInt(width.W)))
  })

  private val block = Reg(Vec(n, Vec(n, UInt(width.W))))
  when(io.capture) {
    block(io.row) := io.in
  }
  io.out := VecInit(Seq.tabulate(n)(j => block(j)(io.column)))
}
