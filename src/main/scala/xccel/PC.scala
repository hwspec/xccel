package xccel

import chisel3._
import chisel3.util._


class PC(width: Int = 32) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val stop = Input(Bool())
    val running = Output(Bool())
    val pc = Output(UInt(width.W))
  })

  val pcReg = RegInit(0.U(width.W))
  val runningReg = RegInit(false.B)

  when (io.start) {
    pcReg := 0.U
    runningReg := true.B
  } .elsewhen (runningReg) {
    when (io.stop) {
      runningReg := false.B
    } .otherwise {
      pcReg := pcReg + 1.U
    }
  }

  io.pc := pcReg
  io.running := runningReg
}
