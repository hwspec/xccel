package xccel

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class PCTest extends AnyFlatSpec with ChiselScalatestTester {
  it should "initialize to pc=0 and running=false" in {
    test(new PC()) { c =>
      c.io.start.poke(false.B)
      c.io.stop.poke(false.B)

      c.io.pc.expect(0.U)
      c.io.running.expect(false.B)
    }
  }

  it should "start and increment pc while running" in {
    test(new PC()) { c =>
      c.io.start.poke(false.B)
      c.io.stop.poke(false.B)

      c.io.pc.expect(0.U)
      c.io.running.expect(false.B)

      c.io.start.poke(true.B)
      c.clock.step(1)

      c.io.start.poke(false.B)
      c.io.pc.expect(0.U)
      c.io.running.expect(true.B)

      c.clock.step(1)
      c.io.pc.expect(1.U)
      c.io.running.expect(true.B)

      c.clock.step(1)
      c.io.pc.expect(2.U)
      c.io.running.expect(true.B)

      c.clock.step(1)
      c.io.pc.expect(3.U)
      c.io.running.expect(true.B)
    }
  }

  it should "stop when stop is asserted" in {
    test(new PC()) { c =>
      c.io.start.poke(false.B)
      c.io.stop.poke(false.B)

      c.io.start.poke(true.B)
      c.clock.step(1)
      c.io.start.poke(false.B)

      c.clock.step(2)
      c.io.pc.expect(2.U)
      c.io.running.expect(true.B)

      c.io.stop.poke(true.B)
      c.clock.step(1)

      c.io.stop.poke(false.B)
      c.io.running.expect(false.B)
      c.io.pc.expect(2.U)

      c.clock.step(2)
      c.io.pc.expect(2.U)
      c.io.running.expect(false.B)
    }
  }

  it should "restart from zero when start is asserted again" in {
    test(new PC()) { c =>
      c.io.start.poke(false.B)
      c.io.stop.poke(false.B)

      c.io.start.poke(true.B)
      c.clock.step(1)
      c.io.start.poke(false.B)

      c.clock.step(3)
      c.io.pc.expect(3.U)
      c.io.running.expect(true.B)

      c.io.stop.poke(true.B)
      c.clock.step(1)
      c.io.stop.poke(false.B)
      c.io.running.expect(false.B)

      c.io.start.poke(true.B)
      c.clock.step(1)
      c.io.start.poke(false.B)

      c.io.pc.expect(0.U)
      c.io.running.expect(true.B)
    }
  }
}
