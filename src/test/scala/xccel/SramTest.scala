package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SramTest extends AnyFlatSpec with ChiselScalatestTester {

  def valueAt(a: Int, width: Int): BigInt = {
    val mask = (BigInt(1) << width) - 1
    (BigInt(a) * 7 + 3) & mask
  }

  "Sram" should "read back what was written, respecting readLatency/writeLatency (1 bank)" in {
    val cfg = SramConfig(
      depth = 32,
      width = 16,
      numBanks = 1,
      maskGranularity = 8,
      readLatency = 2,
      writeLatency = 2,
    )
    test(new Sram(cfg)) { c =>
      val b = c.io.ports(0)
      b.wmask.poke(cfg.fullMask)

      for (a <- 0 until cfg.depth) {
        b.addr.poke(a.U)
        b.en.poke(true.B)
        b.wen.poke(true.B)
        b.wdata.poke(valueAt(a, cfg.width).U)
        c.clock.step()
      }
      b.en.poke(false.B)
      b.wen.poke(false.B)
      c.clock.step(cfg.writeLatency)

      for (a <- 0 until cfg.depth) {
        b.addr.poke(a.U)
        b.en.poke(true.B)
        b.wen.poke(false.B)
        c.clock.step(cfg.readLatency)
        b.rdata.expect(valueAt(a, cfg.width).U)
      }
      b.en.poke(false.B)
    }
  }

  it should "keep each bank independent (same address, different banks, different data)" in {
    val cfg = SramConfig(
      depth = 8,
      width = 8,
      numBanks = 2,
      maskGranularity = 8,
      readLatency = 1,
      writeLatency = 1,
    )
    test(new Sram(cfg)) { c =>
      val b0 = c.io.ports(0)
      val b1 = c.io.ports(1)
      b0.wmask.poke(cfg.fullMask)
      b1.wmask.poke(cfg.fullMask)

      // write every address on both banks simultaneously, but with
      // different data per bank, to confirm they don't share storage
      for (a <- 0 until cfg.depth) {
        b0.addr.poke(a.U)
        b0.en.poke(true.B)
        b0.wen.poke(true.B)
        b0.wdata.poke(valueAt(a, cfg.width).U)

        b1.addr.poke(a.U)
        b1.en.poke(true.B)
        b1.wen.poke(true.B)
        b1.wdata.poke(valueAt(a + 100, cfg.width).U)

        c.clock.step()
      }
      b0.en.poke(false.B)
      b0.wen.poke(false.B)
      b1.en.poke(false.B)
      b1.wen.poke(false.B)

      for (a <- 0 until cfg.depth) {
        b0.addr.poke(a.U)
        b0.en.poke(true.B)
        b0.wen.poke(false.B)
        b1.addr.poke(a.U)
        b1.en.poke(true.B)
        b1.wen.poke(false.B)
        c.clock.step()
        b0.rdata.expect(valueAt(a, cfg.width).U)
        b1.rdata.expect(valueAt(a + 100, cfg.width).U)
      }
      b0.en.poke(false.B)
      b1.en.poke(false.B)
    }
  }

  it should "write only the wmask-enabled chunks of a wide row" in {
    val cfg = SramConfig(
      depth = 4,
      width = 32,
      numBanks = 1,
      maskGranularity = 8, // 4 mask bits per row
      readLatency = 1,
      writeLatency = 1,
    )
    test(new Sram(cfg)) { c =>
      val b = c.io.ports(0)
      def write(a: Int, data: BigInt, mask: Int): Unit = {
        b.addr.poke(a.U); b.en.poke(true.B); b.wen.poke(true.B)
        b.wdata.poke(data.U); b.wmask.poke(mask.U)
        c.clock.step()
      }
      write(1, BigInt("aabbccdd", 16), 0xf)
      write(1, BigInt("11223344", 16), 0x6) // chunks 1 and 2 only
      b.en.poke(true.B); b.wen.poke(false.B)
      c.clock.step()
      b.rdata.expect(BigInt("aa2233dd", 16).U)
      b.en.poke(false.B)
    }
  }

  it should "assert when a read is requested in the cycle a delayed write (writeLatency > 1) reaches the port" in {
    val cfg = SramConfig(depth = 8, width = 8, numBanks = 1, maskGranularity = 8,
      readLatency = 1, writeLatency = 2)
    intercept[chisel3.simulator.Exceptions.AssertionFailed] {
      test(new Sram(cfg)) { c =>
        val b = c.io.ports(0)
        b.wmask.poke(cfg.fullMask)
        // cycle 0: write row 1 (reaches the bank at cycle 1)
        b.addr.poke(1.U); b.en.poke(true.B); b.wen.poke(true.B); b.wdata.poke(5.U)
        c.clock.step()
        // cycle 1: read row 2 while that write takes the port
        b.addr.poke(2.U); b.wen.poke(false.B)
        c.clock.step()
      }
    }
  }

  "Sram (OneR1W)" should "read one address while simultaneously writing a different one" in {
    val cfg = SramConfig(
      depth = 8,
      width = 8,
      numBanks = 1,
      maskGranularity = 8,
      readLatency = 1,
      writeLatency = 1,
      portType = BankPortType.OneR1W,
    )
    test(new Sram(cfg)) { c =>
      val rp = c.io.ports(0)
      val wp = c.io.ports(1)
      wp.wmask.poke(cfg.fullMask)

      for (a <- 0 until cfg.depth) {
        wp.addr.poke(a.U)
        wp.en.poke(true.B)
        wp.wdata.poke(valueAt(a, cfg.width).U)
        c.clock.step()
      }
      wp.en.poke(false.B)

      // read address i (original data) from the low half while
      // simultaneously overwriting the disjoint high half with new data on
      // the same cycle, so no address is both read and (earlier-in-loop)
      // overwritten within this same loop
      val half = cfg.depth / 2
      for (i <- 0 until half) {
        val overwriteAddr = half + i
        rp.addr.poke(i.U)
        rp.en.poke(true.B)
        wp.addr.poke(overwriteAddr.U)
        wp.en.poke(true.B)
        wp.wdata.poke(valueAt(i + 100, cfg.width).U)
        c.clock.step()
        rp.rdata.expect(valueAt(i, cfg.width).U)
      }
      wp.en.poke(false.B)

      // confirm the simultaneous writes landed correctly
      for (i <- 0 until half) {
        val overwriteAddr = half + i
        rp.addr.poke(overwriteAddr.U)
        rp.en.poke(true.B)
        c.clock.step()
        rp.rdata.expect(valueAt(i + 100, cfg.width).U)
      }
      rp.en.poke(false.B)
    }
  }

  "Sram (TwoRW)" should "let both ports write different addresses on the same cycle, into shared storage" in {
    val cfg = SramConfig(
      depth = 8,
      width = 8,
      numBanks = 1,
      maskGranularity = 8,
      readLatency = 1,
      writeLatency = 1,
      portType = BankPortType.TwoRW,
    )
    test(new Sram(cfg)) { c =>
      val p0 = c.io.ports(0)
      val p1 = c.io.ports(1)
      p0.wmask.poke(cfg.fullMask)
      p1.wmask.poke(cfg.fullMask)

      // port0 writes the low half, port1 writes the high half, same cycles
      val half = cfg.depth / 2
      for (i <- 0 until half) {
        p0.addr.poke(i.U)
        p0.en.poke(true.B)
        p0.wen.poke(true.B)
        p0.wdata.poke(valueAt(i, cfg.width).U)

        p1.addr.poke((half + i).U)
        p1.en.poke(true.B)
        p1.wen.poke(true.B)
        p1.wdata.poke(valueAt(half + i, cfg.width).U)

        c.clock.step()
      }
      p0.wen.poke(false.B)
      p1.wen.poke(false.B)

      // read every address back through the OTHER port than the one that
      // wrote it, proving both ports see the same shared storage
      for (i <- 0 until half) {
        p1.addr.poke(i.U)
        p1.en.poke(true.B)
        p0.addr.poke((half + i).U)
        p0.en.poke(true.B)
        c.clock.step()
        p1.rdata.expect(valueAt(i, cfg.width).U)
        p0.rdata.expect(valueAt(half + i, cfg.width).U)
      }
      p0.en.poke(false.B)
      p1.en.poke(false.B)
    }
  }
}
