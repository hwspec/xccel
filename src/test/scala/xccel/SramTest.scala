package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SramTest extends AnyFlatSpec with ChiselScalatestTester {

  def valueAt(a: Int, width: Int): BigInt = {
    val mask = (BigInt(1) << width) - 1
    (BigInt(a) * 7 + 3) & mask
  }

  def allOnesMask(cfg: SramConfig): UInt = ((BigInt(1) << cfg.maskBits) - 1).U(cfg.maskBits.W)

  "Sram" should "read back what was written, respecting readLatency/writeLatency (1 bank, 1 word/row)" in {
    val cfg = SramConfig(
      depth = 32,
      wordLength = 16,
      numBanks = 1,
      bankBandwidth = 16,
      maskGranularity = 8,
      readLatency = 2,
      writeLatency = 2,
    )
    test(new Sram(cfg)) { c =>
      val b = c.io.banks(0)
      b.wmask.poke(allOnesMask(cfg))

      for (a <- 0 until cfg.rowsPerBank) {
        b.addr.poke(a.U)
        b.en.poke(true.B)
        b.wen.poke(true.B)
        b.wdata.poke(valueAt(a, cfg.bankBandwidth).U)
        c.clock.step()
      }
      b.en.poke(false.B)
      b.wen.poke(false.B)
      c.clock.step(cfg.writeLatency)

      for (a <- 0 until cfg.rowsPerBank) {
        b.addr.poke(a.U)
        b.en.poke(true.B)
        b.wen.poke(false.B)
        c.clock.step(cfg.readLatency)
        b.rdata.expect(valueAt(a, cfg.bankBandwidth).U)
      }
      b.en.poke(false.B)
    }
  }

  it should "keep each bank independent (same address, different banks, different data)" in {
    val cfg = SramConfig(
      depth = 16,
      wordLength = 8,
      numBanks = 2,
      bankBandwidth = 8,
      maskGranularity = 8,
      readLatency = 1,
      writeLatency = 1,
    )
    test(new Sram(cfg)) { c =>
      val b0 = c.io.banks(0)
      val b1 = c.io.banks(1)
      b0.wmask.poke(allOnesMask(cfg))
      b1.wmask.poke(allOnesMask(cfg))

      // write every address on both banks simultaneously, but with
      // different data per bank, to confirm they don't share storage
      for (a <- 0 until cfg.rowsPerBank) {
        b0.addr.poke(a.U)
        b0.en.poke(true.B)
        b0.wen.poke(true.B)
        b0.wdata.poke(valueAt(a, cfg.bankBandwidth).U)

        b1.addr.poke(a.U)
        b1.en.poke(true.B)
        b1.wen.poke(true.B)
        b1.wdata.poke(valueAt(a + 100, cfg.bankBandwidth).U)

        c.clock.step()
      }
      b0.en.poke(false.B)
      b0.wen.poke(false.B)
      b1.en.poke(false.B)
      b1.wen.poke(false.B)

      for (a <- 0 until cfg.rowsPerBank) {
        b0.addr.poke(a.U)
        b0.en.poke(true.B)
        b0.wen.poke(false.B)
        b1.addr.poke(a.U)
        b1.en.poke(true.B)
        b1.wen.poke(false.B)
        c.clock.step()
        b0.rdata.expect(valueAt(a, cfg.bankBandwidth).U)
        b1.rdata.expect(valueAt(a + 100, cfg.bankBandwidth).U)
      }
      b0.en.poke(false.B)
      b1.en.poke(false.B)
    }
  }

  it should "pack multiple words into one wide-bank row" in {
    val cfg = SramConfig(
      depth = 16,
      wordLength = 8,
      numBanks = 1,
      bankBandwidth = 32, // 4 words per row
      maskGranularity = 8,
      readLatency = 1,
      writeLatency = 1,
    )
    assert(cfg.wordsPerRow == 4)

    test(new Sram(cfg)) { c =>
      val bank = c.io.banks(0)
      bank.wmask.poke(allOnesMask(cfg))

      // each row b packs words [4b, 4b+1, 4b+2, 4b+3], word w's value = w+1
      for (rowIdx <- 0 until cfg.rowsPerBank) {
        val words = (0 until cfg.wordsPerRow).map(i => BigInt(rowIdx * cfg.wordsPerRow + i + 1))
        val rowValue = words.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (w, i)) =>
          acc | (w << (i * cfg.wordLength))
        }
        bank.addr.poke(rowIdx.U)
        bank.en.poke(true.B)
        bank.wen.poke(true.B)
        bank.wdata.poke(rowValue.U)
        c.clock.step()
      }
      bank.en.poke(false.B)
      bank.wen.poke(false.B)

      for (rowIdx <- 0 until cfg.rowsPerBank) {
        val words = (0 until cfg.wordsPerRow).map(i => BigInt(rowIdx * cfg.wordsPerRow + i + 1))
        val expected = words.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (w, i)) =>
          acc | (w << (i * cfg.wordLength))
        }
        bank.addr.poke(rowIdx.U)
        bank.en.poke(true.B)
        bank.wen.poke(false.B)
        c.clock.step()
        bank.rdata.expect(expected.U)
      }
      bank.en.poke(false.B)
    }
  }

  it should "assert when a read is requested in the cycle a delayed write (writeLatency > 1) reaches the channel" in {
    val cfg = SramConfig(depth = 8, wordLength = 8, numBanks = 1, bankBandwidth = 8, maskGranularity = 8,
      readLatency = 1, writeLatency = 2)
    intercept[chisel3.simulator.Exceptions.AssertionFailed] {
      test(new Sram(cfg)) { c =>
        val b = c.io.banks(0)
        b.wmask.poke(allOnesMask(cfg))
        // cycle 0: write row 1 (reaches the bank at cycle 1)
        b.addr.poke(1.U); b.en.poke(true.B); b.wen.poke(true.B); b.wdata.poke(5.U)
        c.clock.step()
        // cycle 1: read row 2 while that write takes the channel
        b.addr.poke(2.U); b.wen.poke(false.B)
        c.clock.step()
      }
    }
  }

  "Sram (OneR1W)" should "read one address while simultaneously writing a different one" in {
    val cfg = SramConfig(
      depth = 8,
      wordLength = 8,
      numBanks = 1,
      bankBandwidth = 8,
      maskGranularity = 8,
      readLatency = 1,
      writeLatency = 1,
      portType = BankPortType.OneR1W,
    )
    test(new Sram(cfg)) { c =>
      val rp = c.io.banks(0)
      val wp = c.io.banks(1)
      wp.wmask.poke(allOnesMask(cfg))

      for (a <- 0 until cfg.rowsPerBank) {
        wp.addr.poke(a.U)
        wp.en.poke(true.B)
        wp.wdata.poke(valueAt(a, cfg.bankBandwidth).U)
        c.clock.step()
      }
      wp.en.poke(false.B)

      // read address i (original data) from the low half while
      // simultaneously overwriting the disjoint high half with new data on
      // the same cycle, so no address is both read and (earlier-in-loop)
      // overwritten within this same loop
      val half = cfg.rowsPerBank / 2
      for (i <- 0 until half) {
        val overwriteAddr = half + i
        rp.addr.poke(i.U)
        rp.en.poke(true.B)
        wp.addr.poke(overwriteAddr.U)
        wp.en.poke(true.B)
        wp.wdata.poke(valueAt(i + 100, cfg.bankBandwidth).U)
        c.clock.step()
        rp.rdata.expect(valueAt(i, cfg.bankBandwidth).U)
      }
      wp.en.poke(false.B)

      // confirm the simultaneous writes landed correctly
      for (i <- 0 until half) {
        val overwriteAddr = half + i
        rp.addr.poke(overwriteAddr.U)
        rp.en.poke(true.B)
        c.clock.step()
        rp.rdata.expect(valueAt(i + 100, cfg.bankBandwidth).U)
      }
      rp.en.poke(false.B)
    }
  }

  "Sram (TwoRW)" should "let both ports write different addresses on the same cycle, into shared storage" in {
    val cfg = SramConfig(
      depth = 8,
      wordLength = 8,
      numBanks = 1,
      bankBandwidth = 8,
      maskGranularity = 8,
      readLatency = 1,
      writeLatency = 1,
      portType = BankPortType.TwoRW,
    )
    test(new Sram(cfg)) { c =>
      val p0 = c.io.banks(0)
      val p1 = c.io.banks(1)
      p0.wmask.poke(allOnesMask(cfg))
      p1.wmask.poke(allOnesMask(cfg))

      // port0 writes the low half, port1 writes the high half, same cycles
      val half = cfg.rowsPerBank / 2
      for (i <- 0 until half) {
        p0.addr.poke(i.U)
        p0.en.poke(true.B)
        p0.wen.poke(true.B)
        p0.wdata.poke(valueAt(i, cfg.bankBandwidth).U)

        p1.addr.poke((half + i).U)
        p1.en.poke(true.B)
        p1.wen.poke(true.B)
        p1.wdata.poke(valueAt(half + i, cfg.bankBandwidth).U)

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
        p1.rdata.expect(valueAt(i, cfg.bankBandwidth).U)
        p0.rdata.expect(valueAt(half + i, cfg.bankBandwidth).U)
      }
      p0.en.poke(false.B)
      p1.en.poke(false.B)
    }
  }
}
