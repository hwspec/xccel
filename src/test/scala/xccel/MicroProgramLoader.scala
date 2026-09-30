package xccel

import chisel3._
import chiseltest._

// Loads MicroWords into consecutive program slots through a host-facing
// progWriteEn/progWriteAddr/progWriteData port. Every field not set by
// `fill` is zero (i.e. that slot is disabled for the word).
class MicroProgramLoader(clock: Clock, en: Bool, addr: UInt, data: MicroWord) {
  private var next = 0

  def word(cycles: Int, halt: Boolean = false)(fill: MicroWord => Unit = _ => ()): Unit = {
    MicroProgramLoader.clear(data)
    data.cycles.poke(cycles.U)
    data.halt.poke(halt.B)
    fill(data)
    addr.poke(next.U)
    en.poke(true.B)
    clock.step()
    next += 1
  }

  def done(): Unit = en.poke(false.B)
}

object MicroProgramLoader {
  def clear(d: Data): Unit = d match {
    case b: Bundle => b.elements.values.foreach(clear)
    case b: Bool   => b.poke(false.B)
    case u: UInt   => u.poke(0.U)
  }

  // all ones in `width` bits: a stride that steps backwards by one
  def minusOne(width: Int): UInt = ((BigInt(1) << width) - 1).U(width.W)
}

// One cycle of resolved control, for building a program as a per-cycle
// table (what each element does in each cycle) and letting
// CycleProgram.compress fold it into MicroWords. shiftAmt/groupShift and
// the accRead slot are left at zero.
case class CycleControl(
  wbEn: Boolean = false,
  wbAddr: Int = 0,
  wbwEn: Boolean = false,
  wbwAddr: Int = 0,
  wbwFromRequantizer: Boolean = false,
  wbwTransposed: Boolean = false,
  wbwColumn: Int = 0,
  tCapture: Boolean = false,
  tRow: Int = 0,
  smOp: Int = SoftmaxOp.None,
  smFirst: Boolean = false,
  smLanes: Int = 0,
  smStat: Int = 0,
  rsEn: Boolean = false,
  rsSet: Int = 0,
  abwEn: Boolean = false,
  abwAddr: Int = 0,
  abwFromRequantizer: Boolean = false,
  accrEn: Boolean = false,
  accrAddr: Int = 0,
  requantSet: Int = 0,
  wpEn: Boolean = false,
  wpAddr: Int = 0,
  wpSlot: Boolean = false,
  abEn: Boolean = false,
  abAddr: Int = 0,
  weightSelect: Boolean = false,
  accAddr: Int = 0,
  accumulate: Boolean = false,
) {
  def flags = (rsEn, rsSet, smOp, smFirst, smLanes, smStat, wbEn, wbwEn, wbwFromRequantizer, wbwTransposed, tCapture, abwEn, abwFromRequantizer, accrEn, requantSet, wpEn, wpSlot, abEn, weightSelect, accumulate)
  def addrs = Seq(wbAddr, wpAddr, abAddr, accAddr, wbwAddr, abwAddr, accrAddr, wbwColumn, tRow)
}

// A run of cycles folded into one word: `first` plus per-address strides.
case class FoldedWord(first: CycleControl, strides: Seq[Int], cycles: Int)

object CycleProgram {
  // Greedy: extend the current word while every flag is unchanged and every
  // address keeps advancing by the stride set by the word's first two cycles.
  def compress(cycles: Seq[CycleControl], maxRunLength: Int): Seq[FoldedWord] = {
    val words = scala.collection.mutable.ArrayBuffer.empty[FoldedWord]
    for (c <- cycles) {
      val extend = words.lastOption.exists { w =>
        w.cycles < maxRunLength && w.first.flags == c.flags && {
          val strides = if (w.cycles == 1) c.addrs.zip(w.first.addrs).map { case (a, b) => a - b } else w.strides
          c.addrs.indices.forall(i => c.addrs(i) == w.first.addrs(i) + strides(i) * w.cycles)
        }
      }
      if (extend) {
        val w = words.last
        val strides = if (w.cycles == 1) c.addrs.zip(w.first.addrs).map { case (a, b) => a - b } else w.strides
        words(words.length - 1) = w.copy(strides = strides, cycles = w.cycles + 1)
      } else {
        words += FoldedWord(c, Seq.fill(c.addrs.length)(0), 1)
      }
    }
    words.toSeq
  }

  // loads `words`, with a halt on the last one
  def load(p: MicroProgramLoader, words: Seq[FoldedWord]): Unit = {
    for ((fw, i) <- words.zipWithIndex) {
      val c = fw.first
      val Seq(wbS, wpS, abS, accS, wbwS, abwS, accrS, colS, rowS) = fw.strides
      p.word(cycles = fw.cycles, halt = i == words.length - 1) { w =>
        def poke(u: UInt, v: Int): Unit = u.poke((BigInt(v) & ((BigInt(1) << u.getWidth) - 1)).U(u.getWidth.W))
        w.wbRead.en.poke(c.wbEn.B)
        poke(w.wbRead.addr, c.wbAddr)
        poke(w.wbRead.addrStride, wbS)
        w.weightPort.en.poke(c.wpEn.B)
        w.weightPort.slot.poke(c.wpSlot.B)
        poke(w.weightPort.addr, c.wpAddr)
        poke(w.weightPort.addrStride, wpS)
        w.abRead.en.poke(c.abEn.B)
        poke(w.abRead.addr, c.abAddr)
        poke(w.abRead.addrStride, abS)
        w.compute.weightSelect.poke(c.weightSelect.B)
        poke(w.accWrite.addr, c.accAddr)
        poke(w.accWrite.addrStride, accS)
        w.accWrite.accumulate.poke(c.accumulate.B)
        w.wbWrite.en.poke(c.wbwEn.B)
        w.wbWrite.fromRequantizer.poke(c.wbwFromRequantizer.B)
        w.wbWrite.transposed.poke(c.wbwTransposed.B)
        poke(w.wbWrite.column, c.wbwColumn)
        poke(w.wbWrite.columnStride, colS)
        w.residual.en.poke(c.rsEn.B)
        poke(w.residual.paramSet, c.rsSet)
        poke(w.softmax.op, c.smOp)
        w.softmax.first.poke(c.smFirst.B)
        poke(w.softmax.lanes, c.smLanes)
        poke(w.softmax.stat, c.smStat)
        w.transposer.capture.poke(c.tCapture.B)
        poke(w.transposer.row, c.tRow)
        poke(w.transposer.rowStride, rowS)
        poke(w.wbWrite.addr, c.wbwAddr)
        poke(w.wbWrite.addrStride, wbwS)
        w.abWrite.en.poke(c.abwEn.B)
        w.abWrite.fromRequantizer.poke(c.abwFromRequantizer.B)
        w.accRead.en.poke(c.accrEn.B)
        poke(w.accRead.addr, c.accrAddr)
        poke(w.accRead.addrStride, accrS)
        poke(w.requant.paramSet, c.requantSet)
        poke(w.abWrite.addr, c.abwAddr)
        poke(w.abWrite.addrStride, abwS)
      }
    }
    p.done()
  }
}
