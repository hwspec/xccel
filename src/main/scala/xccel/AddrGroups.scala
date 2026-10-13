package xccel

import chisel3._

// Independent row addresses for a buffer's banks. The banks are split into
// addrGroups equal runs of consecutive banks, and each run reads and writes
// its own row: one beat can gather words from different rows, and one write
// can scatter a row's words to different rows. With one group every bank
// shares the address, so the banks act as one wide row. More groups cost
// one address (and stride) per group in every microcode slot that
// addresses the buffer.
//
// Group 0's address is the plain `addr` field of an interface; groups 1 ..
// addrGroups-1 come in an optional Vec next to it (None with one group, so
// a one-group interface keeps a single address and stays unchanged).
object AddrGroups {
  def validate(numBanks: Int, addrGroups: Int): Unit =
    require(addrGroups > 0 && numBanks % addrGroups == 0,
      s"addrGroups ($addrGroups) must divide numBanks ($numBanks)")

  // the address fields of groups 1 .. groups-1
  def extra[T <: Data](groups: Int, gen: => T): Option[Vec[T]] =
    if (groups > 1) Some(Vec(groups - 1, gen)) else None

  // every group's address, group 0 first
  def all[T <: Data](addr: T, extra: Option[Vec[T]]): Seq[T] = addr +: extra.toSeq.flatten

  // bank n's address
  def perBank[T <: Data](groupAddrs: Seq[T], numBanks: Int): Seq[T] = {
    val banksPerGroup = numBanks / groupAddrs.length
    Seq.tabulate(numBanks)(n => groupAddrs(n / banksPerGroup))
  }
}
