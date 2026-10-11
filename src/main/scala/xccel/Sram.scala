package xccel

import chisel3._
import chisel3.util._

sealed trait BankPortType
object BankPortType {
  case object OneRW extends BankPortType // 1RW: one shared read/write port
  case object OneR1W extends BankPortType // 1R1W: one dedicated read port + one dedicated write port
  case object TwoRW extends BankPortType // 2RW: two independent read-or-write ports
}

// Each bank is wired directly to its own external access port. As in
// a memory macro, a word is one row: one address reads or writes `width`
// bits, and what those bits hold (e.g. several lanes) is up to the caller.
case class SramConfig(
  depth: Int, // rows (words) per bank
  width: Int, // bits per row
  numBanks: Int,
  maskGranularity: Int,
  readLatency: Int,
  writeLatency: Int,
  portType: BankPortType = BankPortType.OneRW,
) {
  require(depth > 0, "depth must be positive")
  require(width > 0, "width must be positive")
  require(numBanks > 0, "numBanks must be positive")
  require(maskGranularity > 0, "maskGranularity must be positive")
  require(readLatency > 0, "readLatency must be positive")
  require(writeLatency > 0, "writeLatency must be positive")
  require(
    width % maskGranularity == 0,
    s"width ($width) must be a multiple of maskGranularity ($maskGranularity)"
  )

  val maskBits: Int = width / maskGranularity
  // a wmask that writes the whole row
  def fullMask: UInt = ((BigInt(1) << maskBits) - 1).U(maskBits.W)
  val addrWidth: Int = log2Ceil(depth)

  val portsPerBank: Int = if (portType == BankPortType.OneRW) 1 else 2
  val totalPorts: Int = numBanks * portsPerBank
}

// A read/write port: one address bus (row-addressed, within this
// bank), `en` gates activity this cycle, `wen` selects write (true) vs
// read (false). For a OneR1W write-only port `wen` is unused (always a
// write); for its paired read-only port `wen`/`wdata`/`wmask` are unused.
class SramPortIO(cfg: SramConfig) extends Bundle {
  val addr = Input(UInt(cfg.addrWidth.W))
  val en = Input(Bool())
  val wen = Input(Bool())
  val wdata = Input(UInt(cfg.width.W))
  val wmask = Input(UInt(cfg.maskBits.W))
  val rdata = Output(UInt(cfg.width.W))
}

class SramIO(cfg: SramConfig) extends Bundle {
  val ports = Vec(cfg.totalPorts, new SramPortIO(cfg))
}


object Sram {
  // Drives one read stream and one write stream onto an Sram's ports,
  // for any port type, and returns each bank's rdata. `wdata(n)` is bank
  // n's packed row and `wmask(n)` its write mask. OneRW: both share the
  // bank's single port; a read takes it, so a write in the same cycle is
  // dropped, and with writeLatency > 1 an accepted write takes the port
  // writeLatency - 1 cycles later, when a read requested in that cycle
  // returns undefined data (the caller must keep both collisions apart).
  // OneR1W: each stream gets its own port. TwoRW: port 0 reads, port 1
  // writes (its ability to do two reads per cycle is not used here).
  def connectReadWrite(
    cfg: SramConfig,
    io: SramIO,
    readAddr: UInt,
    readEn: Bool,
    writeAddr: UInt,
    writeEn: Bool,
    wdata: Seq[UInt],
    wmask: Seq[UInt],
  ): Seq[UInt] = {
    (0 until cfg.numBanks).map { n =>
      cfg.portType match {
        case BankPortType.OneRW =>
          val p = io.ports(n)
          p.addr := Mux(readEn, readAddr, writeAddr)
          p.en := readEn || writeEn
          p.wen := !readEn
          p.wdata := wdata(n)
          p.wmask := wmask(n)
          p.rdata
        case BankPortType.OneR1W | BankPortType.TwoRW =>
          val r = io.ports(2 * n)
          val w = io.ports(2 * n + 1)
          r.addr := readAddr
          r.en := readEn
          r.wen := false.B
          r.wdata := 0.U
          r.wmask := 0.U
          w.addr := writeAddr
          w.en := writeEn
          w.wen := true.B
          w.wdata := wdata(n)
          w.wmask := wmask(n)
          r.rdata
      }
    }
  }
}

class Sram(cfg: SramConfig) extends Module {
  val io = IO(new SramIO(cfg))

  private val mems = Seq.fill(cfg.numBanks)(
    SyncReadMem(cfg.depth, Vec(cfg.maskBits, UInt(cfg.maskGranularity.W)))
  )

  // Splitting a write value into chunks
  private def toChunks(data: UInt): Vec[UInt] =
    VecInit(Seq.tabulate(cfg.maskBits)(i => data((i + 1) * cfg.maskGranularity - 1, i * cfg.maskGranularity)))

  // reverse because little-endian
  private def fromChunks(vec: Vec[UInt]): UInt = Cat(vec.reverse)

  // A shared read/write port on one bank. Modeled with SyncReadMem.readWrite
  private def readWriteAccess(b: SramPortIO, mem: SyncReadMem[Vec[UInt]]): Unit = {
    val wAddrPipe = ShiftRegister(b.addr, cfg.writeLatency - 1, 0.U(cfg.addrWidth.W), true.B)
    val wEnPipe = ShiftRegister(b.en && b.wen, cfg.writeLatency - 1, false.B, true.B)
    val wDataPipe = ShiftRegister(b.wdata, cfg.writeLatency - 1, 0.U(cfg.width.W), true.B)
    val wMaskPipe = ShiftRegister(b.wmask, cfg.writeLatency - 1, 0.U(cfg.maskBits.W), true.B)

    val readReq = b.en && !b.wen
    val bankIsWrite = wEnPipe
    val bankEn = readReq || wEnPipe
    val bankAddr = Mux(bankIsWrite, wAddrPipe, b.addr)

    // a write accepted writeLatency - 1 cycles ago takes the port now, so
    // a read requested in this cycle would return undefined data (never with
    // writeLatency = 1: then the write and the read are the same request)
    assert(!(readReq && wEnPipe),
      "Sram: read requested in the cycle a delayed write reaches the same port (writeLatency > 1)")
    val raw = mem.readWrite(bankAddr, toChunks(wDataPipe), wMaskPipe.asBools, bankEn, bankIsWrite)
    b.rdata := ShiftRegister(fromChunks(raw), cfg.readLatency - 1, 0.U(cfg.width.W), true.B)
  }

  // A dedicated read-only port
  private def readOnlyAccess(b: SramPortIO, mem: SyncReadMem[Vec[UInt]]): Unit = {
    val raw = mem.read(b.addr, b.en)
    b.rdata := ShiftRegister(fromChunks(raw), cfg.readLatency - 1, 0.U(cfg.width.W), true.B)
  }

  // The write-only counterpart to readOnlyAccess; rdata is unused (tied low).
  private def writeOnlyAccess(b: SramPortIO, mem: SyncReadMem[Vec[UInt]]): Unit = {
    val wAddrPipe = ShiftRegister(b.addr, cfg.writeLatency - 1, 0.U(cfg.addrWidth.W), true.B)
    val wEnPipe = ShiftRegister(b.en, cfg.writeLatency - 1, false.B, true.B)
    val wDataPipe = ShiftRegister(b.wdata, cfg.writeLatency - 1, 0.U(cfg.width.W), true.B)
    val wMaskPipe = ShiftRegister(b.wmask, cfg.writeLatency - 1, 0.U(cfg.maskBits.W), true.B)

    when(wEnPipe) {
      mem.write(wAddrPipe, toChunks(wDataPipe), wMaskPipe.asBools)
    }
    b.rdata := 0.U
  }

  cfg.portType match {
    case BankPortType.OneRW =>
      for (n <- 0 until cfg.numBanks) readWriteAccess(io.ports(n), mems(n))

    case BankPortType.OneR1W =>
      for (n <- 0 until cfg.numBanks) {
        readOnlyAccess(io.ports(2 * n), mems(n))
        writeOnlyAccess(io.ports(2 * n + 1), mems(n))
      }

    case BankPortType.TwoRW =>
      for (n <- 0 until cfg.numBanks) {
        readWriteAccess(io.ports(2 * n), mems(n))
        readWriteAccess(io.ports(2 * n + 1), mems(n))
      }
  }
}
