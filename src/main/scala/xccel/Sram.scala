package xccel

import chisel3._
import chisel3.util._

sealed trait BankPortType
object BankPortType {
  case object OneRW extends BankPortType // 1RW: one shared read/write port
  case object OneR1W extends BankPortType // 1R1W: one dedicated read port + one dedicated write port
  case object TwoRW extends BankPortType // 2RW: two independent read-or-write ports
}

// Each bank is wired directly to its own external access channel
case class SramConfig(
  depth: Int,
  wordLength: Int,
  numBanks: Int,
  bankBandwidth: Int,
  maskGranularity: Int,
  readLatency: Int,
  writeLatency: Int,
  portType: BankPortType = BankPortType.OneRW,
) {
  require(depth > 0, "depth must be positive")
  require(wordLength > 0, "wordLength must be positive")
  require(numBanks > 0, "numBanks must be positive")
  require(bankBandwidth > 0, "bankBandwidth must be positive")
  require(maskGranularity > 0, "maskGranularity must be positive")
  require(readLatency > 0, "readLatency must be positive")
  require(writeLatency > 0, "writeLatency must be positive")
  require(
    bankBandwidth % wordLength == 0,
    s"bankBandwidth ($bankBandwidth) must be a multiple of wordLength ($wordLength);" +
      " a bank narrower than one word isn't modeled here"
  )
  require(
    wordLength % maskGranularity == 0,
    s"wordLength ($wordLength) must be a multiple of maskGranularity ($maskGranularity)"
  )
  require(depth % numBanks == 0, s"depth ($depth) must be a multiple of numBanks ($numBanks)")

  // depth of the single bank owned by each channel
  val depthPerBank: Int = depth / numBanks
  // words packed into a single physical row (one addr access) of a bank
  val wordsPerRow: Int = bankBandwidth / wordLength
  require(
    depthPerBank % wordsPerRow == 0,
    s"depthPerBank ($depthPerBank = depth/numBanks) must be a multiple of" +
      s" wordsPerRow ($wordsPerRow = bankBandwidth/wordLength)"
  )

  // how many addressable rows each bank has
  val rowsPerBank: Int = depthPerBank / wordsPerRow
  val maskBits: Int = bankBandwidth / maskGranularity
  val addrWidth: Int = log2Ceil(rowsPerBank)

  val channelsPerBank: Int = if (portType == BankPortType.OneRW) 1 else 2
  val totalChannels: Int = numBanks * channelsPerBank
}

// A read/write channel: one address bus (row-addressed, within this
// bank), `en` gates activity this cycle, `wen` selects write (true) vs
// read (false). For a OneR1W write-only channel `wen` is unused (always a
// write); for its paired read-only channel `wen`/`wdata`/`wmask` are unused.
class SramBankIO(cfg: SramConfig) extends Bundle {
  val addr = Input(UInt(cfg.addrWidth.W))
  val en = Input(Bool())
  val wen = Input(Bool())
  val wdata = Input(UInt(cfg.bankBandwidth.W))
  val wmask = Input(UInt(cfg.maskBits.W))
  val rdata = Output(UInt(cfg.bankBandwidth.W))
}

class SramIO(cfg: SramConfig) extends Bundle {
  val banks = Vec(cfg.totalChannels, new SramBankIO(cfg))
}


class Sram(cfg: SramConfig) extends Module {
  val io = IO(new SramIO(cfg))

  private val mems = Seq.fill(cfg.numBanks)(
    SyncReadMem(cfg.rowsPerBank, Vec(cfg.maskBits, UInt(cfg.maskGranularity.W)))
  )

  // Splitting a write value into chunks
  private def toChunks(data: UInt): Vec[UInt] =
    VecInit(Seq.tabulate(cfg.maskBits)(i => data((i + 1) * cfg.maskGranularity - 1, i * cfg.maskGranularity)))

  // reverse because little-endian
  private def fromChunks(vec: Vec[UInt]): UInt = Cat(vec.reverse)

  // A shared read/write channel on one bank. Modeled with SyncReadMem.readWrite
  private def readWriteAccess(b: SramBankIO, mem: SyncReadMem[Vec[UInt]]): Unit = {
    val wAddrPipe = ShiftRegister(b.addr, cfg.writeLatency - 1, 0.U(cfg.addrWidth.W), true.B)
    val wEnPipe = ShiftRegister(b.en && b.wen, cfg.writeLatency - 1, false.B, true.B)
    val wDataPipe = ShiftRegister(b.wdata, cfg.writeLatency - 1, 0.U(cfg.bankBandwidth.W), true.B)
    val wMaskPipe = ShiftRegister(b.wmask, cfg.writeLatency - 1, 0.U(cfg.maskBits.W), true.B)

    val readReq = b.en && !b.wen
    val bankIsWrite = wEnPipe
    val bankEn = readReq || wEnPipe
    val bankAddr = Mux(bankIsWrite, wAddrPipe, b.addr)

    val raw = mem.readWrite(bankAddr, toChunks(wDataPipe), wMaskPipe.asBools, bankEn, bankIsWrite)
    b.rdata := ShiftRegister(fromChunks(raw), cfg.readLatency - 1, 0.U(cfg.bankBandwidth.W), true.B)
  }

  // A dedicated read-only channel
  private def readOnlyAccess(b: SramBankIO, mem: SyncReadMem[Vec[UInt]]): Unit = {
    val raw = mem.read(b.addr, b.en)
    b.rdata := ShiftRegister(fromChunks(raw), cfg.readLatency - 1, 0.U(cfg.bankBandwidth.W), true.B)
  }

  // The write-only counterpart to readOnlyAccess; rdata is unused (tied low).
  private def writeOnlyAccess(b: SramBankIO, mem: SyncReadMem[Vec[UInt]]): Unit = {
    val wAddrPipe = ShiftRegister(b.addr, cfg.writeLatency - 1, 0.U(cfg.addrWidth.W), true.B)
    val wEnPipe = ShiftRegister(b.en, cfg.writeLatency - 1, false.B, true.B)
    val wDataPipe = ShiftRegister(b.wdata, cfg.writeLatency - 1, 0.U(cfg.bankBandwidth.W), true.B)
    val wMaskPipe = ShiftRegister(b.wmask, cfg.writeLatency - 1, 0.U(cfg.maskBits.W), true.B)

    when(wEnPipe) {
      mem.write(wAddrPipe, toChunks(wDataPipe), wMaskPipe.asBools)
    }
    b.rdata := 0.U
  }

  cfg.portType match {
    case BankPortType.OneRW =>
      for (n <- 0 until cfg.numBanks) readWriteAccess(io.banks(n), mems(n))

    case BankPortType.OneR1W =>
      for (n <- 0 until cfg.numBanks) {
        readOnlyAccess(io.banks(2 * n), mems(n))
        writeOnlyAccess(io.banks(2 * n + 1), mems(n))
      }

    case BankPortType.TwoRW =>
      for (n <- 0 until cfg.numBanks) {
        readWriteAccess(io.banks(2 * n), mems(n))
        readWriteAccess(io.banks(2 * n + 1), mems(n))
      }
  }
}
