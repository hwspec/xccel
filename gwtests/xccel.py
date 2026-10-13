import cocotb

from xccel_bridge import SRV32I_Bridge, ST_ECALL, ST_HALTED, ST_ILLEGAL, ST_RUNNING, ECALL

program = [
    0x12345537,   # lui x10, 0x12345          (rA hi)
    0x67850513,   # addi x10, x10, 0x678      (rA lo) -> rA=0x12345678
    0xabcdf5b7,   # lui x11, 0xabcdf          (rS hi)
    0xf0158593,   # addi x11, x11, -0xff (0xf01) -> rS=0xabcdef01
    0x000103b7,   # lui x7, 0x10              (t2 hi)
    0xf0038393,   # addi x7, x7, -256 (0xf00) -> t2=0x0000ff00
    0x00859e13,   # slli x28, x11, 8          -> t0 = rS<<8
    0x0185de93,   # srli x29, x11, 24         -> t1 = rS>>24
    0x01de6e33,   # or   x28, x28, x29        -> t0 = rotl(rS,8)
    0x007e7e33,   # and  x28, x28, x7         -> t0 &= mask
    0xfff3c393,   # xori x7, x7, -1           -> t2 = ~mask
    0x00757533,   # and  x10, x10, x7         -> rA &= ~mask
    0x01c56533,   # or   x10, x10, x28        -> rA |= t0
    ECALL,        # ecall (halt)
]


def rotl32(v, sh):
    v &= 0xFFFFFFFF
    return ((v << sh) | (v >> (32 - sh))) & 0xFFFFFFFF


@cocotb.test()
async def tb_rlwimieq(cocotb_dut):
    dut = SRV32I_Bridge(cocotb_dut)
    await dut.setup()
    await dut.softReset()

    st = await dut.run(program, entry=0)

    assert st & ST_ECALL, f"ecall not set: {st:#010x}"
    assert st & ST_HALTED, f"halted not set: {st:#010x}"
    assert not (st & ST_ILLEGAL), f"illegal set: {st:#010x}"
    assert not (st & ST_RUNNING), f"still running: {st:#010x}"

    CONST1 = 0x12345678
    CONST2 = 0xABCDEF01
    MASK = 0x0000FF00
    SH = 8

    rotated = rotl32(CONST2, SH)
    expected_rA = (CONST1 & ~MASK & 0xFFFFFFFF) | (rotated & MASK)

    assert await dut.readReg(11) == CONST2
    assert await dut.readReg(10) == expected_rA

    cycles = await dut.readCycles()
    dut.log.info(f"rlwimi cycles = {cycles - 6}")
    assert cycles > 0, "Cycle counter should be non-zero after execution"

    dut.log.info("Verified!")
