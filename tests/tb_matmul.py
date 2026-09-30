import random

import cocotb

from xccel_bridge import XCCEL_Bridge
from xccel_program import Table, compress, schedule_passes


def requant(v, bias, mult, shift, lo, hi):
    """Requantizer: clamp(roundShift((v + bias) * mult, shift), lo, hi),
    rounding half away from zero."""
    p = (v + bias) * mult
    if shift:
        half = 1 << (shift - 1)
        p = (p + half - 1 if p < 0 else p + half) >> shift
    return max(lo, min(hi, p))


@cocotb.test()
async def tb_matmul(cocotb_dut):
    """One quantized matmul layer on the 16x16 array: Y = X . W with
    C_in = 32 (two C_in tiles accumulated in Accumulator), C_out = 16,
    N = 20 rows, requantized per lane (bias, scale, ReLU) on the host drain."""
    dut = XCCEL_Bridge(cocotb_dut)
    await dut.setup()
    await dut.softReset()
    p = dut.p
    R, C = p.rows, p.cols
    CIN, N = 2 * R, 20

    rnd = random.Random(1)
    x = [[rnd.randint(-128, 127) for _ in range(CIN)] for _ in range(N)]
    w = [[rnd.randint(-128, 127) for _ in range(C)] for _ in range(CIN)]
    bias = [rnd.randint(-5000, 5000) for _ in range(C)]
    mult = [rnd.randint(1, 255) for _ in range(C)]
    shift = [rnd.randint(14, 17) for _ in range(C)]
    acc = [[sum(x[n][k] * w[k][j] for k in range(CIN)) for j in range(C)] for n in range(N)]
    expected = [[requant(acc[n][j], bias[j], mult[j], shift[j], 0, 127) for j in range(C)] for n in range(N)]
    assert any(v == 0 for row in expected for v in row) and any(v == 127 for row in expected for v in row), \
        "data should exercise ReLU and saturation"

    # WeightBuffer: C_in tile t at rows t*R ..; ActivationBuffer: tile t of
    # sample n at t*N + n; Accumulator: row n
    for t in range(2):
        for r in range(R):
            await dut.writeWbRow(t * R + r, w[t * R + r])
        for n in range(N):
            await dut.writeAbRow(t * N + n, x[n][t * R:(t + 1) * R])
    await dut.loadRequant(0, bias, mult, shift, 0, 127)

    passes = [{"ab": [t * N + n for n in range(N)], "wb": t * R, "acc": list(range(N)), "accumulate": t > 0}
              for t in range(2)]
    table = Table()
    predicted = schedule_passes(table, passes, p)
    words = compress(table.cycles, p.max_run_length, dut.layout["microword"])
    dut.log.info(f"program: {len(words)} words, predicted {predicted} cycles")
    await dut.loadProgram(words)

    await dut.run()
    cycles = await dut.readCycles()
    dut.log.info(f"cycles = {cycles}")
    assert cycles == predicted, f"measured {cycles} cycles, predicted {predicted}"

    for n in range(N):
        got = await dut.readAcc(n, pset=0)
        assert got == expected[n], f"row {n}: got {got}, expected {expected[n]}"

    dut.log.info("Verified!")
