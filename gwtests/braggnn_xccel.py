"""Lowering BraggNN layers onto Xccel: buffer layouts, passes, Requantizer
parameter sets, and a runner that checks one lowered layer through AXI.

Layout conventions (R = rows = C_in tile, C = cols = C_out tile):
  ActivationBuffer  one pixel's R channels per row, row = base + ct * H*W + y*W + x
  WeightBuffer      one R-row 1x1 tile per (cot, ct, kernel position), row = base + tile*R + r
  Accumulator       one output pixel's C channels per row, row = base + cot * OH*OW + y*OW + x
Convs use accumulating kn2row with only the valid outputs streamed (passes
ordered cot -> ct -> kernel position); every C_out tile has its own
Requantizer parameter set because the bias differs per output channel.
"""

from dataclasses import dataclass, field

import numpy as np

from braggnn_model import scale_to_mult_shift
from xccel_program import Table, compress, schedule_passes


def ceil_div(a, b):
    return (a + b - 1) // b


@dataclass
class Lowered:
    ab_rows: list = field(default_factory=list)  # (addr, lanes)
    wb_rows: list = field(default_factory=list)  # (addr, lanes)
    passes: list = field(default_factory=list)  # schedule_passes entries
    requant: list = field(default_factory=list)  # (set, bias, mult, shift, lo, hi)
    outputs: list = field(default_factory=list)  # (acc row, set, expected lanes)


def lower_conv(x, w, b, scale, relu, expected, R, C, ab_base=0, wb_base=0, acc_base=0, set_base=0, acc_bases=None):
    """x [H, W, Cin] int8, w [K, K, Cin, Cout] (HWIO), b [Cout];
    expected [OH, OW, Cout] (the integer model's output). acc_bases, if
    given, places C_out tile cot at Accumulator rows acc_bases[cot] + pixel
    instead of acc_base + cot * OH*OW + pixel. Every pass also records
    "wrows", its weight tile's valid rows (the C_in tile's channels)."""
    H, W, CIN = x.shape
    K, _, _, COUT = w.shape
    OH, OW = H - K + 1, W - K + 1
    nct, ncot = ceil_div(CIN, R), ceil_div(COUT, C)
    low = Lowered()

    def lanes(vec, n):
        out = [0] * n
        out[:len(vec)] = [int(v) for v in vec]
        return out

    for ct in range(nct):
        for y in range(H):
            for xx in range(W):
                low.ab_rows.append((ab_base + ct * H * W + y * W + xx, lanes(x[y, xx, ct * R:(ct + 1) * R], R)))
    positions = [(kh, kw) for kh in range(K) for kw in range(K)]
    mult, shift = scale_to_mult_shift(scale)

    def acc_row(cot, pix):
        return (acc_bases[cot] if acc_bases else acc_base + cot * OH * OW) + pix

    for cot in range(ncot):
        cols = slice(cot * C, (cot + 1) * C)
        low.requant.append((set_base + cot, lanes(b[cols], C), [mult] * C, [shift] * C, 0 if relu else -128, 127))
        for ct in range(nct):
            for pos, (kh, kw) in enumerate(positions):
                tile = (cot * nct + ct) * len(positions) + pos
                for r in range(R):
                    ci = ct * R + r
                    row = w[kh, kw, ci, cols] if ci < CIN else []
                    low.wb_rows.append((wb_base + tile * R + r, lanes(row, C)))
                outs = [(y, xx) for y in range(OH) for xx in range(OW)]
                low.passes.append({
                    "ab": [ab_base + ct * H * W + (y + kh) * W + xx + kw for y, xx in outs],
                    "wb": wb_base + tile * R,
                    "acc": [acc_row(cot, y * OW + xx) for y, xx in outs],
                    "accumulate": not (ct == 0 and pos == 0),
                    "wrows": min(R, CIN - ct * R),
                })
        for y in range(OH):
            for xx in range(OW):
                low.outputs.append((acc_row(cot, y * OW + xx), set_base + cot,
                                    lanes(expected[y, xx, cols], C)))
    return low


def lower_conv_rows(x, w, b, scale, relu, expected, R, C, ab_base=0, wb_base=0, acc_base=0, set_base=0,
                    acc_bases=None):
    """Conv for a SystolicArrayN with K arrays (K x K kernel): one pass per
    (C_out tile, C_in tile, kernel row kh) loads W[kh, kw] into array kw
    ("loads": (WeightBuffer row, weight port address kw*R + r)) and streams
    the input rows kh .. kh+OH-1 whole. The result of stream beat i is output
    pixel p = i - (K-1) (row p // W, column p % W) when that window stays in
    one input row; the others ("acc" None) are discarded. Same ActivationBuffer
    / Accumulator layouts and outputs as lower_conv."""
    H, W, CIN = x.shape
    K, _, _, COUT = w.shape
    OH, OW = H - K + 1, W - K + 1
    nct, ncot = ceil_div(CIN, R), ceil_div(COUT, C)
    low = Lowered()

    def lanes(vec, n):
        out = [0] * n
        out[:len(vec)] = [int(v) for v in vec]
        return out

    for ct in range(nct):
        for y in range(H):
            for xx in range(W):
                low.ab_rows.append((ab_base + ct * H * W + y * W + xx, lanes(x[y, xx, ct * R:(ct + 1) * R], R)))
    mult, shift = scale_to_mult_shift(scale)

    def acc_row(cot, pix):
        return (acc_bases[cot] if acc_bases else acc_base + cot * OH * OW) + pix

    for cot in range(ncot):
        cols = slice(cot * C, (cot + 1) * C)
        low.requant.append((set_base + cot, lanes(b[cols], C), [mult] * C, [shift] * C, 0 if relu else -128, 127))
        for ct in range(nct):
            wrows = min(R, CIN - ct * R)
            for kh in range(K):
                loads = []
                for kw in range(K):
                    tile = ((cot * nct + ct) * K + kh) * K + kw
                    for r in range(R):
                        ci = ct * R + r
                        wb_row = wb_base + tile * R + r
                        low.wb_rows.append((wb_row, lanes(w[kh, kw, ci, cols] if ci < CIN else [], C)))
                        if r < wrows:
                            loads.append((wb_row, kw * R + r))
                acc = []
                for i in range(OH * W):
                    p = i - (K - 1)
                    acc.append(acc_row(cot, p // W * OW + p % W) if p >= 0 and p % W <= W - K else None)
                low.passes.append({
                    "ab": [ab_base + ct * H * W + (y + kh) * W + xx for y in range(OH) for xx in range(W)],
                    "wb": wb_base + ((cot * nct + ct) * K + kh) * K * R,
                    "loads": loads,
                    "acc": acc,
                    "accumulate": not (ct == 0 and kh == 0),
                    "wrows": wrows,
                })
        for y in range(OH):
            for xx in range(OW):
                low.outputs.append((acc_row(cot, y * OW + xx), set_base + cot,
                                    lanes(expected[y, xx, cols], C)))
    return low


def lower_conv_window(x, w, b, scale, relu, expected, R, C, pitch, ab_base=0, wb_base=0, acc_base=0, set_base=0,
                      acc_bases=None, window=3):
    """Conv for a SystolicArrayN holding a window x window block of kernel
    positions (array a = i*window + j) with wide weight beats. A K x K kernel
    (K >= window) is covered by window x window tiles with origins
    min(window*t, K - window) per axis; a tile owns only the positions no
    earlier tile covers, the others get zero weights. One pass per (C_out
    tile, C_in tile, kernel tile (a, b)) loads WeightBuffer rows
    r = [W[a+i, b+j, ci, cols] or 0 for each array] (one window^2 * C-word
    row per C_in lane, "loads": (row, r)) and streams the input in raster
    order from row a, `pitch` beats per input row (idle beats, "ab" None, after
    a row narrower than pitch), up to the beat its last output needs. The
    result of the beat of input (y + a + window-1, x + b + window-1) is that
    tile's part of output pixel (y, x); the others ("acc" None) are
    discarded. Same ActivationBuffer / Accumulator layouts and outputs as
    lower_conv."""
    H, W, CIN = x.shape
    K, _, _, COUT = w.shape
    OH, OW = H - K + 1, W - K + 1
    KW = window
    assert pitch >= W and K >= KW
    nct, ncot = ceil_div(CIN, R), ceil_div(COUT, C)
    origins = [min(KW * t, K - KW) for t in range(ceil_div(K, KW))]
    tiles = [(ta, tb) for ta in range(len(origins)) for tb in range(len(origins))]
    low = Lowered()

    def lanes(vec, n):
        out = [0] * n
        out[:len(vec)] = [int(v) for v in vec]
        return out

    for ct in range(nct):
        for y in range(H):
            for xx in range(W):
                low.ab_rows.append((ab_base + ct * H * W + y * W + xx, lanes(x[y, xx, ct * R:(ct + 1) * R], R)))
    mult, shift = scale_to_mult_shift(scale)

    def acc_row(cot, pix):
        return (acc_bases[cot] if acc_bases else acc_base + cot * OH * OW) + pix

    tile_rows = 0
    for cot in range(ncot):
        cols = slice(cot * C, (cot + 1) * C)
        low.requant.append((set_base + cot, lanes(b[cols], C), [mult] * C, [shift] * C, 0 if relu else -128, 127))
        for ct in range(nct):
            wrows = min(R, CIN - ct * R)
            for n, (ta, tb) in enumerate(tiles):
                a, bo = origins[ta], origins[tb]
                base = wb_base + tile_rows
                tile_rows += R
                for r in range(R):
                    ci = ct * R + r
                    row = []
                    for i in range(KW):
                        for j in range(KW):
                            kh, kw = a + i, bo + j
                            owned = kh >= KW * ta and kw >= KW * tb  # not covered by an earlier tile
                            row += lanes(w[kh, kw, ci, cols] if ci < CIN and owned else [], C)
                    low.wb_rows.append((base + r, row))
                ab, acc = [], []
                last = (OH - 1 + KW - 1) * pitch + OW - 1 + bo + KW - 1  # the last output's beat, from row a
                for s in range(last + 1):
                    yr, xx = divmod(s, pitch)
                    y = a + yr
                    ab.append(ab_base + ct * H * W + y * W + xx if xx < W else None)
                    oy, ox = yr - (KW - 1), xx - bo - (KW - 1)
                    acc.append(acc_row(cot, oy * OW + ox) if xx < W and 0 <= oy < OH and 0 <= ox < OW else None)
                low.passes.append({
                    "ab": ab,
                    "wb": base,
                    "loads": [(base + r, r) for r in range(wrows)],
                    "acc": acc,
                    "accumulate": ct > 0 or n > 0,
                    "wrows": wrows,
                })
        for y in range(OH):
            for xx in range(OW):
                low.outputs.append((acc_row(cot, y * OW + xx), set_base + cot,
                                    lanes(expected[y, xx, cols], C)))
    return low


def lower_conv_im2col(x, w, b, scale, relu, expected, R, C, ab_base=0, wb_base=0, acc_base=0, set_base=0):
    """Conv as a matmul over the im2col matrix the host materializes:
    ActivationBuffer row = one output pixel's K*K*Cin window values in one
    R-wide chunk (chunk-major), WeightBuffer = W' (K*K*Cin x Cout, column
    kk = (kh*K + kw)*Cin + ci) zero-padded to whole chunks."""
    H, W, CIN = x.shape
    K, _, _, COUT = w.shape
    OH, OW = H - K + 1, W - K + 1
    npix, kred = OH * OW, K * K * CIN
    nch, ncot = ceil_div(kred, R), ceil_div(COUT, C)
    low = Lowered()

    def lanes(vec, n):
        out = [0] * n
        out[:len(vec)] = [int(v) for v in vec]
        return out

    def window(y, xx):
        return [x[y + kk // CIN // K, xx + kk // CIN % K, kk % CIN] for kk in range(kred)]

    for ch in range(nch):
        for pix in range(npix):
            win = window(pix // OW, pix % OW)
            low.ab_rows.append((ab_base + ch * npix + pix, lanes(win[ch * R:(ch + 1) * R], R)))
    w2 = w.reshape(kred, COUT)  # HWIO flattens to (kh*K + kw)*Cin + ci
    mult, shift = scale_to_mult_shift(scale)
    for cot in range(ncot):
        cols = slice(cot * C, (cot + 1) * C)
        low.requant.append((set_base + cot, lanes(b[cols], C), [mult] * C, [shift] * C, 0 if relu else -128, 127))
        for ch in range(nch):
            tile = cot * nch + ch
            for r in range(R):
                kk = ch * R + r
                low.wb_rows.append((wb_base + tile * R + r, lanes(w2[kk, cols] if kk < kred else [], C)))
            low.passes.append({
                "ab": [ab_base + ch * npix + pix for pix in range(npix)],
                "wb": wb_base + tile * R,
                "acc": [acc_base + cot * npix + pix for pix in range(npix)],
                "accumulate": ch > 0,
                "wrows": min(R, kred - ch * R),
            })
        for pix in range(npix):
            low.outputs.append((acc_base + cot * npix + pix, set_base + cot,
                                lanes(expected[pix // OW, pix % OW, cols], C)))
    return low


def fc_as_conv(w_oc_hw, h, wd):
    """FC weights [O, C*H*W] in CHW flatten order -> HWIO conv weights over
    the whole [H, W, C] input (1x1 output)."""
    o = w_oc_hw.shape[0]
    return w_oc_hw.reshape(o, -1, h, wd).transpose(2, 3, 1, 0)


async def load_and_run(dut, name, table, end, wb_rows=(), ab_rows=(), requant=(), residual=(), softmax=None):
    """Load rows / parameters and the program of `table`, run it, and check
    that it takes exactly `end` cycles."""
    p = dut.p
    for addr, row in wb_rows:
        await dut.writeWbRow(addr, row)
    for addr, row in ab_rows:
        await dut.writeAbRow(addr, row)
    for args in requant:
        await dut.loadRequant(*args)
    for args in residual:
        await dut.loadResidual(*args)
    if softmax is not None:
        await dut.loadSoftmax(softmax["qln2"], softmax["qln2_inv"], softmax["qb"], softmax["qc"],
                              softmax["max_shift"], softmax["k"], softmax["out_shift"])
    words = compress(table.cycles, p.max_run_length, dut.layout["microword"])
    assert len(words) <= p.program_length, f"{name}: {len(words)} words exceed the program store"
    await dut.loadProgram(words)
    dut.log.info(f"{name}: {len(words)} words, predicted {end} cycles")
    await dut.run()
    cycles = await dut.readCycles()
    dut.log.info(f"{name}: cycles = {cycles}")
    assert cycles == end, f"{name}: measured {cycles} cycles, predicted {end}"
    return cycles


async def check_rows(dut, name, outputs):
    """outputs: (acc row, requant set, expected lanes[, number of lanes to compare])."""
    mismatches = 0
    for out in outputs:
        addr, pset, exp = out[:3]
        n = out[3] if len(out) > 3 else len(exp)
        got = await dut.readAcc(addr, pset)
        if got[:n] != list(exp[:n]):
            mismatches += 1
            if mismatches <= 5:
                dut.log.info(f"{name}: acc row {addr}: got {got[:n]}, expected {list(exp[:n])}")
    assert mismatches == 0, f"{name}: {mismatches} of {len(outputs)} output rows differ"


async def run_lowered(dut, low, name):
    """Load a lowered layer, run it, check its cycles and every output."""
    table = Table()
    end = schedule_passes(table, low.passes, dut.p)
    dut.log.info(f"{name}: {len(low.passes)} passes")
    cycles = await load_and_run(dut, name, table, end, low.wb_rows, low.ab_rows, low.requant)
    await check_rows(dut, name, low.outputs)
    return cycles
