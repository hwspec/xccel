import cocotb
import numpy as np

from braggnn_model import BraggNN, scale_to_mult_shift
from braggnn_xccel import check_rows, load_and_run, lower_conv
from xccel_bridge import XCCEL_Bridge
from xccel_program import Table, schedule_drain, schedule_passes

PATCH = 0
N = 81  # pixels of the 9x9 NLB feature maps


def pad(vec, n=16):
    out = [0] * n
    out[:len(vec)] = [int(v) for v in vec]
    return out


def tile_lanes(jt, n=N, width=16):
    return min(width, n - jt * width)


async def setup(cocotb_dut):
    dut = XCCEL_Bridge(cocotb_dut)
    await dut.setup()
    await dut.softReset()
    m = BraggNN()
    return dut, m, m.forward(m.inputs[PATCH])


def requant_set(pset, scale, bias=None, relu=False, C=16):
    mult, shift = scale_to_mult_shift(scale)
    return (pset, pad(bias if bias is not None else [], C), [mult] * C, [shift] * C, 0 if relu else -128, 127)


@cocotb.test()
async def tb_braggnn_theta_phi(cocotb_dut):
    """theta . phi^T (81x32 . 32x81): phi is computed on chip (its 1x1 conv),
    drained through the Requantizer into the Transposer and written into
    WeightBuffer as phi^T (2 channel tiles x 6 pixel tiles = 12 blocks);
    theta (host-loaded) then streams against it. The logits are checked on
    the 81 valid lanes of each of the 6 pixel tiles."""
    dut, m, t = await setup(cocotb_dut)
    p = dut.p
    R = p.rows
    commit = p.acc_read_latency + p.acc_write_latency

    phi = lower_conv(t["conv1"], m.w["phi"], m.b["phi"], m.scale["phi"], False, t["phi"], R, p.cols)
    table = Table()
    end = schedule_passes(table, phi.passes, p)  # phi -> Accumulator rows cot*81 + pixel

    WBT = len(phi.wb_rows)  # phi^T tiles after phi's weights: tile (jt, ct)
    start = end + commit
    last_emit = None
    for jt in range(6):
        for ct in range(2):
            pixels = range(jt * 16, min(N, (jt + 1) * 16))
            items = [(ct * N + j, {"requant.paramSet": ct, "transposer.capture": 1, "transposer.row": r})
                     for r, j in enumerate(pixels)]
            emit = schedule_drain(table, start, items, p)
            for c in range(16):
                table.put(emit + c, {"wbWrite.en": 1, "wbWrite.addr": WBT + (jt * 2 + ct) * 16 + c,
                                     "wbWrite.transposed": 1, "wbWrite.column": c})
            last_emit = emit + 15
            start = emit + 16 - p.acc_read_latency  # next block's first capture after the last column read

    ABT, ACCL = len(phi.ab_rows), 2 * N
    passes = [{"ab": [ABT + ct * N + i for i in range(N)], "wb": WBT + (jt * 2 + ct) * 16,
               "acc": [ACCL + jt * N + i for i in range(N)], "accumulate": ct > 0}
              for jt in range(6) for ct in range(2)]
    end = schedule_passes(table, passes, p, start=last_emit + p.wb_write_latency + 1)

    theta = t["theta"].reshape(N, 32)
    ab_rows = phi.ab_rows + [(ABT + ct * N + i, pad(theta[i, ct * 16:(ct + 1) * 16])) for ct in range(2) for i in range(N)]
    requant = phi.requant + [requant_set(2, m.scale["theta_phi"])]
    await load_and_run(dut, "theta_phi", table, end, phi.wb_rows, ab_rows, requant)

    logits = t["theta_phi"]
    await check_rows(dut, "theta_phi", [(ACCL + jt * N + i, 2, pad(logits[i, jt * 16:(jt + 1) * 16]), tile_lanes(jt))
                                        for jt in range(6) for i in range(N)])
    dut.log.info("theta . phi^T verified")


@cocotb.test()
async def tb_braggnn_softmax(cocotb_dut):
    """Softmax over the 81 rows of 81 logits (6 beats per row, the last one
    1 lane): the host's int8 logits go into Accumulator through identity
    passes, then each pair of rows (two stat slots) is drained three times --
    Max, Sum, Divide, Out -- and written back to ActivationBuffer as the
    attention operand. Identity passes read it back for the check."""
    dut, m, t = await setup(cocotb_dut)
    p = dut.p
    consts = dut.layout["microword_consts"]
    op = consts["softmax_op"]
    div_latency = consts["softmax_div_latency"]
    commit = p.acc_read_latency + p.acc_write_latency

    logits = t["theta_phi"]
    ab_rows = [(jt * N + i, pad(logits[i, jt * 16:(jt + 1) * 16])) for jt in range(6) for i in range(N)]
    wb_rows = [(r, [1 if c == r else 0 for c in range(16)]) for r in range(16)]  # identity tile
    table = Table()
    passes = [{"ab": [jt * N + i for i in range(N)], "wb": 0, "acc": [jt * N + i for i in range(N)],
               "accumulate": False} for jt in range(6)]
    end = schedule_passes(table, passes, p)

    ABA, ACCV = 6 * N, 6 * N
    start = end + commit
    last_write = None
    for a in range(0, N, 2):
        pair = [(row, stat) for stat, row in enumerate((a, a + 1)) if row < N]

        def beats(row, stat, which, extra=None):
            return [(jt * N + row, {"requant.paramSet": 0, "softmax.op": which, "softmax.first": int(jt == 0),
                                    "softmax.lanes": tile_lanes(jt), "softmax.stat": stat, **(extra(jt) if extra else {})})
                    for jt in range(6)]
        items = []
        for row, stat in pair:
            items += beats(row, stat, op["Max"]) + beats(row, stat, op["Sum"])
        divide = schedule_drain(table, start, items, p)
        for k, (row, stat) in enumerate(pair):
            table.put(divide + k, {"softmax.op": op["Divide"], "softmax.stat": stat})
        out_items = []
        for row, stat in pair:
            out_items += beats(row, stat, op["Out"], lambda jt, row=row: {
                "abWrite.en": 1, "abWrite.addr": ABA + jt * N + row, "abWrite.fromRequantizer": 1})
        out_start = divide + len(pair) - 1 + div_latency - p.acc_read_latency
        start = schedule_drain(table, out_start, out_items, p) - p.acc_read_latency
        last_write = start + p.acc_read_latency - 1

    passes = [{"ab": [ABA + jt * N + i for i in range(N)], "wb": 0, "acc": [ACCV + jt * N + i for i in range(N)],
               "accumulate": False} for jt in range(6)]
    end = schedule_passes(table, passes, p, start=last_write + p.ab_write_latency + 1)
    requant = [(0, [0] * 16, [1] * 16, [0] * 16, -128, 127)]  # identity
    await load_and_run(dut, "softmax", table, end, wb_rows, ab_rows, requant, softmax=m.softmax)

    attention = t["attention"]
    await check_rows(dut, "softmax", [(ACCV + jt * N + i, 0, pad(attention[i, jt * 16:(jt + 1) * 16]))
                                      for jt in range(6) for i in range(N)])
    dut.log.info("softmax verified")


@cocotb.test()
async def tb_braggnn_attention_g(cocotb_dut):
    """attention . g (81x81 . 81x32): g is computed on chip (its 1x1 conv)
    and drained through the Requantizer straight into WeightBuffer, one
    weight row per pixel (the reduction index); the host's attention rows
    (6 pixel tiles) then stream against it."""
    dut, m, t = await setup(cocotb_dut)
    p = dut.p
    R = p.rows
    commit = p.acc_read_latency + p.acc_write_latency

    g = lower_conv(t["conv1"], m.w["g"], m.b["g"], m.scale["g"], False, t["g"], R, p.cols)
    table = Table()
    end = schedule_passes(table, g.passes, p)  # g -> Accumulator rows cot*81 + pixel

    WBG = len(g.wb_rows)  # g tiles: tile (cot, jt), row = pixel within jt
    items = [(cot * N + j, {"requant.paramSet": cot, "wbWrite.en": 1, "wbWrite.fromRequantizer": 1,
                            "wbWrite.addr": WBG + (cot * 6 + j // 16) * 16 + j % 16})
             for cot in range(2) for j in range(N)]
    last_write = schedule_drain(table, end + commit, items, p) - 1

    ABA, ACCO = len(g.ab_rows), 2 * N
    passes = [{"ab": [ABA + jt * N + i for i in range(N)], "wb": WBG + (cot * 6 + jt) * 16,
               "acc": [ACCO + cot * N + i for i in range(N)], "accumulate": jt > 0}
              for cot in range(2) for jt in range(6)]
    end = schedule_passes(table, passes, p, start=last_write + p.wb_write_latency + 1)

    attention = t["attention"]
    ab_rows = g.ab_rows + [(ABA + jt * N + i, pad(attention[i, jt * 16:(jt + 1) * 16])) for jt in range(6) for i in range(N)]
    requant = g.requant + [requant_set(2, m.scale["attention_g"])]
    await load_and_run(dut, "attention_g", table, end, g.wb_rows, ab_rows, requant)

    out = t["attention_g"].reshape(N, 32)
    await check_rows(dut, "attention_g", [(ACCO + cot * N + i, 2, pad(out[i, cot * 16:(cot + 1) * 16]))
                                          for cot in range(2) for i in range(N)])
    dut.log.info("attention . g verified")


@cocotb.test()
async def tb_braggnn_resadd(cocotb_dut):
    """resadd: ReLU(requant(a_q + b_q)) with a = conv1 output, b = NLB
    output, per channel tile: pass a through the residual scaler (set 0,
    overwrite), pass b (set 1, accumulate), host drain with C_scale + ReLU."""
    dut, m, t = await setup(cocotb_dut)
    p = dut.p
    a, b = t["conv1"].reshape(N, 64), t["nlb_out"].reshape(N, 64)
    ABB = 4 * N
    ab_rows = [(ct * N + i, pad(a[i, ct * 16:(ct + 1) * 16])) for ct in range(4) for i in range(N)] + \
              [(ABB + ct * N + i, pad(b[i, ct * 16:(ct + 1) * 16])) for ct in range(4) for i in range(N)]
    table = Table()
    k = 0
    for ct in range(4):
        for base, pset in ((0, 0), (ABB, 1)):
            for i in range(N):
                issue = k * N + i
                table.put(issue, {"abRead.en": 1, "abRead.addr": base + ct * N + i})
                table.put(issue + p.ab_read_latency, {"residual.en": 1, "residual.paramSet": pset,
                                                      "accWrite.addr": ct * N + i, "accWrite.accumulate": pset})
            k += 1
    end = k * N + p.ab_read_latency
    table.pad(end)
    residual = [requant_set(0, m.scale["resadd_a"]), requant_set(1, m.scale["resadd_b"])]
    await load_and_run(dut, "resadd", table, end, ab_rows=ab_rows, requant=[requant_set(0, m.scale["resadd_c"], relu=True)],
                       residual=residual)
    out = t["resadd"].reshape(N, 64)
    await check_rows(dut, "resadd", [(ct * N + i, 0, pad(out[i, ct * 16:(ct + 1) * 16])) for ct in range(4) for i in range(N)])
    dut.log.info("resadd verified")
