"""The whole BraggNN inference as one Xccel program.

Only conv1's im2col rows come from the host for each patch; every other
activation stays on chip, passed from layer to layer through the
Requantizer paths (Accumulator -> ActivationBuffer / WeightBuffer /
Transposer). Weights, Requantizer / residual / Softmax parameters and the
program are loaded once. Phases run one after another; each waits for the
previous one's results (Accumulator commitLatency, buffer writeLatency + 1).
"""

from dataclasses import dataclass, field

import numpy as np

from braggnn_model import scale_to_mult_shift
from braggnn_xccel import fc_as_conv, lower_conv, lower_conv_im2col
from xccel_program import Table, schedule_drain, schedule_passes

N = 81  # pixels of the 9x9 feature maps


def pad(vec, n=16):
    out = [0] * n
    out[:len(vec)] = [int(v) for v in vec]
    return out


class Allocator:
    def __init__(self, name, size):
        self.name, self.size, self.next = name, size, 0

    def take(self, n):
        base = self.next
        self.next += n
        assert self.next <= self.size, f"{self.name}: {self.next} rows exceed {self.size}"
        return base


@dataclass
class Inference:
    table: Table
    end: int  # cycles from start to halt
    marks: list = field(default_factory=list)  # (layer, cycle its phase ends)
    wb_rows: list = field(default_factory=list)  # static weights
    requant: list = field(default_factory=list)
    residual: list = field(default_factory=list)
    conv1_ab_base: int = 0
    output_acc_row: int = 0
    output_set: int = 0

    def conv1_rows(self, low_conv1):
        return low_conv1.ab_rows


def build(m, p, softmax_ops, div_latency):
    """m: BraggNN integer model (weights / scales), p: params.json namespace.
    softmax_ops: {"Max": .., "Sum": .., "Divide": .., "Out": ..}."""
    R, C = p.rows, p.cols
    assert R == C == 16, "the lowering assumes a 16 x 16 array"
    commit = p.acc_read_latency + p.acc_write_latency
    ab_ready = p.ab_write_latency + 1  # a row written at t is readable from t + this
    wb_ready = p.wb_write_latency + 1
    wb, ab = Allocator("WeightBuffer", p.wb_rows), Allocator("ActivationBuffer", p.ab_rows)
    sets = Allocator("Requantizer sets", p.requant_param_sets)
    table = Table()
    inf = Inference(table, 0)

    def mult_shift(scale):
        return scale_to_mult_shift(scale)

    def add_set(scale, bias=None, relu=False):
        s = sets.take(1)
        mult, shift = mult_shift(scale)
        inf.requant.append((s, pad(bias if bias is not None else []), [mult] * C, [shift] * C, 0 if relu else -128, 127))
        return s

    def conv_passes(low, now):
        """Schedule a lowered conv's passes (its weights become static rows)."""
        inf.wb_rows.extend(low.wb_rows)
        return schedule_passes(table, low.passes, p, start=now)

    def drain_to_ab(acc_rows, set_of, ab_rows, now):
        """Drain Accumulator rows through the Requantizer into ActivationBuffer."""
        items = [(a, {"requant.paramSet": set_of(k), "abWrite.en": 1, "abWrite.addr": b, "abWrite.fromRequantizer": 1})
                 for k, (a, b) in enumerate(zip(acc_rows, ab_rows))]
        return schedule_drain(table, now, items, p)

    def lower(kind, x_shape, w, b, scale, relu, ab_base, acc_base):
        """Lower a conv whose input already sits on chip at ab_base."""
        x = np.zeros(x_shape, dtype=np.int64)
        k = w.shape[0]
        out = np.zeros((x_shape[0] - k + 1, x_shape[1] - k + 1, w.shape[3]), dtype=np.int64)
        set_base = sets.next
        fn = lower_conv_im2col if kind == "im2col" else lower_conv
        low = fn(x, w, b, scale, relu, out, R, C, ab_base=ab_base, wb_base=wb.take(0), acc_base=acc_base,
                 set_base=set_base)
        wb.take(len(low.wb_rows))
        for s in low.requant:
            sets.take(1)
            inf.requant.append(s)
        return low

    def conv_layer(kind, x_shape, w, b, scale, relu, ab_in, acc_base, now):
        """Conv from ActivationBuffer ab_in to a fresh ActivationBuffer region; returns (ab_out, low, now)."""
        low = lower(kind, x_shape, w, b, scale, relu, ab_in, acc_base)
        now = conv_passes(low, now) + commit
        acc_rows = [o[0] for o in low.outputs]
        ab_out = ab.take(len(acc_rows))
        # Accumulator row cot*npix + pix -> ActivationBuffer row ab_out + cot*npix + pix
        sets_of = [o[1] for o in low.outputs]
        now = drain_to_ab(acc_rows, lambda k: sets_of[k], [ab_out + r - acc_base for r in acc_rows], now)
        return ab_out, low, now - 1 + ab_ready

    # ---- conv1 (im2col; the host writes its input rows per patch) ----------------
    inf.conv1_ab_base = ab.take(N)
    ab_c1, low_c1, now = conv_layer("im2col", (11, 11, 1), m.w["conv1"], m.b["conv1"], m.scale["conv1"], False,
                                    inf.conv1_ab_base, 0, 0)
    inf.marks.append(("conv1", now))
    inf.low_conv1 = low_c1

    # ---- NLB: theta, phi -> phi^T, logits, softmax, g, attention . g, out conv ----
    ab_th, _, now = conv_layer("kn2row", (9, 9, 64), m.w["theta"], m.b["theta"], m.scale["theta"], False,
                               ab_c1, 324, now)
    inf.marks.append(("theta", now))

    low_phi = lower("kn2row", (9, 9, 64), m.w["phi"], m.b["phi"], m.scale["phi"], False, ab_c1, 486)
    now = conv_passes(low_phi, now) + commit
    phi_sets = [s[0] for s in low_phi.requant]
    wb_pt = wb.take(12 * 16)  # tile (jt, ct) = phi^T rows for channel tile ct, pixel tile jt
    last = None
    for jt in range(6):
        for ct in range(2):
            items = [(486 + ct * N + j, {"requant.paramSet": phi_sets[ct], "transposer.capture": 1, "transposer.row": r})
                     for r, j in enumerate(range(jt * 16, min(N, (jt + 1) * 16)))]
            emit = schedule_drain(table, now, items, p)
            for c in range(16):
                table.put(emit + c, {"wbWrite.en": 1, "wbWrite.addr": wb_pt + (jt * 2 + ct) * 16 + c,
                                     "wbWrite.transposed": 1, "wbWrite.column": c})
            last = emit + 15
            now = emit + 16 - p.acc_read_latency
    now = last + wb_ready
    inf.marks.append(("phi + phi^T", now))

    logit_set = add_set(m.scale["theta_phi"])
    passes = [{"ab": [ab_th + ct * N + i for i in range(N)], "wb": wb_pt + (jt * 2 + ct) * 16,
               "acc": [jt * N + i for i in range(N)], "accumulate": ct > 0} for jt in range(6) for ct in range(2)]
    now = schedule_passes(table, passes, p, start=now) + commit
    inf.marks.append(("theta.phi^T", now))

    ab_att = ab.take(6 * N)
    for a in range(0, N, 2):
        pair = [(row, stat) for stat, row in enumerate((a, a + 1)) if row < N]

        def beats(row, stat, op, extra=None):
            return [(jt * N + row, {"requant.paramSet": logit_set, "softmax.op": op, "softmax.first": int(jt == 0),
                                    "softmax.lanes": min(16, N - jt * 16), "softmax.stat": stat,
                                    **(extra(jt) if extra else {})}) for jt in range(6)]
        items = []
        for row, stat in pair:
            items += beats(row, stat, softmax_ops["Max"]) + beats(row, stat, softmax_ops["Sum"])
        divide = schedule_drain(table, now, items, p)
        for k, (row, stat) in enumerate(pair):
            table.put(divide + k, {"softmax.op": softmax_ops["Divide"], "softmax.stat": stat})
        out_items = []
        for row, stat in pair:
            out_items += beats(row, stat, softmax_ops["Out"], lambda jt, row=row: {
                "abWrite.en": 1, "abWrite.addr": ab_att + jt * N + row, "abWrite.fromRequantizer": 1})
        now = schedule_drain(table, divide + len(pair) - 1 + div_latency - p.acc_read_latency, out_items, p) \
            - p.acc_read_latency
    now = now + p.acc_read_latency - 1 + ab_ready
    inf.marks.append(("softmax", now))

    low_g = lower("kn2row", (9, 9, 64), m.w["g"], m.b["g"], m.scale["g"], False, ab_c1, 486)
    now = conv_passes(low_g, now) + commit
    g_sets = [s[0] for s in low_g.requant]
    wb_g = wb.take(12 * 16)  # tile (cot, jt), row = pixel within jt
    items = [(486 + cot * N + j, {"requant.paramSet": g_sets[cot], "wbWrite.en": 1, "wbWrite.fromRequantizer": 1,
                                  "wbWrite.addr": wb_g + (cot * 6 + j // 16) * 16 + j % 16})
             for cot in range(2) for j in range(N)]
    now = schedule_drain(table, now, items, p) - 1 + wb_ready
    inf.marks.append(("g -> WB", now))

    ag_set = add_set(m.scale["attention_g"])
    passes = [{"ab": [ab_att + jt * N + i for i in range(N)], "wb": wb_g + (cot * 6 + jt) * 16,
               "acc": [648 + cot * N + i for i in range(N)], "accumulate": jt > 0} for cot in range(2) for jt in range(6)]
    now = schedule_passes(table, passes, p, start=now) + commit
    ab_ag = ab.take(2 * N)
    now = drain_to_ab([648 + k for k in range(2 * N)], lambda k: ag_set, [ab_ag + k for k in range(2 * N)], now) \
        - 1 + ab_ready
    inf.marks.append(("attention.g", now))

    ab_no, _, now = conv_layer("kn2row", (9, 9, 32), m.w["nlb_out"], m.b["nlb_out"], m.scale["nlb_out"], False,
                               ab_ag, 0, now)
    inf.marks.append(("nlb_out", now))

    # ---- resadd: a = conv1 output, b = NLB output ---------------------------------
    ma, sa = mult_shift(m.scale["resadd_a"])
    mb, sb = mult_shift(m.scale["resadd_b"])
    inf.residual = [(0, [0] * C, [ma] * C, [sa] * C, -128, 127), (1, [0] * C, [mb] * C, [sb] * C, -128, 127)]
    k = 0
    for ct in range(4):
        for base, pset in ((ab_c1, 0), (ab_no, 1)):
            for i in range(N):
                issue = now + k * N + i
                table.put(issue, {"abRead.en": 1, "abRead.addr": base + ct * N + i})
                table.put(issue + p.ab_read_latency, {"residual.en": 1, "residual.paramSet": pset,
                                                      "accWrite.addr": 324 + ct * N + i, "accWrite.accumulate": pset})
            k += 1
    now = now + k * N + p.ab_read_latency + commit
    rs_set = add_set(m.scale["resadd_c"], relu=True)
    ab_rs = ab.take(4 * N)
    now = drain_to_ab([324 + k for k in range(4 * N)], lambda k: rs_set, [ab_rs + k for k in range(4 * N)], now) \
        - 1 + ab_ready
    inf.marks.append(("resadd", now))

    # ---- conv2, conv3, fc1..output ---------------------------------------------------
    ab_c2, _, now = conv_layer("kn2row", (9, 9, 64), m.w["conv2"], m.b["conv2"], m.scale["conv2"], True, ab_rs, 0, now)
    inf.marks.append(("conv2", now))
    ab_c3, _, now = conv_layer("kn2row", (7, 7, 32), m.w["conv3"], m.b["conv3"], m.scale["conv3"], True, ab_c2, 98, now)
    inf.marks.append(("conv3", now))
    ab_x, _, now = conv_layer("kn2row", (5, 5, 8), fc_as_conv(m.w["fc1"].reshape(16, -1), 5, 5), m.b["fc1"],
                              m.scale["fc1"], True, ab_c3, 123, now)
    inf.marks.append(("fc1", now))
    for acc_row, name in zip((124, 125, 126), ("fc2", "fc3", "fc4")):
        w = fc_as_conv(m.w[name], 1, 1)
        ab_x, _, now = conv_layer("kn2row", (1, 1, w.shape[2]), w, m.b[name], m.scale[name], True, ab_x, acc_row, now)
        inf.marks.append(("fc2..fc4", now))
    w = fc_as_conv(m.w["output"], 1, 1)
    low_out = lower("kn2row", (1, 1, 2), w, m.b["output"], m.scale["output"], False, ab_x, 127)
    inf.end = conv_passes(low_out, now)
    inf.marks.append(("output", inf.end))
    inf.output_acc_row, inf.output_set = 127, low_out.requant[0][0]
    table.pad(inf.end)
    inf.usage = {"wb_rows": wb.next, "ab_rows": ab.next, "requant_sets": sets.next}
    return inf
