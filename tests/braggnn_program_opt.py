"""The whole BraggNN inference as one Xccel program, list-scheduled.

Same lowering and on-chip data flow as braggnn_program.py, but instead of
running phases one after another, every operation (a pass, one drained
Accumulator row, a softmax Divide, a Transposer column) is placed at the
earliest cycle where its data is ready and the resources it needs are free,
in program order. Independent work therefore overlaps: theta / phi / g,
softmax with g, drains with the next layer's first passes, several softmax
rows in flight.

Tracked per cycle: every microcode slot / port (one use per cycle), and the
Accumulator read port, which accumulating beats (read-modify-write) and
drains may have to share, depending on `acc_read`:
  "shared"  one read port (the current OneR1W Accumulator)
  "banked"  two independent 1R1W banks (rows < half vs. >= half); an
            accumulating beat blocks drains of its own bank only
  "dual"    a separate drain port (2R1W), no conflict (upper bound)
With a SystolicArrayN compute unit (`arrays` > 1), 3x3 convs run one kernel
row (3 arrays, lower_conv_rows) or the whole kernel (9 arrays with wide weight
beats, lower_conv_window) per pass, and every other pass loads the last,
undelayed array only. With one array row per weight beat the scheduler keeps,
per weight slot, which rows of the other arrays may hold nonzero weights and
zeroes those a pass's operand lanes would meet (from one all-zero
WeightBuffer row); with wide beats a 1x1 row is stored zero-padded in the
first C words of a WeightBuffer row and rotated (wbRead.shiftAmt) onto the
last array, which zeroes the others' row in the same load. It keeps the
cycles within the arrays' longest delay before such a pass free of kernel
beats, whose products would otherwise be summed into its first results.
Discarded results ("acc" None) go to one scratch Accumulator row; idle beats
of a kernel pass ("ab" None) write nothing.

Tracked per row: when an ActivationBuffer / WeightBuffer row becomes readable
(write + writeLatency + 1), when an Accumulator row can be read or
accumulated into (last write + commitLatency), and its last drain read
(a later overwrite of a reused region must come after it).
"""

import numpy as np

from braggnn_model import scale_to_mult_shift
from braggnn_program import Allocator, Inference, pad
from braggnn_xccel import fc_as_conv, lower_conv, lower_conv_im2col, lower_conv_rows, lower_conv_window
from xccel_program import Table

N = 81


class ListScheduler:
    def __init__(self, p, acc_read="shared", softmax_stats=2, div_latency=33, partial_weights=False, tight_slots=False,
                 arrays=1, zero_row=None, junk_row=None, kernel_cols=0, row_pitch=0, wide=False, tree_latency=0):
        self.p = p
        # tree_latency > 0 models a reduction-tree compute unit (not built): every
        # weight is used when an operand arrives (no skew), results tree_latency later
        self.tree = tree_latency > 0
        self.arrays = arrays
        self.wide = arrays > 1 and wide
        kc = kernel_cols or arrays
        # the arrays' longest result delay (SystolicArrayN.maxDelay)
        self.max_delay = (arrays // kc - 1) * row_pitch + kc - 1
        # wide beats (WeightBuffer rows of arrays * C words): rotating by C brings a
        # row's first C words onto the last array (port j gets word (j + C) mod row)
        if self.wide:
            assert p.cols * p.wb_words_per_bank == arrays * p.cols
        self.wide_shift = p.cols if self.wide else 0
        self.zero_row, self.junk_row = zero_row, junk_row  # an all-zero WeightBuffer row, a scratch Accumulator row
        # per slot and array (other than the last): rows that may hold nonzero weights
        self.dirty = [[set(range(p.rows)) for _ in range(arrays - 1)] for _ in range(2)]
        self.partial_weights = partial_weights  # load only a tile's valid rows ("wrows")
        self.tight_slots = tight_slots  # reuse a weight slot as soon as its last operand has left the array
        self.acc_read = acc_read
        self.table = Table()
        self.busy = {}  # (resource, cycle) -> True
        self.ab_ready, self.wb_ready = {}, {}
        self.acc_written, self.acc_read_last = {}, {}
        self.commit = p.acc_read_latency + p.acc_write_latency
        self.latency = tree_latency if self.tree else p.rows + p.cols - 1
        self.slot_free = [0, 0]
        self.softmax_stats = softmax_stats
        self.stat_free = [0] * softmax_stats
        self.div_latency = div_latency
        self.transposer_free = 0
        self.end = 0
        self.layer = ""  # the layer operations are attributed to (for reporting)
        self.trace = []  # (layer, kind, first cycle, last cycle)

    # -- resources ---------------------------------------------------------------
    def acc_port(self, row):
        """The Accumulator read-port resource an access to `row` uses."""
        if self.acc_read == "dual":
            return None
        if self.acc_read == "banked":
            return f"accport{int(row >= self.p.acc_rows // 2)}"
        return "accport"

    # "rowbeat" (a kernel-row beat reaching the arrays) and "guard" (a cycle
    # before another pass's first beat) exclude each other, not themselves
    EXCLUDES = {"rowbeat": "guard", "guard": "rowbeat"}

    def free(self, uses):
        for r, t in uses:
            if r in self.EXCLUDES:
                if (self.EXCLUDES[r], t) in self.busy:
                    return False
            elif r is not None and (r, t) in self.busy:
                return False
        return True

    def claim(self, uses):
        for r, t in uses:
            if r in self.EXCLUDES:
                assert (self.EXCLUDES[r], t) not in self.busy, (r, t)
                self.busy[(r, t)] = True
            elif r is not None:
                assert (r, t) not in self.busy, (r, t)
                self.busy[(r, t)] = True

    def fit(self, lower, uses_at):
        """Earliest t >= lower whose uses_at(t) are all free."""
        t = max(0, lower)
        while not self.free(uses_at(t)):
            t += 1
        return t

    # -- a weight-stationary pass ------------------------------------------------------
    def schedule_pass(self, ps):
        p = self.p
        R, stream = p.rows, len(ps["ab"])
        nload = ps.get("wrows", R) if self.partial_weights else R
        slot = 0 if self.slot_free[0] <= self.slot_free[1] else 1
        rows_pass = "loads" in ps
        # (WeightBuffer row, weight port address, wbRead.shiftAmt): the pass's own
        # weights, then zeros for other arrays' rows its operand lanes (< wrows) would meet
        if rows_pass:
            loads = [(row, a, 0) for row, a in ps["loads"]]
        elif self.wide:
            loads = [(ps["wb"] + r, r, self.wide_shift) for r in range(nload)]
        else:
            last = (self.arrays - 1) * R
            loads = [(ps["wb"] + r, last + r, 0) for r in range(nload)]
        if not self.wide:
            loaded = {a // R for _, a, _ in loads}
            wr = ps.get("wrows", R)
            loads += [(self.zero_row, k * R + r, 0) for k in range(self.arrays - 1) if k not in loaded
                      for r in sorted(self.dirty[slot][k]) if r < wr]
        beats = [i for i, row in enumerate(ps["ab"]) if row is not None]
        # offsets relative to the weight load start w: load j read at w + j, on the weight
        # port at w + wbRL + j, stream from S = w + wbRL + 1 - abRL (+ the lag of the
        # latest-loaded row behind its PE row, so every weight is in before its operand)
        if self.tree:  # every row before the first operand
            lag = max(0, len(loads) - 1)
        else:
            lag = max([0] + [j - a % R for j, (_, a, _) in enumerate(loads)])
        s_off = p.wb_read_latency + 1 - p.ab_read_latency + lag

        lower = self.slot_free[slot] if self.tight_slots else self.slot_free[slot] - p.wb_read_latency
        for j, (row, _, _) in enumerate(loads):
            lower = max(lower, self.wb_ready.get(row, 0) - j)
        for i in beats:
            lower = max(lower, self.ab_ready.get(ps["ab"][i], 0) - s_off - i)
        for i, row in enumerate(ps["acc"]):
            if row is None:
                continue
            c = s_off + p.ab_read_latency + i + self.latency  # result cycle offset
            if ps["accumulate"]:
                lower = max(lower, self.acc_written.get(row, -10**9) + self.commit - c)
            lower = max(lower, self.acc_read_last.get(row, -1) + 1 - c,
                        self.acc_written.get(row, -10**9) + 1 - c)

        def uses(w):
            u = [("wbRead", w + j) for j in range(len(loads))] + \
                [("weightPort", w + p.wb_read_latency + j) for j in range(len(loads))]
            S = w + s_off
            u += [("abRead", S + i) for i in beats]
            A = S + p.ab_read_latency  # first arrival
            if self.arrays > 1:
                if rows_pass:
                    u += [("rowbeat", A + i) for i in beats]
                else:
                    u += [("guard", A - d) for d in range(1, self.max_delay + 1)]
            for i in beats:
                row = ps["acc"][i]
                c = A + i + self.latency
                u.append(("accWrite", c))
                if ps["accumulate"] and row is not None:
                    u.append((self.acc_port(row), c))
            return u

        w = self.fit(lower, uses)
        self.claim(uses(w))
        S = w + s_off
        tb = self.table
        for j, (row, addr, shift) in enumerate(loads):
            tb.put(w + j, {"wbRead.en": 1, "wbRead.addr": row, **({"wbRead.shiftAmt": shift} if shift else {})})
            tb.put(w + p.wb_read_latency + j, {"weightPort.en": 1, "weightPort.addr": addr, "weightPort.slot": slot})
            k = addr // R
            if not self.wide and k < self.arrays - 1:
                (self.dirty[slot][k].discard if row == self.zero_row else self.dirty[slot][k].add)(addr % R)
        for i in beats:
            tb.put(S + i, {"abRead.en": 1, "abRead.addr": ps["ab"][i]})
            arrive = S + p.ab_read_latency + i
            if slot:
                tb.put(arrive, {"compute.weightSelect": 1})
            c = arrive + self.latency
            row = ps["acc"][i]
            if row is None:  # a discarded result
                tb.put(c, {"accWrite.addr": self.junk_row, "accWrite.accumulate": 0})
            else:
                tb.put(c, {"accWrite.addr": row, "accWrite.accumulate": int(ps["accumulate"])})
                self.acc_written[row] = c
            self.end = max(self.end, c + 1)
        last_arrival = S + p.ab_read_latency + stream - 1
        self.trace.append((self.layer, "weights", w, w + p.wb_read_latency + len(loads) - 1))
        self.trace.append((self.layer, "stream", S, S + stream - 1))
        self.trace.append((self.layer, "results", S + p.ab_read_latency + self.latency,
                           S + p.ab_read_latency + self.latency + stream - 1))
        if self.tight_slots:
            # PE (r, c) uses its weight at arrival + r + c; a new row r written at
            # w' + wbRL + r takes effect the cycle after, so the slot's next
            # load may start at w' = last_arrival + C - 1 - wbRL
            # a tree multiplies one cycle after the operand arrives (input register)
            skew = 1 if self.tree else p.cols - 1
            self.slot_free[slot] = last_arrival + skew - p.wb_read_latency
        else:
            self.slot_free[slot] = last_arrival + 1 + R + p.cols
        return self.end

    # -- one drained Accumulator row ---------------------------------------------------
    def schedule_drain(self, row, at_arrival, lower=0, resources=(), writes_ab=None, writes_wb=None, post=True):
        """Read `row` and put `at_arrival` fields accumulator.readLatency later.
        resources: extra resources used at the arrival cycle. post: the beat's
        Softmax output ("post") is consumed (False for a row-buffered Max beat,
        which leaves post to a buffered Out beat of the same cycle)."""
        p = self.p
        rl = p.acc_read_latency
        lower = max(lower, self.acc_written.get(row, -10**9) + self.commit)

        def uses(t):
            u = [("accRead", t), (self.acc_port(row), t), ("requant", t + rl)]
            if post:
                u.append(("post", t + rl))
            u += [(r, t + rl) for r in resources]
            if writes_ab is not None:
                u.append(("abWrite", t + rl))
            if writes_wb is not None:
                u.append(("wbWrite", t + rl))
            return u

        t = self.fit(lower, uses)
        self.claim(uses(t))
        self.table.put(t, {"accRead.en": 1, "accRead.addr": row})
        self.table.put(t + rl, at_arrival)
        self.acc_read_last[row] = max(self.acc_read_last.get(row, -1), t)
        self.trace.append((self.layer, "drain", t, t + rl))
        if writes_ab is not None:
            self.ab_ready[writes_ab] = t + rl + p.ab_write_latency + 1
        if writes_wb is not None:
            self.wb_ready[writes_wb] = t + rl + p.wb_write_latency + 1
        self.end = max(self.end, t + rl + 1)
        return t + rl  # arrival cycle


def build_opt(m, p, softmax_ops, div_latency, acc_read="shared", softmax_stats=2,
              partial_weights=True, tile_banks=(), arrays=1, sum_ops=None, tree_latency=0):
    """partial_weights: load only each weight tile's valid rows and reuse a
    weight slot as soon as its last operand has left the array (10,904 ->
    10,616 cycles with two row banks and four softmax stats).
    tile_banks: layers ("conv2", "nlb_out", or True for both) whose C_out
    tiles alternate between the two Accumulator banks, so a finished tile
    drains while the next one accumulates. Off by default: the next layer
    cannot start before the array finishes this one anyway, and the moved
    drains cost the greedy placement a few cycles (+8 .. +32).
    arrays: SystolicArrays in the compute unit (compute_arrays); with 3,
    conv2 and conv3 run one kernel row per pass (lower_conv_rows), with 9 in a
    3 x 3 window and wide weight beats the whole kernel (lower_conv_window)."""
    R, C = p.rows, p.cols
    assert R == C, "the Transposer, residual and feedback paths assume a square array"
    CH = 32  # theta / phi / g / attention . g channels
    njt = -(-N // C)  # pixel blocks of C (logit columns, attention . g input tiles)
    nct = -(-CH // R)  # channel tiles of theta / phi / g
    nrs = -(-64 // R)  # channel tiles of conv1 / nlb_out / resadd
    wb, ab = Allocator("WeightBuffer", p.wb_rows), Allocator("ActivationBuffer", p.ab_rows)
    sets = Allocator("Requantizer sets", p.requant_param_sets)
    half = p.acc_rows // 2  # Accumulator bank boundary (used by "banked")
    kernel_cols = getattr(p, "compute_kernel_cols", 0) or arrays
    pitch = getattr(p, "compute_row_pitch", 0)
    wide = arrays > 1 and getattr(p, "compute_wide_weights", False)
    zero_row = wb.take(1) if arrays > 1 and not wide else None
    sch = ListScheduler(p, acc_read, softmax_stats, div_latency, partial_weights, partial_weights,
                        arrays, zero_row, junk_row=half - 1,  # bank 0's last row is unused
                        kernel_cols=kernel_cols, row_pitch=pitch, wide=wide, tree_latency=tree_latency)
    tb = sch.table
    inf = Inference(tb, 0)
    if zero_row is not None:
        inf.wb_rows.append((zero_row, [0] * C))

    def add_set(scale, bias=None, relu=False):
        s = sets.take(1)
        mult, shift = scale_to_mult_shift(scale)
        inf.requant.append((s, pad(bias if bias is not None else [], C), [mult] * C, [shift] * C, 0 if relu else -128, 127))
        return s

    def lower(kind, x_shape, w, b, scale, relu, ab_base, acc_base, acc_bases=None):
        x = np.zeros(x_shape, dtype=np.int64)
        k = w.shape[0]
        out = np.zeros((x_shape[0] - k + 1, x_shape[1] - k + 1, w.shape[3]), dtype=np.int64)
        if kind == "window":
            low = lower_conv_window(x, w, b, scale, relu, out, R, C, pitch, ab_base=ab_base, wb_base=wb.take(0),
                                    acc_base=acc_base, set_base=sets.next, acc_bases=acc_bases)
        elif kind == "rows":
            low = lower_conv_rows(x, w, b, scale, relu, out, R, C, ab_base=ab_base, wb_base=wb.take(0),
                                  acc_base=acc_base, set_base=sets.next, acc_bases=acc_bases)
        elif kind == "im2col":
            low = lower_conv_im2col(x, w, b, scale, relu, out, R, C, ab_base=ab_base, wb_base=wb.take(0),
                                    acc_base=acc_base, set_base=sets.next)
        else:
            low = lower_conv(x, w, b, scale, relu, out, R, C, ab_base=ab_base, wb_base=wb.take(0), acc_base=acc_base,
                             set_base=sets.next, acc_bases=acc_bases)
        wb.take(len(low.wb_rows))
        for s in low.requant:
            sets.take(1)
            inf.requant.append(s)
        inf.wb_rows.extend(low.wb_rows)
        return low

    def run_passes(passes):
        for ps in passes:
            sch.schedule_pass(ps)

    def drain_to_ab(acc_rows, set_of, ab_rows):
        for k, (a, b) in enumerate(zip(acc_rows, ab_rows)):
            sch.schedule_drain(a, {"requant.paramSet": set_of(k), "abWrite.en": 1, "abWrite.addr": b,
                                   "abWrite.fromRequantizer": 1}, writes_ab=b)

    def conv_layer(kind, x_shape, w, b, scale, relu, ab_in, acc_base, acc_bases=None):
        low = lower(kind, x_shape, w, b, scale, relu, ab_in, acc_base, acc_bases)
        run_passes(low.passes)
        acc_rows = [o[0] for o in low.outputs]
        sets_of = [o[1] for o in low.outputs]
        ab_out = ab.take(len(acc_rows))
        # outputs are listed C_out tile by tile, pixel by pixel: the next layer's layout
        drain_to_ab(acc_rows, lambda k: sets_of[k], [ab_out + k for k in range(len(acc_rows))])
        return ab_out

    # Accumulator regions alternate between the two halves (banks) so a layer's
    # drain and the next layer's accumulation use different banks
    B0, B1 = 0, half
    inf.conv1_ab_base = ab.take(N)
    sch.layer = "conv1"
    ab_c1 = conv_layer("im2col", (11, 11, 1), m.w["conv1"], m.b["conv1"], m.scale["conv1"], False,
                       inf.conv1_ab_base, B0)                                      # B0 0..323
    sch.layer = "theta"
    ab_th = conv_layer("kn2row", (9, 9, 64), m.w["theta"], m.b["theta"], m.scale["theta"], False,
                       ab_c1, B1)                                                  # B1 0..161

    sch.layer = "phi + phi^T"
    PHI = B1 + 2 * N
    low_phi = lower("kn2row", (9, 9, 64), m.w["phi"], m.b["phi"], m.scale["phi"], False, ab_c1, PHI)
    run_passes(low_phi.passes)                                                     # B1 162..323
    phi_sets = [s[0] for s in low_phi.requant]
    wb_pt = wb.take(njt * nct * R)
    for jt in range(njt):
        for ct in range(nct):
            pixels = list(range(jt * C, min(N, (jt + 1) * C)))
            last = 0
            for r, j in enumerate(pixels):
                last = sch.schedule_drain(PHI + ct * N + j, {"requant.paramSet": phi_sets[ct], "transposer.capture": 1,
                                                             "transposer.row": r},
                                          lower=sch.transposer_free - p.acc_read_latency, resources=("transposer",))
            t = last + 1
            for c in range(min(R, CH - ct * R)):  # the tile's channels
                t = sch.fit(t, lambda x: [("wbWrite", x)])
                sch.claim([("wbWrite", t)])
                row = wb_pt + (jt * nct + ct) * R + c
                tb.put(t, {"wbWrite.en": 1, "wbWrite.addr": row, "wbWrite.transposed": 1, "wbWrite.column": c})
                sch.wb_ready[row] = t + p.wb_write_latency + 1
                t += 1
            sch.transposer_free = t  # the next block's first capture arrives after the last column read

    sch.layer = "theta.phi^T"
    logit_set = add_set(m.scale["theta_phi"])
    LOG = B0  # reuses conv1's region (drained)
    run_passes([{"ab": [ab_th + ct * N + i for i in range(N)], "wb": wb_pt + (jt * nct + ct) * R,
                 "acc": [LOG + jt * N + i for i in range(N)], "accumulate": ct > 0, "wrows": min(R, CH - ct * R)}
                for jt in range(njt) for ct in range(nct)])                           # B0 0..485

    sch.layer = "softmax"
    ab_att = ab.take(njt * N)
    ops = softmax_ops
    row_beats = getattr(p, "softmax_row_beats", 0)
    rl = p.acc_read_latency
    for row in (range(N) if row_beats else ()):
        # row buffer: one drain per beat (Max, kept in the buffer); the sum and
        # out stages read the buffer through their own fields, so they overlap
        # other rows' drains
        assert row_beats >= njt
        stat = row % softmax_stats
        lanes = [min(C, N - jt * C) for jt in range(njt)]
        last = sch.stat_free[stat] - 1
        for jt in range(njt):
            last = sch.schedule_drain(LOG + jt * N + row, {
                "requant.paramSet": logit_set, "softmax.op": ops["Max"], "softmax.first": int(jt == 0),
                "softmax.lanes": lanes[jt], "softmax.stat": stat, "softmax.buf.store": 1, "softmax.buf.beat": jt},
                lower=last + 1 - rl, resources=("softmax",), post=False)
        t = last + 1  # the row's max and last beat are in
        for jt in range(njt + 1):  # Sum beats, then Divide
            t = sch.fit(t, lambda x: [("softmaxSum", x)])
            sch.claim([("softmaxSum", t)])
            if jt < njt:
                tb.put(t, {"softmax.buf.sumOp": sum_ops["Sum"], "softmax.buf.sumStat": stat, "softmax.buf.sumBeat": jt,
                           "softmax.buf.sumFirst": int(jt == 0), "softmax.buf.sumLanes": lanes[jt]})
            else:
                tb.put(t, {"softmax.buf.sumOp": sum_ops["Divide"], "softmax.buf.sumStat": stat})
            t += 1
        t += div_latency - 1
        for jt in range(njt):
            dst = ab_att + jt * N + row
            t = sch.fit(t, lambda x: [("softmaxOut", x), ("post", x), ("abWrite", x)])
            sch.claim([("softmaxOut", t), ("post", t), ("abWrite", t)])
            tb.put(t, {"softmax.buf.outEn": 1, "softmax.buf.outStat": stat, "softmax.buf.outBeat": jt,
                       "softmax.buf.outLanes": lanes[jt], "abWrite.en": 1, "abWrite.addr": dst,
                       "abWrite.fromRequantizer": 1})
            sch.ab_ready[dst] = t + p.ab_write_latency + 1
            sch.trace.append((sch.layer, "out", t, t))
            sch.end = max(sch.end, t + 1)
            t += 1
        sch.stat_free[stat] = t
    for row in (() if row_beats else range(N)):
        stat = row % softmax_stats
        lanes = [min(C, N - jt * C) for jt in range(njt)]

        def beat(op, jt, extra=None):
            return {"requant.paramSet": logit_set, "softmax.op": op, "softmax.first": int(jt == 0),
                    "softmax.lanes": lanes[jt], "softmax.stat": stat, **(extra or {})}
        # Max beats: the stat slot must be free (its previous row fully out)
        last = sch.stat_free[stat] - 1
        for jt in range(njt):
            last = sch.schedule_drain(LOG + jt * N + row, beat(ops["Max"], jt), lower=last + 1 - p.acc_read_latency,
                                      resources=("softmax",))
        for jt in range(njt):  # Sum beats after every Max beat has arrived
            last = sch.schedule_drain(LOG + jt * N + row, beat(ops["Sum"], jt), lower=last + 1 - p.acc_read_latency,
                                      resources=("softmax",))
        d = sch.fit(last + 1, lambda x: [("softmax", x)])
        sch.claim([("softmax", d)])
        tb.put(d, {"softmax.op": ops["Divide"], "softmax.stat": stat})
        last = d + div_latency - 1
        for jt in range(njt):
            dst = ab_att + jt * N + row
            last = sch.schedule_drain(LOG + jt * N + row, beat(ops["Out"], jt, {
                "abWrite.en": 1, "abWrite.addr": dst, "abWrite.fromRequantizer": 1}),
                lower=last + 1 - p.acc_read_latency, resources=("softmax",), writes_ab=dst)
        sch.stat_free[stat] = last + 1

    sch.layer = "g -> WB"
    G = B1 + 4 * N  # B1 324..485
    low_g = lower("kn2row", (9, 9, 64), m.w["g"], m.b["g"], m.scale["g"], False, ab_c1, G)
    run_passes(low_g.passes)
    g_sets = [s[0] for s in low_g.requant]
    wb_g = wb.take(nct * njt * R)
    for cot in range(nct):
        for j in range(N):
            dst = wb_g + (cot * njt + j // R) * R + j % R
            sch.schedule_drain(G + cot * N + j, {"requant.paramSet": g_sets[cot], "wbWrite.en": 1,
                                                  "wbWrite.fromRequantizer": 1, "wbWrite.addr": dst}, writes_wb=dst)

    sch.layer = "attention.g"
    ag_set = add_set(m.scale["attention_g"])
    AG = B1  # reuses theta's region (drained)
    # the last pixel tile has one valid row (pixel 80); attention is 0 on the other lanes
    run_passes([{"ab": [ab_att + jt * N + i for i in range(N)], "wb": wb_g + (cot * njt + jt) * R,
                 "acc": [AG + cot * N + i for i in range(N)], "accumulate": jt > 0, "wrows": min(R, N - jt * R)}
                for cot in range(nct) for jt in range(njt)])
    ab_ag = ab.take(nct * N)
    drain_to_ab([AG + k for k in range(nct * N)], lambda k: ag_set, [ab_ag + k for k in range(nct * N)])

    sch.layer = "nlb_out"
    NO = B0  # reuses the logits' region (drained)
    # with tile_banks, tiles 1 and 3 go to bank 1, reusing attention . g's region (drained)
    no_bases = [NO, AG, NO + N, AG + N] if tile_banks is True or "nlb_out" in (tile_banks or ()) else None
    ab_no = conv_layer("kn2row", (9, 9, 32), m.w["nlb_out"], m.b["nlb_out"], m.scale["nlb_out"], False, ab_ag, NO,
                       no_bases)

    sch.layer = "resadd"
    ma, sa = scale_to_mult_shift(m.scale["resadd_a"])
    mb, sb = scale_to_mult_shift(m.scale["resadd_b"])
    inf.residual = [(0, [0] * C, [ma] * C, [sa] * C, -128, 127), (1, [0] * C, [mb] * C, [sb] * C, -128, 127)]
    RS = B1 + 2 * N  # B1 162..485 (phi / g regions, drained)
    for ct in range(nrs):
        for base, pset in ((ab_c1, 0), (ab_no, 1)):
            t = 0
            for i in range(N):
                src, dst = base + ct * N + i, RS + ct * N + i
                lower_t = max(t, sch.ab_ready.get(src, 0),
                              sch.acc_read_last.get(dst, -1) + 1 - p.ab_read_latency,
                              sch.acc_written.get(dst, -10**9) + (sch.commit if pset else 1) - p.ab_read_latency)
                rl = p.ab_read_latency

                def uses(x, dst=dst, pset=pset):
                    u = [("abRead", x), ("accWrite", x + rl), ("residual", x + rl)]
                    if pset:
                        u.append((sch.acc_port(dst), x + rl))
                    return u
                x = sch.fit(lower_t, uses)
                sch.claim(uses(x))
                tb.put(x, {"abRead.en": 1, "abRead.addr": src})
                tb.put(x + rl, {"residual.en": 1, "residual.paramSet": pset, "accWrite.addr": dst,
                                "accWrite.accumulate": pset})
                sch.acc_written[dst] = x + rl
                sch.trace.append((sch.layer, "stream", x, x))
                sch.end = max(sch.end, x + rl + 1)
                t = x + 1
    rs_set = add_set(m.scale["resadd_c"], relu=True)
    ab_rs = ab.take(nrs * N)
    drain_to_ab([RS + k for k in range(nrs * N)], lambda k: rs_set, [ab_rs + k for k in range(nrs * N)])

    # with tile_banks, tile 1 goes to bank 1 (theta / attention . g's region, drained)
    sch.layer = "conv2"
    c2_bases = [B0 + 324, B1] if tile_banks is True or "conv2" in (tile_banks or ()) else None
    if arrays == 9 and kernel_cols == 3 and wide:
        kxk = "window"
    elif arrays == 3 and kernel_cols == 3:
        kxk = "rows"
    else:
        kxk = "kn2row"
    ab_c2 = conv_layer(kxk, (9, 9, 64), m.w["conv2"], m.b["conv2"], m.scale["conv2"], True, ab_rs, B0 + 324,
                       c2_bases)
    sch.layer = "conv3"
    ab_c3 = conv_layer(kxk, (7, 7, 32), m.w["conv3"], m.b["conv3"], m.scale["conv3"], True, ab_c2, B1 + 486)
    sch.layer = "fc1"
    # with the 3 x 3 window, fc1 (a 5x5 kernel over the whole 5x5 conv3 output)
    # takes four kernel-tile passes instead of 25 one-beat passes
    ab_x = conv_layer("window" if kxk == "window" else "kn2row", (5, 5, 8), fc_as_conv(m.w["fc1"].reshape(16, -1), 5, 5), m.b["fc1"], m.scale["fc1"],
                      True, ab_c3, B0 + 422)
    sch.layer = "fc2..fc4"
    for acc_row, name in zip((B1 + 511, B0 + 423, B1 + 510), ("fc2", "fc3", "fc4")):
        w = fc_as_conv(m.w[name], 1, 1)
        ab_x = conv_layer("kn2row", (1, 1, w.shape[2]), w, m.b[name], m.scale[name], True, ab_x, acc_row)
    sch.layer = "output"
    w = fc_as_conv(m.w["output"], 1, 1)
    OUT = B0 + 424
    low_out = lower("kn2row", (1, 1, 2), w, m.b["output"], m.scale["output"], False, ab_x, OUT)
    run_passes(low_out.passes)
    inf.end = sch.end
    inf.output_acc_row, inf.output_set = OUT, low_out.requant[0][0]
    tb.pad(inf.end)
    inf.usage = {"wb_rows": wb.next, "ab_rows": ab.next, "requant_sets": sets.next}
    inf.trace = sch.trace
    return inf
