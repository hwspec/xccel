"""Building Xccel microcode from a per-cycle control table.

A program is first written as one dict per cycle, {MicroWord field path:
value}, naming what every element does in that cycle (strided fields hold
the value of that cycle). `compress` folds runs of cycles into MicroWords:
a run continues while every non-strided field is unchanged and every
strided field keeps advancing by the stride set by the run's first two
cycles. This mirrors CycleProgram.compress in the Scala tests.
"""

# strided field -> its stride field (every other field must stay constant within a word)
STRIDES = {"addr": "addrStride", "shiftAmt": "shiftStride", "column": "columnStride", "row": "rowStride"}


def _stride_of(path):
    head, _, leaf = path.rpartition(".")
    return f"{head}.{STRIDES[leaf]}" if leaf in STRIDES and head else None


def compress(cycles, max_run_length, microword_layout):
    """cycles: list of {path: value} (missing fields are 0). Returns a list
    of {path: value} MicroWords with `length` and, on the last, `halt`."""
    fields = microword_layout["fields"]
    strided = [p for p in fields if _stride_of(p) in fields]
    words = []  # (first cycle, strides, length)
    for c in cycles:
        if words:
            first, strides, length = words[-1]
            flags_same = all(c.get(p, 0) == first.get(p, 0)
                             for p in fields if p not in strided and not p.endswith("Stride")
                             and p not in ("length", "halt"))
            if length < max_run_length and flags_same:
                if length == 1:
                    strides = {p: c.get(p, 0) - first.get(p, 0) for p in strided}
                if all(c.get(p, 0) == first.get(p, 0) + strides[p] * length for p in strided):
                    words[-1] = (first, strides, length + 1)
                    continue
        words.append((c, {p: 0 for p in strided}, 1))

    out = []
    for i, (first, strides, length) in enumerate(words):
        w = {p: v for p, v in first.items() if v}
        for p, s in strides.items():
            if s:
                w[_stride_of(p)] = s
        w["length"] = length
        if i == len(words) - 1:
            w["halt"] = 1
        out.append(w)
    return out


class Table:
    """A per-cycle control table that grows as cycles are written."""

    def __init__(self):
        self.cycles = []

    def set(self, t, **kw):
        """Set fields for cycle t; keyword names use '__' for '.'
        (abRead__addr=5 sets abRead.addr)."""
        while len(self.cycles) <= t:
            self.cycles.append({})
        for k, v in kw.items():
            path = k.replace("__", ".")
            old = self.cycles[t].get(path)
            assert old is None or old == v, f"cycle {t}: {path} set to {old} and {v}"
            self.cycles[t][path] = v

    def put(self, t, fields):
        """Like set, with dotted field paths: put(t, {"abRead.addr": 5})."""
        self.set(t, **{k.replace(".", "__"): v for k, v in fields.items()})

    def pad(self, n):
        while len(self.cycles) < n:
            self.cycles.append({})


def schedule_passes(table, passes, p, start=0):
    """Weight-stationary passes back to back, by the conv-cycles.py rules,
    the first one's schedule starting at cycle `start` (its weight load may
    begin there).

    passes: list of dicts {ab: [ActivationBuffer rows], wb: first WeightBuffer
    row of the R-row tile, acc: [Accumulator rows], accumulate: bool}, all
    with the same stream length. p: params (rows, cols, latencies).
    Returns the cycle after the last result write (the cycle count from
    start to halt when start = 0)."""
    stream = len(passes[0]["ab"])
    assert all(len(ps["ab"]) == stream for ps in passes)
    R, C = p.rows, p.cols
    commit = p.acc_read_latency + p.acc_write_latency
    latency = R + C - 1
    period = max(stream, R, commit, (stream + C) // 2)
    s0 = max(0, p.wb_read_latency + 1 - p.ab_read_latency)
    base = start
    for t, ps in enumerate(passes):
        start = base + s0 + t * period
        slot = t % 2
        wld = start + p.ab_read_latency - p.wb_read_latency - 1
        for r in range(R):
            table.set(wld + r, wbRead__en=1, wbRead__addr=ps["wb"] + r)
            table.set(wld + p.wb_read_latency + r, weightPort__en=1, weightPort__addr=r, weightPort__slot=slot)
        for i in range(stream):
            table.set(start + i, abRead__en=1, abRead__addr=ps["ab"][i])
            arrive = start + p.ab_read_latency + i
            if slot:
                table.set(arrive, compute__weightSelect=1)
            table.set(arrive + latency, accWrite__addr=ps["acc"][i],
                      accWrite__accumulate=int(ps["accumulate"]))
    total = base + s0 + (len(passes) - 1) * period + stream + p.ab_read_latency + latency
    table.pad(total)
    return total


def schedule_drain(table, start, items, p):
    """One Accumulator row per cycle from `start`: items are (acc row,
    {field: value} to set when the row arrives accumulator.readLatency
    later, e.g. requant.paramSet and a destination). None skips a cycle.
    Returns the cycle after the last arrival."""
    for i, item in enumerate(items):
        if item is None:
            continue
        row, at_arrival = item
        table.set(start + i, accRead__en=1, accRead__addr=row)
        table.put(start + i + p.acc_read_latency, at_arrival)
    end = start + len(items) + p.acc_read_latency
    table.pad(end)
    return end
