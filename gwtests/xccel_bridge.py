import json
import os
from pathlib import Path

from garageworks.bridge.cocotb_bridge import COCOTB_Bridge

# ---------------------------------------------------------------------------
# Status register bit masks (status_r)
# ---------------------------------------------------------------------------
ST_RUNNING = 1 << 0
ST_DONE = 1 << 1


def pack(layout, values):
    """Pack {field path: value} into the integer asUInt would produce, using
    a layout.json entry ({"width": w, "fields": {path: [lsb, width]}}).
    Unset fields are 0; values are taken modulo their field width (so a
    negative stride becomes its two's complement)."""
    fields = layout["fields"]
    word = 0
    for path, value in values.items():
        if path not in fields:
            raise KeyError(f"unknown field {path}")
        lsb, width = fields[path]
        word |= (int(value) & ((1 << width) - 1)) << lsb
    return word


def pack_lanes(lanes, width):
    """Pack a row of lane values (lane 0 in the least significant bits)."""
    word = 0
    for i, v in enumerate(lanes):
        word |= (int(v) & ((1 << width) - 1)) << (i * width)
    return word


def to_signed(v, width):
    return v - (1 << width) if v >> (width - 1) & 1 else v


class XCCEL_Bridge(COCOTB_Bridge):
    def __init__(self, cocotb_dut):
        super().__init__(cocotb_dut)
        # layout.json sits next to params.json
        with open(Path(os.getenv("PARAMFN")).with_name("layout.json")) as f:
            self.layout = json.load(f)

    # -- staging ---------------------------------------------------------
    async def writeStage(self, value, width):
        nwords = (width + 31) // 32
        assert nwords <= self.p.stage_words, f"{width}-bit value exceeds the staging buffer"
        for i in range(nwords):
            await self.writeWord(self.p.stage_base_rw + i * 0x10, (value >> (32 * i)) & 0xFFFFFFFF)

    async def commit(self, addr, index):
        await self.writeWord(addr, index)

    # -- program, buffers, parameters (Xccel idle) -----------------------
    async def loadProgram(self, words):
        """words: list of {field path: value} dicts (see xccel_program)."""
        lay = self.layout["microword"]
        for i, w in enumerate(words):
            await self.writeStage(pack(lay, w), lay["width"])
            await self.commit(self.p.prog_commit_w, i)

    async def writeWbRow(self, addr, lanes):
        """lanes: up to a whole WeightBuffer row (cols * wb_words_per_bank words); the rest is zero."""
        words = self.p.cols * getattr(self.p, "wb_words_per_bank", 1)
        assert len(lanes) <= words
        await self.writeStage(pack_lanes(lanes, self.p.weight_width), words * self.p.weight_width)
        await self.commit(self.p.wb_commit_w, addr)

    async def writeAbRow(self, addr, lanes):
        await self.writeStage(pack_lanes(lanes, self.p.input_width), self.p.rows * self.p.input_width)
        await self.commit(self.p.ab_commit_w, addr)

    def _params(self, bias, mult, shift, lo, hi):
        v = {"lo": lo, "hi": hi}
        for i in range(self.p.cols):
            v[f"bias.{i}"] = bias[i]
            v[f"mult.{i}"] = mult[i]
            v[f"shift.{i}"] = shift[i]
        return v

    async def loadRequant(self, pset, bias, mult, shift, lo, hi):
        lay = self.layout["requant_params"]
        await self.writeStage(pack(lay, self._params(bias, mult, shift, lo, hi)), lay["width"])
        await self.commit(self.p.requant_commit_w, pset)

    async def loadResidual(self, pset, bias, mult, shift, lo, hi):
        lay = self.layout["residual_params"]
        await self.writeStage(pack(lay, self._params(bias, mult, shift, lo, hi)), lay["width"])
        await self.commit(self.p.residual_commit_w, pset)

    async def loadSoftmax(self, qln2, qln2_inv, qb, qc, max_shift, k, out_shift):
        lay = self.layout["softmax_params"]
        v = {"qln2": qln2, "qln2Inv": qln2_inv, "qb": qb, "qc": qc,
             "maxShift": max_shift, "k": k, "outShift": out_shift}
        await self.writeStage(pack(lay, v), lay["width"])
        await self.commit(self.p.softmax_commit_w, 0)

    # -- run ---------------------------------------------------------------
    async def start(self):
        await self.writeWord(self.p.start_rw, 1)

    async def readStatus(self):
        return await self.readWord(self.p.status_r)

    async def waitForDone(self, timeout=100000):
        st = 0
        for _ in range(timeout):
            st = await self.readStatus()
            if st & ST_DONE:
                return st
        raise TimeoutError(f"Timed out waiting for halt; last status={st:#010x}")

    async def readCycles(self):
        lo = await self.readWord(self.p.cycles_lo_r)
        hi = await self.readWord(self.p.cycles_hi_r)
        return (hi << 32) | lo

    async def run(self, timeout=100000):
        await self.start()
        return await self.waitForDone(timeout)

    # -- drain ---------------------------------------------------------------
    async def readAcc(self, addr, pset=0, timeout=100):
        """Accumulator row `addr` through the Requantizer (set `pset`) and
        Softmax (pass-through), as signed requantized lanes."""
        await self.writeWord(self.p.acc_param_set_rw, pset)
        await self.writeWord(self.p.acc_read_w, addr)
        for _ in range(timeout):
            if await self.readWord(self.p.acc_result_valid_r):
                break
        else:
            raise TimeoutError(f"no drain result for row {addr}")
        word = 0
        for i in range(self.p.result_words):
            word |= (await self.readWord(self.p.result_base_r + i * 0x10)) << (32 * i)
        w = self.p.requantized_width
        return [to_signed((word >> (i * w)) & ((1 << w) - 1), w) for i in range(self.p.cols)]
