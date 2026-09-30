import os

import cocotb
import numpy as np

from braggnn_model import BraggNN
from braggnn_program_opt import build_opt
from braggnn_xccel import lower_conv_im2col
from xccel_bridge import XCCEL_Bridge
from xccel_program import compress

PATCHES = int(os.getenv("BRAGGNN_PATCHES", "10"))
# Accumulator read port model: from the generated configuration (acc_row_banks)
# unless XCCEL_ACC_READ (shared, banked or dual) overrides it
ACC_READ = os.getenv("XCCEL_ACC_READ")


@cocotb.test()
async def tb_braggnn_inference_opt(cocotb_dut):
    """List-scheduled version of tb_braggnn_inference (braggnn_program_opt): the host
    writes only conv1's im2col rows, runs, and reads the two output lanes.
    Accuracy against the ground-truth peak positions (mean |error| <= 0.5 px
    per axis, as in the Gemmini evaluation harness)."""
    dut = XCCEL_Bridge(cocotb_dut)
    await dut.setup()
    await dut.softReset()
    p = dut.p
    m = BraggNN()
    consts = dut.layout["microword_consts"]
    acc_read = ACC_READ or ("banked" if getattr(p, "acc_row_banks", 1) == 2 else "shared")
    arrays = getattr(p, "compute_arrays", 1)
    inf = build_opt(m, p, consts["softmax_op"], consts["softmax_div_latency"], acc_read, p.softmax_stats,
                    arrays=arrays, sum_ops=consts.get("softmax_sum_op"),
                    tree_latency=consts["compute_latency"] if consts.get("compute_tree") else 0)
    words = compress(inf.table.cycles, p.max_run_length, dut.layout["microword"])
    dut.log.info(f"program ({acc_read} Accumulator read, {p.softmax_stats} softmax stats, {arrays} arrays, {inf.order} order): "
                 f"{len(words)} words, "
                 f"{inf.end} cycles, usage {inf.usage}")
    assert len(words) <= p.program_length, f"{len(words)} words exceed the program store"

    # static: weights, parameters, program
    for addr, row in inf.wb_rows:
        await dut.writeWbRow(addr, row)
    for args in inf.requant:
        await dut.loadRequant(*args)
    for args in inf.residual:
        await dut.loadResidual(*args)
    s = m.softmax
    await dut.loadSoftmax(s["qln2"], s["qln2_inv"], s["qb"], s["qc"], s["max_shift"], s["k"], s["out_shift"])
    await dut.loadProgram(words)

    preds, model_preds = [], []
    for i in range(PATCHES):
        t = m.forward(m.inputs[i])
        low = lower_conv_im2col(t["input"], m.w["conv1"], m.b["conv1"], m.scale["conv1"], False, t["conv1"],
                                p.rows, p.cols, ab_base=inf.conv1_ab_base)
        for addr, row in low.ab_rows:
            await dut.writeAbRow(addr, row)
        await dut.run()
        cycles = await dut.readCycles()
        out = (await dut.readAcc(inf.output_acc_row, inf.output_set))[:2]
        preds.append(out)
        model_preds.append(t["output"].tolist())
        err = np.abs(np.array(out) * 11.0 / 127.0 - m.labels[i])
        dut.log.info(f"patch {i}: cycles={cycles} pred={out} err=({err[0]:.3f}, {err[1]:.3f}) px"
                     f"{'' if out == model_preds[-1] else f' (integer model: {model_preds[-1]})'}")
        assert cycles == inf.end, f"patch {i}: {cycles} cycles, predicted {inf.end}"

    _, avg, ok = m.accuracy(np.array(preds))
    dut.log.info(f"avg error: ({avg[0]:.3f}, {avg[1]:.3f}) px (tolerance 0.5) {'PASS' if ok else 'FAIL'}")
    assert ok, f"average error {avg} px exceeds 0.5 px"
