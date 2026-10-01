import os

import cocotb
import numpy as np

from braggnn_model import BraggNN
from braggnn_program_opt import build_opt
from braggnn_xccel import lower_conv_im2col
from xccel_bridge import XCCEL_Bridge
from xccel_program import compress

# patches inferred by one program run (the program store bounds it: about 530
# MicroWords per patch on AxiXccelOpt9x64T)
BATCH = int(os.getenv("BRAGGNN_BATCH", "6"))


@cocotb.test()
async def tb_braggnn_batch(cocotb_dut):
    """Batch version of tb_braggnn_inference_opt: one program infers BATCH
    patches (build_opt(batch=...)). The host writes every patch's conv1 im2col
    rows, runs once, and reads each patch's output row. Checks the cycles
    against the schedule, every prediction against the integer model, and the
    accuracy against the ground truth."""
    dut = XCCEL_Bridge(cocotb_dut)
    await dut.setup()
    await dut.softReset()
    p = dut.p
    m = BraggNN()
    consts = dut.layout["microword_consts"]
    acc_read = "banked" if getattr(p, "acc_row_banks", 1) == 2 else "shared"
    arrays = getattr(p, "compute_arrays", 1)
    inf = build_opt(m, p, consts["softmax_op"], consts["softmax_div_latency"], acc_read, p.softmax_stats,
                    arrays=arrays, sum_ops=consts.get("softmax_sum_op"),
                    tree_latency=consts["compute_latency"] if consts.get("compute_tree") else 0, batch=BATCH)
    words = compress(inf.table.cycles, p.max_run_length, dut.layout["microword"])
    dut.log.info(f"batch of {BATCH} ({inf.order} order): {len(words)} words, {inf.end} cycles "
                 f"({inf.end / BATCH:.0f} per patch), usage {inf.usage}")
    assert len(words) <= p.program_length, f"{len(words)} words exceed the program store"

    for addr, row in inf.wb_rows:
        await dut.writeWbRow(addr, row)
    for args in inf.requant:
        await dut.loadRequant(*args)
    for args in inf.residual:
        await dut.loadResidual(*args)
    s = m.softmax
    await dut.loadSoftmax(s["qln2"], s["qln2_inv"], s["qb"], s["qc"], s["max_shift"], s["k"], s["out_shift"])
    await dut.loadProgram(words)

    patches = [i % len(m.inputs) for i in range(BATCH)]
    expected = []
    for k, i in enumerate(patches):
        t = m.forward(m.inputs[i])
        expected.append(t["output"].tolist())
        low = lower_conv_im2col(t["input"], m.w["conv1"], m.b["conv1"], m.scale["conv1"], False, t["conv1"],
                                p.rows, p.cols, ab_base=inf.conv1_ab_bases[k])
        for addr, row in low.ab_rows:
            await dut.writeAbRow(addr, row)

    await dut.run()
    cycles = await dut.readCycles()
    dut.log.info(f"cycles = {cycles} (predicted {inf.end}, {cycles / BATCH:.0f} per patch)")
    assert cycles == inf.end, f"{cycles} cycles, predicted {inf.end}"

    preds = []
    for k, i in enumerate(patches):
        out = (await dut.readAcc(inf.output_acc_rows[k], inf.output_set))[:2]
        preds.append(out)
        err = np.abs(np.array(out) * 11.0 / 127.0 - m.labels[i])
        dut.log.info(f"patch {i}: pred={out} err=({err[0]:.3f}, {err[1]:.3f}) px"
                     f"{'' if out == expected[k] else f' (integer model: {expected[k]})'}")
        assert out == expected[k], f"patch {i}: {out}, integer model {expected[k]}"

    err = np.abs(np.array(preds) * 11.0 / 127.0 - m.labels[patches])
    avg = err.mean(axis=0)
    ok = bool((avg <= 0.5).all())
    dut.log.info(f"avg error: ({avg[0]:.3f}, {avg[1]:.3f}) px (tolerance 0.5) {'PASS' if ok else 'FAIL'}")
    assert ok, f"average error {avg} px exceeds 0.5 px"
