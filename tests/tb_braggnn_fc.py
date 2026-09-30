import cocotb
import numpy as np

from braggnn_model import BraggNN
from braggnn_xccel import fc_as_conv, lower_conv, run_lowered
from xccel_bridge import XCCEL_Bridge

PATCH = 0


@cocotb.test()
async def tb_braggnn_fc(cocotb_dut):
    """BraggNN fc1 (flattened 5x5x8 -> 16, the whole-input conv), fc2..fc4
    (ReLU) and output (no ReLU), each run on its own with the integer model's
    previous activation as input."""
    dut = XCCEL_Bridge(cocotb_dut)
    await dut.setup()
    await dut.softReset()
    m = BraggNN()
    t = m.forward(m.inputs[PATCH])
    R, C = dut.p.rows, dut.p.cols

    layers = [("fc1", t["conv3"], fc_as_conv(m.w["fc1"].reshape(16, -1), 5, 5), True)]
    prev = t["fc1"]
    for name in ("fc2", "fc3", "fc4", "output"):
        layers.append((name, prev.reshape(1, 1, -1), fc_as_conv(m.w[name], 1, 1), name != "output"))
        prev = t[name]

    total = 0
    for name, x, w, relu in layers:
        low = lower_conv(x, w, m.b[name], m.scale[name], relu, t[name].reshape(1, 1, -1), R, C)
        total += await run_lowered(dut, low, name)
    dut.log.info(f"fc1..output verified, {total} cycles in all; prediction {t['output'].tolist()}")
