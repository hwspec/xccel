import cocotb

from braggnn_model import BraggNN
from braggnn_xccel import lower_conv, run_lowered
from xccel_bridge import XCCEL_Bridge

PATCH = 0


@cocotb.test()
async def tb_braggnn_nlb1x1(cocotb_dut):
    """BraggNN's NLB 1x1 convs, each on its own with the integer model's
    input: theta, phi, g (9x9x64 -> 9x9x32) and the NLB output conv
    (9x9x32 -> 9x9x64), none with ReLU."""
    dut = XCCEL_Bridge(cocotb_dut)
    await dut.setup()
    await dut.softReset()
    m = BraggNN()
    t = m.forward(m.inputs[PATCH])
    R, C = dut.p.rows, dut.p.cols
    for name, x in (("theta", t["conv1"]), ("phi", t["conv1"]), ("g", t["conv1"]), ("nlb_out", t["attention_g"])):
        low = lower_conv(x, m.w[name], m.b[name], m.scale[name], False, t[name], R, C)
        await run_lowered(dut, low, name)
    dut.log.info("NLB 1x1 convs verified")
