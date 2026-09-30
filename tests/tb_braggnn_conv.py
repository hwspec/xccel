import cocotb
import numpy as np

from braggnn_model import BraggNN
from braggnn_xccel import lower_conv, lower_conv_im2col, run_lowered
from xccel_bridge import XCCEL_Bridge

PATCH = 0


async def setup(cocotb_dut):
    dut = XCCEL_Bridge(cocotb_dut)
    await dut.setup()
    await dut.softReset()
    return dut


@cocotb.test()
async def tb_braggnn_conv2(cocotb_dut):
    """BraggNN conv2 (9x9x64 -> 7x7x32, ReLU) of test patch 0, with the
    integer model's resadd output as input: 72 passes (9 kernel positions x
    4 C_in tiles x 2 C_out tiles) of 49 valid outputs each."""
    dut = await setup(cocotb_dut)
    m = BraggNN()
    t = m.forward(m.inputs[PATCH])
    low = lower_conv(t["resadd"], m.w["conv2"], m.b["conv2"], m.scale["conv2"], True, t["conv2"],
                     dut.p.rows, dut.p.cols)
    cycles = await run_lowered(dut, low, "conv2")
    assert np.any(t["conv2"] > 0)
    dut.log.info(f"conv2 verified in {cycles} cycles")


@cocotb.test()
async def tb_braggnn_conv3(cocotb_dut):
    """BraggNN conv3 (7x7x32 -> 5x5x8, ReLU): 18 passes of 25 outputs,
    8 of the 16 output lanes used."""
    dut = await setup(cocotb_dut)
    m = BraggNN()
    t = m.forward(m.inputs[PATCH])
    low = lower_conv(t["conv2"], m.w["conv3"], m.b["conv3"], m.scale["conv3"], True, t["conv3"],
                     dut.p.rows, dut.p.cols)
    cycles = await run_lowered(dut, low, "conv3")
    dut.log.info(f"conv3 verified in {cycles} cycles")


@cocotb.test()
async def tb_braggnn_conv1(cocotb_dut):
    """BraggNN conv1 (11x11x1 -> 9x9x64, no ReLU) by im2col: the host writes
    81 rows of 9 window values; 4 passes (one chunk x 4 C_out tiles) of 81."""
    dut = await setup(cocotb_dut)
    m = BraggNN()
    t = m.forward(m.inputs[PATCH])
    low = lower_conv_im2col(t["input"], m.w["conv1"], m.b["conv1"], m.scale["conv1"], False, t["conv1"],
                            dut.p.rows, dut.p.cols)
    cycles = await run_lowered(dut, low, "conv1")
    dut.log.info(f"conv1 verified in {cycles} cycles")
