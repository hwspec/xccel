"""Quantized BraggNN computed with XCCEL's integer semantics (numpy).

This is the specification the per-layer cocotb tests check against, not the
Gemmini reference: every float scale of the quantized model becomes a
Requantizer multiplier / shift (round half away from zero, saturation), and
softmax is the integer I-BERT softmax with K / sum. Accuracy is measured
against the ground-truth peak positions only.

Layer order (chia-braggnn-workspace exo/braggnn_reference.py):
  conv1 -> NLB (theta, phi, g; theta . phi^T; softmax; attention . g; out conv;
  resadd + ReLU) -> conv2 + ReLU -> conv3 + ReLU -> CHW flatten ->
  fc1..fc4 + ReLU -> output
"""

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data/braggnn.json"
DEQUANT_SCALE = 11.0 / 127.0  # int8 prediction -> pixels (braggnn_main.c)
MAX_AVG_ERROR_PX = 0.5
MULT_WIDTH = 16  # Requantizer multWidth
SOFTMAX_OUT_SHIFT = 16


def load(path=DATA):
    raw = json.loads(Path(path).read_text())
    arrays = {k: np.array(v["data"]).reshape(v["shape"]) for k, v in raw["arrays"].items()}
    return raw["defines"], arrays


def scale_to_mult_shift(scale, mult_width=MULT_WIDTH, max_shift=63):
    """Integer mult / shift with mult / 2^shift ~ scale, mult as wide as possible."""
    shift = 0
    while shift < max_shift and round(scale * (1 << (shift + 1))) < (1 << mult_width):
        shift += 1
    mult = round(scale * (1 << shift))
    assert 0 < mult < (1 << mult_width), f"scale {scale} not representable"
    return mult, shift


def requant(acc, bias, mult, shift, lo=-128, hi=127):
    """Requantizer: clamp(roundShift((acc + bias) * mult, shift), lo, hi)."""
    p = (np.asarray(acc, dtype=np.int64) + bias) * mult
    if shift:
        half = 1 << (shift - 1)
        p = np.where(p < 0, p + half - 1, p + half) >> shift
    return np.clip(p, lo, hi)


def requant_scale(acc, bias, scale, relu=False):
    mult, shift = scale_to_mult_shift(scale)
    return requant(acc, bias, mult, shift, 0 if relu else -128, 127)


# ---- integer I-BERT softmax (matches the Softmax unit) -----------------------
def _i32(x):
    x = np.asarray(x, dtype=np.int64) & 0xFFFFFFFF
    return np.where(x >= 1 << 31, x - (1 << 32), x)


def softmax_consts(in_scale, out_scale, out_shift=SOFTMAX_OUT_SHIFT):
    qln2 = int(0.693147 / in_scale)
    qln2_inv = 65536 // qln2
    return dict(
        qln2=qln2,
        qln2_inv=qln2_inv,
        qb=int(1.353 / in_scale),
        qc=int(0.344 / (0.3585 * in_scale * in_scale)),
        max_shift=(255 * qln2_inv) // 65536 + 1,
        k=round((1 << out_shift) / out_scale),
        out_shift=out_shift,
    )


def softmax_rows(x, c):
    """x: int8 logits [rows, n]; softmax over each row -> int8."""
    x = np.asarray(x, dtype=np.int64)
    neg = x.max(axis=1, keepdims=True) - x
    z = _i32((neg * c["qln2_inv"]) >> 16)
    qp = _i32(_i32(z * c["qln2"]) - neg)
    qpb = _i32(qp + c["qb"])
    q = _i32(_i32(qpb * qpb) + c["qc"])
    q = q >> np.minimum(z, c["max_shift"])
    s = _i32(q.sum(axis=1, keepdims=True))
    recip = c["k"] // s
    out = (q * recip + (1 << (c["out_shift"] - 1))) >> c["out_shift"]
    return np.clip(out, -128, 127)


# ---- layers -------------------------------------------------------------------
def conv(x, w, b, scale, relu):
    """x [H, W, Cin] int8, w [K, K, Cin, Cout] (HWIO), b [Cout] -> requantized."""
    k = w.shape[0]
    oh, ow = x.shape[0] - k + 1, x.shape[1] - k + 1
    acc = np.zeros((oh, ow, w.shape[3]), dtype=np.int64)
    for kh in range(k):
        for kw in range(k):
            acc += np.tensordot(x[kh:kh + oh, kw:kw + ow, :].astype(np.int64), w[kh, kw].astype(np.int64), axes=1)
    return requant_scale(acc, b, scale, relu)


class BraggNN:
    def __init__(self, path=DATA):
        self.d, self.a = load(path)
        d, a = self.d, self.a
        self.w = {
            "conv1": a["conv1_weights_flat"].reshape(3, 3, 1, 64),
            "theta": a["nlb_theta_weights_flat"].reshape(1, 1, 64, 32),
            "phi": a["nlb_phi_weights_flat"].reshape(1, 1, 64, 32),
            "g": a["nlb_g_weights_flat"].reshape(1, 1, 64, 32),
            "nlb_out": a["nlb_out_weights_flat"].reshape(1, 1, 32, 64),
            "conv2": a["conv2_weights_flat"].reshape(3, 3, 64, 32),
            "conv3": a["conv3_weights_flat"].reshape(3, 3, 32, 8),
            "fc1": a["fc1_weights"].reshape(16, 8, 5, 5),
            "fc2": a["fc2_weights"].reshape(8, 16),
            "fc3": a["fc3_weights"].reshape(4, 8),
            "fc4": a["fc4_weights"].reshape(2, 4),
            "output": a["output_weights"].reshape(2, 2),
        }
        self.b = {k: a[f"{'nlb_' + k if k in ('theta', 'phi', 'g') else k}_bias"] for k in
                  ("conv1", "theta", "phi", "g", "nlb_out", "conv2", "conv3", "fc1", "fc2", "fc3", "fc4", "output")}
        self.scale = {
            "conv1": d["CNN_LAYERS_0_CONV_QUANT_ACC_SCALE"],
            "theta": d["NLB_THETA_LAYER_CONV_QUANT_ACC_SCALE"],
            "phi": d["NLB_PHI_LAYER_CONV_QUANT_ACC_SCALE"],
            "g": d["NLB_G_LAYER_CONV_QUANT_ACC_SCALE"],
            "theta_phi": d["NLB_MATMUL_QUANT_ACC_SCALE"],
            "attention_g": d["NLB_MATMUL_1_QUANT_ACC_SCALE"],
            "nlb_out": d["NLB_OUT_CNN_CONV_QUANT_ACC_SCALE"],
            "resadd_a": d["NLB_ADD_B_SCALE"],  # conv1 output (skip connection)
            "resadd_b": d["NLB_ADD_A_SCALE"],  # NLB output
            "resadd_c": d["CNN_LAYERS_1_LEAKYRELU_QUANT_ACC_SCALE"],
            "conv2": d["CNN_LAYERS_2_CONV_QUANT_ACC_SCALE"] * d["CNN_LAYERS_3_LEAKYRELU_QUANT_ACC_SCALE"],
            "conv3": d["CNN_LAYERS_4_CONV_QUANT_ACC_SCALE"] * d["CNN_LAYERS_5_LEAKYRELU_QUANT_ACC_SCALE"],
            "fc1": d["DENSE_LAYERS_0_GEMM_ACC_SCALE"] * d["DENSE_LAYERS_1_LEAKYRELU_QUANT_ACC_SCALE"],
            "fc2": d["DENSE_LAYERS_2_GEMM_ACC_SCALE"] * d["DENSE_LAYERS_3_LEAKYRELU_QUANT_ACC_SCALE"],
            "fc3": d["DENSE_LAYERS_4_GEMM_ACC_SCALE"] * d["DENSE_LAYERS_5_LEAKYRELU_QUANT_ACC_SCALE"],
            "fc4": d["DENSE_LAYERS_6_GEMM_ACC_SCALE"] * d["DENSE_LAYERS_7_LEAKYRELU_QUANT_ACC_SCALE"],
            "output": d["DENSE_LAYERS_8_GEMM_ACC_SCALE"],
        }
        self.softmax = softmax_consts(d["SOFTMAX_INPUT_SCALE"], d["SOFTMAX_OUTPUT_SCALE"])
        n = d["NUM_TEST_PATCHES"]
        self.inputs = np.stack([a[f"test_input_{i}"] for i in range(n)])
        self.labels = np.stack([a[f"test_label_{i}"] for i in range(n)])

    # -- whole network, keeping every intermediate --------------------------------
    def quantize_input(self, patch):
        return np.trunc(np.asarray(patch) * 127.0).astype(np.int64).reshape(11, 11, 1)

    def forward(self, patch):
        t = {"input": self.quantize_input(patch)}
        t["conv1"] = conv(t["input"], self.w["conv1"], self.b["conv1"], self.scale["conv1"], False)
        for k in ("theta", "phi", "g"):
            t[k] = conv(t["conv1"], self.w[k], self.b[k], self.scale[k], False)
        theta, phi, g = (t[k].reshape(81, 32) for k in ("theta", "phi", "g"))
        t["theta_phi"] = requant_scale(theta @ phi.T, 0, self.scale["theta_phi"])  # [81, 81]
        t["attention"] = softmax_rows(t["theta_phi"], self.softmax)
        t["attention_g"] = requant_scale(t["attention"] @ g, 0, self.scale["attention_g"]).reshape(9, 9, 32)
        t["nlb_out"] = conv(t["attention_g"], self.w["nlb_out"], self.b["nlb_out"], self.scale["nlb_out"], False)
        a_q = requant_scale(t["conv1"], 0, self.scale["resadd_a"])
        b_q = requant_scale(t["nlb_out"], 0, self.scale["resadd_b"])
        t["resadd"] = requant_scale(a_q + b_q, 0, self.scale["resadd_c"], relu=True)
        t["conv2"] = conv(t["resadd"], self.w["conv2"], self.b["conv2"], self.scale["conv2"], True)
        t["conv3"] = conv(t["conv2"], self.w["conv3"], self.b["conv3"], self.scale["conv3"], True)
        flat = t["conv3"].transpose(2, 0, 1).reshape(-1)  # CHW flatten
        v = requant_scale(self.w["fc1"].reshape(16, -1) @ flat, self.b["fc1"], self.scale["fc1"], True)
        t["fc1"] = v
        for k in ("fc2", "fc3", "fc4"):
            v = requant_scale(self.w[k] @ v, self.b[k], self.scale[k], True)
            t[k] = v
        t["output"] = requant_scale(self.w["output"] @ v, self.b["output"], self.scale["output"])
        return t

    def accuracy(self, predictions):
        """predictions: int8 [n, 2] -> mean |error| in pixels per axis and pass flag."""
        err = np.abs(np.asarray(predictions) * DEQUANT_SCALE - self.labels[:len(predictions)])
        avg = err.mean(axis=0)
        return err, avg, bool((avg <= MAX_AVG_ERROR_PX).all())


if __name__ == "__main__":
    m = BraggNN()
    preds = np.array([m.forward(p)["output"] for p in m.inputs])
    err, avg, ok = m.accuracy(preds)
    for i, (p, e) in enumerate(zip(preds, err)):
        print(f"patch {i}: pred={p.tolist()} err=({e[0]:.3f}, {e[1]:.3f}) px")
    print(f"avg error: ({avg[0]:.3f}, {avg[1]:.3f}) px (tolerance {MAX_AVG_ERROR_PX}) {'PASS' if ok else 'FAIL'}")
