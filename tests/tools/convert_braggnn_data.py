#!/usr/bin/env python3
"""Convert the quantized BraggNN model and test data from a C header into JSON.

Usage: convert_braggnn_data.py [braggnn_data.h] [out.json]

The header (chia-braggnn-workspace exo/braggnn_data.h) holds the int8
weights, int32 biases, float scales (#defines), the fp32 test patches and
their ground-truth peak positions as plain C arrays. The JSON keeps every
array as {"shape": [...], "data": [flat values, row-major]} and every
numeric #define as a number, so the cocotb tests need neither the header
nor the workspace.
"""

import json
import re
import sys
from pathlib import Path

DEFAULT_SRC = Path.home() / "repo/chia-braggnn-workspace/exo/braggnn_data.h"
DEFAULT_DST = Path(__file__).resolve().parent.parent / "data/braggnn.json"

# arrays only the Gemmini schedules used (identity matrices for the old
# softmax trick, padding masks); XCCEL does not need them
SKIP = re.compile(r"identity|softmax_mask")

ARRAY = re.compile(
    r"^(?:static\s+)?(?:const\s+)?(?:float|double|elem_t|acc_t|u?int(?:8|16|32|64)_t|int)\s+"
    r"(\w+)((?:\s*\[[^\]]*\])+)\s*=\s*\{",
    re.M,
)
DEFINE = re.compile(r"^#define\s+(\w+)[ \t]+([^\n]+)$", re.M)
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
NUMERIC = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?f?")


def evaluate(expr, macros, seen=()):
    """Value of a macro expression: numbers, other macros, + - * / ( )."""
    expr = expr.strip()
    if NUMERIC.fullmatch(expr):
        v = float(expr.rstrip("f"))
        return int(v) if re.fullmatch(r"[-+]?\d+", expr) else v

    def name(m):
        n = m.group(0)
        if n in seen or n not in macros:
            raise KeyError(n)
        return f"({evaluate(macros[n], macros, seen + (n,))})"

    code = re.sub(r"\b[A-Za-z_]\w*\b", name, re.sub(r"(\d)f\b", r"\1", expr))
    return eval(code, {"__builtins__": {}})


def numeric_defines(macros):
    out = {}
    for n, expr in macros.items():
        try:
            out[n] = evaluate(expr, macros)
        except Exception:
            pass  # not a numeric macro
    return out


def dims(spec, defines):
    # dimensions may be numbers, macros or products of macros (A * B)
    return [int(evaluate(d, {k: str(v) for k, v in defines.items()})) for d in re.findall(r"\[([^\]]*)\]", spec)]


def body(text, start):
    """The initializer between the '{' at start and its matching '}'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unbalanced braces")


def convert(src):
    text = Path(src).read_text()
    text = re.sub(r"//[^\n]*", "", text)  # comments may contain numbers
    defines = numeric_defines({name: value for name, value in DEFINE.findall(text)})
    arrays = {}
    for m in ARRAY.finditer(text):
        name = m.group(1)
        if SKIP.search(name):
            continue
        shape = dims(m.group(2), defines)
        values = [float(x) if re.search(r"[.eE]", x) else int(x)
                  for x in NUMBER.findall(body(text, m.end() - 1))]
        size = 1
        for d in shape:
            size *= d
        if len(values) != size:
            raise ValueError(f"{name}: {len(values)} values for shape {shape}")
        arrays[name] = {"shape": shape, "data": values}
    return {"defines": defines, "arrays": arrays}


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DST
    data = convert(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data))
    print(f"wrote {dst}: {len(data['arrays'])} arrays, {len(data['defines'])} defines")
    for name, a in sorted(data["arrays"].items()):
        print(f"  {name:32s} {a['shape']}")


if __name__ == "__main__":
    main()
