"""Write the whole-BraggNN program that tb_braggnn_inference_opt loads, for
one generated configuration, without running a simulation.

    python tools/dump_program.py ../generated/AxiXccelOpt9x32B [out_prefix]

writes <out_prefix>.json ({"cycles", "config", "words": [{field: value}]},
the MicroWords exactly as loaded, fields at 0 omitted) and <out_prefix>.txt
(one line per word: index, length, the nonzero fields). out_prefix defaults
to <generated dir>/program. The build mirrors the test bench: Accumulator read
model from acc_row_banks, softmax stats, compute arrays and the softmax
constants from params.json / layout.json.
"""

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from braggnn_model import BraggNN  # noqa: E402
from braggnn_program_opt import build_opt  # noqa: E402
from xccel_program import compress  # noqa: E402


def main():
    gen = Path(sys.argv[1])
    prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else gen / "program"
    p = types.SimpleNamespace(**json.loads((gen / "params.json").read_text()))
    layout = json.loads((gen / "layout.json").read_text())
    consts = layout["microword_consts"]
    acc_read = "banked" if getattr(p, "acc_row_banks", 1) == 2 else "shared"
    arrays = getattr(p, "compute_arrays", 1)
    inf = build_opt(BraggNN(), p, consts["softmax_op"], consts["softmax_div_latency"], acc_read, p.softmax_stats,
                    arrays=arrays, sum_ops=consts.get("softmax_sum_op"),
                    tree_latency=consts["compute_latency"] if consts.get("compute_tree") else 0)
    words = compress(inf.table.cycles, p.max_run_length, layout["microword"])
    words = [{k: v for k, v in w.items() if v} for w in words]

    config = {"acc_read": acc_read, "softmax_stats": p.softmax_stats, "compute_arrays": arrays,
              "rows": p.rows, "cols": p.cols}
    Path(f"{prefix}.json").write_text(json.dumps({"cycles": inf.end, "config": config, "words": words}, indent=1))
    with open(f"{prefix}.txt", "w") as f:
        f.write(f"# {len(words)} words, {inf.end} cycles, {config}\n")
        for i, w in enumerate(words):
            rest = " ".join(f"{k}={v}" for k, v in w.items() if k not in ("length", "halt"))
            f.write(f"{i:5d} len={w.get('length', 0):3d}{' HALT' if w.get('halt') else ''}  {rest}\n")
    print(f"{len(words)} words, {inf.end} cycles -> {prefix}.json, {prefix}.txt")


if __name__ == "__main__":
    main()
