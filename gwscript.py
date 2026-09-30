#!/usr/bin/env python3
#                                                                                                                                            # Copyright (c) 2026, UChicago Argonne, LLC.                                                                                                 # License: See LICENSE in the project top-level directory.                                                                                   # Main author: Kazutomo Yoshii <kazutomo@anl.gov>

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd().resolve()
CONFIG_FILE = ROOT / ".gwconfig"


def load_config():
    config = {}

    if not CONFIG_FILE.exists():
        raise RuntimeError(
            f"Missing config file: {CONFIG_FILE}\n"
            "Please create .gwconfig with:\n"
            "name=AxiSRV32I\n"
            "package=srv32i\n"
            "tests=loop\n"
            "fpgatest=true"
        )

    for raw in CONFIG_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()

    required = ["name", "package"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise RuntimeError(
            "Missing required .gwconfig keys: " + ", ".join(missing)
        )

    return config


CONFIG = load_config()
NAME = CONFIG["name"]
PACKAGE = CONFIG["package"]
TESTS = [
    item.strip()
    for item in CONFIG.get("tests", "").split(",")
    if item.strip()
]

AVED = ROOT / f"v80-aved-platform-{PACKAGE}"
AVED_HW = AVED / "hw/amd_v80_gen5x8_25.1"
USER_ACCEL = AVED_HW / "src/rtl/user_accel"
GENERATED = ROOT / "generated" / NAME
VENV = ROOT / "chisel-axi-utils/.venv"

STATUS_FILE = ROOT / ".gwstatus"
RESULTS_FILE = ROOT / ".gwstage-results.json"
LOG_DIR = ROOT / ".gwlogs"

completed = set()
stage_results = {}
stage_times = {}

TMUX_STAGES = {"fpgatiming", "firmware"}

FPGA_STAGES = {
    "fpgatiming",
    "aved",
    "useracc",
    "firmware",
    "program",
    "fpgatest",
}


def config_bool(name, default=False):
    value = CONFIG.get(name)
    if value is None:
        return default

    value = value.strip().lower()

    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    raise RuntimeError(
        f"Invalid boolean value in .gwconfig: {name}={CONFIG[name]}"
    )


FPGA_ENABLED = config_bool("fpgatest", default=False)

STAGE_ORDER = [
    "prereq",
    "repo",
    "bridge",
    "rtlgen",
    "cocotb",
    "fpgatiming",
    "aved",
    "useracc",
    "firmware",
    "program",
    "fpgatest",
]

REQUIRED_STAGE_ORDER = [
    "prereq",
    "repo",
    "bridge",
    "rtlgen",
    "cocotb",
    "aved",
    "useracc",
    "firmware",
    "program",
    "fpgatest",
]

OPTIONAL_STAGES = {"fpgatiming"}


def run(cmd, *, cwd=ROOT, env=None, ignore_returncode=False):
    print(f"\n[{cwd}] $ {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd, cwd=cwd, env=env, check=False)

    if result.returncode != 0:
        if ignore_returncode:
            print(
                f"WARNING: ignoring exit code {result.returncode}: "
                f"{' '.join(map(str, cmd))}"
            )
        else:
            raise subprocess.CalledProcessError(result.returncode, cmd)

    return result


def capture(cmd, *, cwd=ROOT):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def venv_env():
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(VENV)
    env["PATH"] = f"{VENV / 'bin'}:{env['PATH']}"
    return env


def default_status():
    return {name: "pending" for name in STAGE_ORDER}


def load_status():
    status = default_status()

    if not STATUS_FILE.exists():
        return status

    for raw in STATUS_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().lower()

        if name in status:
            status[name] = value

    return status


def save_status(status):
    lines = [
        "# GarageWork build status",
        "# Values: pending, running, success, failed",
        "# This file is intentionally editable by hand.",
        "",
    ]
    for name in STAGE_ORDER:
        lines.append(f"{name}={status.get(name, 'pending')}")
    STATUS_FILE.write_text("\n".join(lines) + "\n")


def set_stage_status(name, value):
    status = load_status()
    status[name] = value
    save_status(status)


def show_status():
    status = load_status()
    for name in STAGE_ORDER:
        print(f"{name:12s} {status[name]}")


def next_runnable_stage():
    status = load_status()

    for name in REQUIRED_STAGE_ORDER:
        if status.get(name) == "success":
            continue

        deps, _ = STAGES[name]
        required_deps = [dep for dep in deps if dep not in OPTIONAL_STAGES]

        if all(status.get(dep) == "success" for dep in required_deps):
            return name

    return None

def show_next():
    name = next_runnable_stage()

    if name is None:
        print("No pending runnable stage.")
        return

    deps, _ = STAGES[name]
    print(f"Next runnable stage: {name}")

    if deps:
        status = load_status()
        print(
            "Depends on: " +
            ", ".join(f"{dep} [{status.get(dep, 'pending')}]" for dep in deps)
        )
    else:
        print("Depends on: none")


def load_stage_results():
    if not RESULTS_FILE.exists():
        return {}

    try:
        return json.loads(RESULTS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_stage_results(data):
    RESULTS_FILE.write_text(json.dumps(data, indent=2) + "\n")


def record_stage_success(name, result, elapsed_sec):
    data = load_stage_results()
    entry = data.setdefault(name, {})
    now = datetime.now().isoformat(timespec="seconds")

    entry["last_attempt"] = {
        "timestamp": now,
        "success": True,
        "elapsed_sec": elapsed_sec,
    }
    entry["last_success"] = {
        "timestamp": now,
        "elapsed_sec": elapsed_sec,
        "result": result,
    }

    save_stage_results(data)


def record_stage_failure(name, elapsed_sec, message):
    data = load_stage_results()
    entry = data.setdefault(name, {})

    entry["last_attempt"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "success": False,
        "elapsed_sec": elapsed_sec,
        "error": message,
    }

    save_stage_results(data)


def get_stage_info(name):
    return load_stage_results().get(name)


def get_stage_result(name):
    info = get_stage_info(name)
    if not info:
        return None

    last_success = info.get("last_success")
    if not last_success:
        return None

    return last_success.get("result")


def stage_succeeded(name):
    return get_stage_result(name) is not None


def prereq():
    verilator = capture(["verilator", "--version"])
    print(f"Verilator: {verilator}")

    m = re.search(r"(\d+\.\d+)", verilator)
    if not m or m.group(1) != "5.044":
        print("WARNING: Verilator 5.044 is recommended.")

    gcc = capture(["gcc", "-dumpfullversion"])
    print(f"GCC: {gcc}")

    if gcc != "11.1.0":
        print("WARNING: GCC 11.1.0 is recommended (non-strict check).")

    return {"verilator": verilator, "gcc": gcc}


def repo():
    status = capture(["git", "submodule", "status"])
    uninitialized = any(
        line.startswith("-")
        for line in status.splitlines()
        if line.strip()
    )

    if uninitialized:
        run(["git", "submodule", "update", "--init"])
    else:
        print("Git submodules already initialized.")

    return {"initialized": True}


def bridge():
    run(["make", "-C", "chisel-axi-utils", "setup"])
    return {"venv": str(VENV)}


def rtlgen():
    run(["sbt", f"runMain {PACKAGE}.{NAME}"])

    if not GENERATED.is_dir():
        raise RuntimeError(
            f"RTL generation succeeded but {GENERATED} does not exist"
        )

    return {"generated_dir": str(GENERATED)}


def cocotb():
    if not TESTS:
        raise RuntimeError("No tests defined in .gwconfig (example: tests=basic,loop)")

    env = venv_env()
    run(["make", "-C", "tests", "clean"], env=env)

    for testname in TESTS:
        print(f"\nRunning cocotb test target: {testname}")
        run(["make", "-C", "tests", testname], env=env)

    return {
        "tests_dir": str(ROOT / "tests"),
        "tests": TESTS,
    }


def _set_xdc_period(xdc_path, period_ns):
    text = xdc_path.read_text()

    pattern = re.compile(
        r"(create_clock\b[^\n]*?-period\s+)([0-9]*\.?[0-9]+)",
        re.IGNORECASE,
    )

    new_text, count = pattern.subn(
        lambda m: f"{m.group(1)}{period_ns:.4f}",
        text,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            f"Could not uniquely update create_clock -period in {xdc_path}"
        )

    xdc_path.write_text(new_text)


def _find_timing_report(workdir):
    candidates = list(workdir.rglob("*timing*.rpt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_wns_from_text(text):
    patterns = [
        r"\bWNS(?:\(ns\))?\s*[:=]?\s*(-?\d+(?:\.\d+)?)",
        r"\bWNS\s+TNS\b.*?\n\s*(-?\d+(?:\.\d+)?)",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            return float(m.group(1))

    return None


def _run_timing_trial(period_ns):
    workdir = GENERATED
    xdc = workdir / "constraints.xdc"

    if not xdc.exists():
        raise RuntimeError(f"Missing XDC file: {xdc}")

    _set_xdc_period(xdc, period_ns)

    LOG_DIR.mkdir(exist_ok=True)
    safe = str(period_ns).replace(".", "p")
    log_path = LOG_DIR / f"fpgatiming-{safe}ns.log"

    print(f"\nTrying FPGA period {period_ns:.4f} ns")
    print(f"Log: {log_path}")

    with log_path.open("w") as logfile:
        proc = subprocess.Popen(
            ["./compile.sh"],
            cwd=workdir,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        assert proc.stdout is not None
        output_lines = []

        for line in proc.stdout:
            sys.stdout.write(line)
            logfile.write(line)
            output_lines.append(line)

        rc = proc.wait()

    output = "".join(output_lines)

    report = _find_timing_report(workdir)
    report_text = report.read_text(errors="replace") if report else ""

    wns = _parse_wns_from_text(report_text) if report_text else None
    if wns is None:
        wns = _parse_wns_from_text(output)

    compile_ok = (rc == 0)
    timing_ok = compile_ok and (wns is not None) and (wns >= 0.0)

    return {
        "period_ns": period_ns,
        "compile_returncode": rc,
        "wns_ns": wns,
        "timing_ok": timing_ok,
        "report": str(report) if report else None,
        "log": str(log_path),
    }


def fpgatiming(start_period=2.0, max_period=10.0, resolution=0.05):
    coarse_periods = [2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
    coarse_periods = [p for p in coarse_periods if p >= start_period]

    if start_period not in coarse_periods:
        coarse_periods.insert(0, start_period)

    trials = []
    last_fail = None
    first_pass = None

    for period in coarse_periods:
        if period > max_period:
            break

        trial = _run_timing_trial(period)
        trials.append(trial)

        state = "PASS" if trial["timing_ok"] else "FAIL"
        print(
            f"Timing {state}: period={period:.4f} ns "
            f"WNS={trial['wns_ns']}"
        )

        if trial["timing_ok"]:
            first_pass = period
            break

        last_fail = period

    if first_pass is None:
        raise RuntimeError(
            f"No passing FPGA timing found through {max_period:.3f} ns"
        )

    best_period = first_pass
    best_trial = trials[-1]

    if last_fail is not None:
        low = last_fail
        high = first_pass

        while (high - low) > resolution:
            mid = (low + high) / 2.0
            trial = _run_timing_trial(mid)
            trials.append(trial)

            state = "PASS" if trial["timing_ok"] else "FAIL"
            print(
                f"Timing {state}: period={mid:.4f} ns "
                f"WNS={trial['wns_ns']}"
            )

            if trial["timing_ok"]:
                high = mid
                best_period = mid
                best_trial = trial
            else:
                low = mid

    result = {
        "period_ns": best_period,
        "frequency_mhz": 1000.0 / best_period,
        "wns_ns": best_trial["wns_ns"],
        "report": best_trial["report"],
        "log": best_trial["log"],
        "resolution_ns": resolution,
        "trials": trials,
    }

    print(
        f"\nLowest passing period: {best_period:.4f} ns "
        f"({result['frequency_mhz']:.2f} MHz)"
    )

    return result


def aved():
    if AVED.exists():
        if not (AVED / ".git").is_dir():
            raise RuntimeError(
                f"{AVED} exists but does not appear to be a git repository"
            )
        print(f"AVED repository already exists: {AVED}")
    else:
        run([
            "git",
            "clone",
            "https://github.com/hwspec/v80-aved-platform-.git",
            str(AVED),
        ])

        if not (AVED / ".git").is_dir():
            raise RuntimeError("AVED clone did not complete successfully")

    os.environ["AVED"] = str(AVED)
    print(f"AVED={AVED}")

    run(["make", "-C", "sw/AMI/api"], cwd=AVED)
    run(["make", "-C", "sw/runtime"], cwd=AVED)

    return {"aved_dir": str(AVED)}


def useracc():
    USER_ACCEL.mkdir(parents=True, exist_ok=True)

    files = []
    for pattern in ("*.v", "*.sv", "*.json"):
        files.extend(GENERATED.glob(pattern))

    if not files:
        raise RuntimeError(f"No generated RTL/JSON files found in {GENERATED}")

    copied = []

    for src in files:
        dst = USER_ACCEL / src.name
        print(f"copy {src} -> {dst}")
        shutil.copy2(src, dst)
        copied.append(str(dst))

    return {"copied": copied}


def firmware():
    log_path = ROOT / "aved-compile.log"
    build_dir = AVED_HW / "build"

    if build_dir.exists():
        print(f"Removing previous firmware build directory: {build_dir}")
        shutil.rmtree(build_dir)

    print(f"\n[{AVED_HW}] $ ./build_all.sh")
    print(f"Logging output to {log_path}")

    with log_path.open("w") as logfile:
        proc = subprocess.Popen(
            ["./build_all.sh"],
            cwd=AVED_HW,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        assert proc.stdout is not None

        for line in proc.stdout:
            sys.stdout.write(line)
            logfile.write(line)

        rc = proc.wait()

    if rc != 0:
        raise subprocess.CalledProcessError(rc, ["./build_all.sh"])

    print("Firmware compilation completed successfully.")
    return {"log": str(log_path)}


def program():
    prog_result = run(
        ["make", "prog"],
        cwd=AVED,
        ignore_returncode=True,
    )

    run([
        "sudo",
        "/usr/local/bin/ami_tool",
        "reload",
        "-t", "sbr",
        "-d", "b1:00.0",
    ])

    return {
        "make_prog_returncode": prog_result.returncode,
        "ami_reload": "success",
    }


def fpgatest(testname=None):
    env = venv_env()
    env["AVED"] = str(AVED)

    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{old_pythonpath}:{AVED}/sw/runtime/src"
        if old_pythonpath
        else f"{AVED}/sw/runtime/src"
    )

    env["LD_LIBRARY_PATH"] = (
        f"{AVED}/sw/runtime/build:"
        f"{AVED}/sw/AMI/api/build/"
    )

    env["PARAMFN"] = str(
        AVED_HW / "src/rtl/user_accel/params.json"
    )

    targets = [testname] if testname else TESTS
    if not targets:
        raise RuntimeError("No tests defined in .gwconfig (example: tests=basic,loop)")

    results = []

    for target in targets:
        print(f"\nRunning FPGA test target: {target}")
        run(
            ["sh", "run_on_fpga.sh", target],
            cwd=ROOT / "tests",
            env=env,
        )

        output_log = ROOT / "tests/output.log"
        if not output_log.is_file():
            raise RuntimeError(f"Expected output file not found: {output_log}")

        print(f"\nFPGA output ({target}): {output_log}")
        print("-" * 72)

        lines = output_log.read_text(errors="replace").splitlines()
        for line in lines[-30:]:
            print(line)

        results.append({
            "testname": target,
            "output_log": str(output_log),
        })

    return {"tests": results}


STAGES = {
    "prereq":     ([], prereq),
    "repo":       (["prereq"], repo),
    "bridge":     (["repo"], bridge),
    "rtlgen":     (["repo"], rtlgen),
    "cocotb":     (["rtlgen", "bridge"], cocotb),
    "fpgatiming": (["rtlgen"], fpgatiming),
    "aved":       (["prereq"], aved),
    "useracc":    (["rtlgen", "aved"], useracc),
    "firmware":   (["useracc"], firmware),
    "program":    (["firmware"], program),
    "fpgatest":       (["program"], fpgatest),
}


def _latest_attempt_succeeded(name):
    info = get_stage_info(name)
    if not info:
        return False

    last_attempt = info.get("last_attempt")
    return bool(last_attempt and last_attempt.get("success"))


def _run_direct_stage(name):
    _, func = STAGES[name]

    print()
    print("=" * 72)
    print(f"STAGE: {name}")
    print("=" * 72)

    set_stage_status(name, "running")
    start = time.monotonic()

    try:
        result = func()
        elapsed = time.monotonic() - start

        stage_results[name] = result
        stage_times[name] = elapsed

        record_stage_success(name, result, elapsed)
        set_stage_status(name, "success")
        return result

    except Exception as e:
        elapsed = time.monotonic() - start
        stage_times[name] = elapsed

        record_stage_failure(name, elapsed, str(e))
        set_stage_status(name, "failed")
        raise


def _run_stage_in_tmux(name):
    import shlex

    LOG_DIR.mkdir(exist_ok=True)

    session = f"gw-{name}"
    log_path = LOG_DIR / f"{name}.log"
    script = Path(__file__).resolve()

    existing = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if existing.returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session}")

    cmd = (
        f"{shlex.quote(sys.executable)} "
        f"{shlex.quote(str(script))} "
        f"--direct-stage {shlex.quote(name)} "
        f"2>&1 | tee {shlex.quote(str(log_path))}"
    )

    run([
        "tmux",
        "new-session",
        "-d",
        "-s", session,
        cmd,
    ])

    print(f"{name}: running in tmux session '{session}'")
    print(f"log: {log_path}")

    while True:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if result.returncode != 0:
            break

        time.sleep(2)

    if not _latest_attempt_succeeded(name):
        raise RuntimeError(f"{name} failed; see {log_path}")

    result = get_stage_result(name)
    stage_results[name] = result

    info = get_stage_info(name) or {}
    last_success = info.get("last_success", {})

    if "elapsed_sec" in last_success:
        stage_times[name] = last_success["elapsed_sec"]

    return result


def run_named_stage(name, stage_args=None):
    """Run exactly one named stage; do not run dependencies."""
    stage_args = stage_args or []

    if name in FPGA_STAGES and not FPGA_ENABLED:
        print(f"{name}: skipped (fpgatest=false in .gwconfig)")
        return {
            "skipped": True,
            "reason": "fpgatest=false in .gwconfig",
        }

    if name in TMUX_STAGES:
        if stage_args:
            raise RuntimeError(f"Stage {name} does not accept positional arguments")
        return _run_stage_in_tmux(name)

    if name == "fpgatest":
        if len(stage_args) > 1:
            raise RuntimeError("fpgatest accepts at most one test name")
        testname = stage_args[0] if stage_args else None
        return _run_direct_stage_with_args(name, testname)

    if stage_args:
        raise RuntimeError(f"Stage {name} does not accept positional arguments")

    return _run_direct_stage(name)


def _run_direct_stage_with_args(name, *args):
    _, func = STAGES[name]

    print()
    print("=" * 72)
    print(f"STAGE: {name}")
    print("=" * 72)

    set_stage_status(name, "running")
    start = time.monotonic()

    try:
        result = func(*args)
        elapsed = time.monotonic() - start

        stage_results[name] = result
        stage_times[name] = elapsed

        record_stage_success(name, result, elapsed)
        set_stage_status(name, "success")
        return result

    except Exception as e:
        elapsed = time.monotonic() - start
        stage_times[name] = elapsed

        record_stage_failure(name, elapsed, str(e))
        set_stage_status(name, "failed")
        raise


def resume_workflow(start_stage=None):
    """
    Continue the required workflow.

    If start_stage is specified, force-run that stage and every later required
    stage, regardless of previous success markers. fpgatiming remains optional
    and runs only when explicitly selected.

    If fpgatest=false in .gwconfig, FPGA-related stages are skipped and their
    .gwstatus entries are left unchanged.
    """
    if not FPGA_ENABLED:
        print(
            "fpgatest=false in .gwconfig; skipping: "
            + ", ".join(
                name for name in STAGE_ORDER
                if name in FPGA_STAGES
            )
        )

    if start_stage == "fpgatiming":
        run_named_stage("fpgatiming")
        return

    order = REQUIRED_STAGE_ORDER

    if start_stage is not None:
        if start_stage not in order:
            raise RuntimeError(f"Unknown required resume stage: {start_stage}")

        start_index = order.index(start_stage)

        for name in order[start_index:]:
            if name in FPGA_STAGES and not FPGA_ENABLED:
                print(f"{name}: skipped (fpgatest=false in .gwconfig)")
                continue

            print(f"\nResuming stage: {name}")
            run_named_stage(name)

        return

    first = next_runnable_stage()
    if first is None:
        print("All required stages are complete.")
        return

    start_index = order.index(first)

    for name in order[start_index:]:
        if name in FPGA_STAGES and not FPGA_ENABLED:
            print(f"{name}: skipped (fpgatest=false in .gwconfig)")
            continue

        status = load_status()

        if status.get(name) == "success":
            continue

        deps, _ = STAGES[name]
        required_deps = [dep for dep in deps if dep not in OPTIONAL_STAGES]

        missing = [
            dep for dep in required_deps
            if status.get(dep) != "success"
            and not (dep in FPGA_STAGES and not FPGA_ENABLED)
        ]
        if missing:
            raise RuntimeError(
                f"Cannot resume {name}: dependencies not successful: "
                + ", ".join(missing)
            )

        print(f"\nResuming stage: {name}")
        run_named_stage(name)


def print_timing_summary(total_elapsed):
    if not stage_times:
        return

    print()
    print("=" * 72)
    print("STAGE TIMING")
    print("=" * 72)

    for name in STAGE_ORDER:
        if name in stage_times:
            print(f"{name:12s} {stage_times[name]:10.1f} s")

    print("-" * 24)
    print(f"{'Total':12s} {total_elapsed:10.1f} s")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show editable stage status",
    )

    parser.add_argument(
        "--next",
        action="store_true",
        help="Show the next runnable stage",
    )

    parser.add_argument(
        "--resume",
        nargs="?",
        const="__auto__",
        choices=["__auto__"] + list(STAGES.keys()),
        metavar="STAGE",
        help=(
            "Resume required workflow from STAGE, or from the first unfinished "
            "required stage when STAGE is omitted"
        ),
    )

    parser.add_argument(
        "--direct-stage",
        choices=STAGES.keys(),
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "stage",
        nargs="?",
        default=None,
        choices=STAGES.keys(),
        help="Run exactly one stage (default: fpgatest)",
    )

    parser.add_argument(
        "stage_args",
        nargs="*",
        help="Optional arguments for the selected stage",
    )

    args = parser.parse_args()

    if not STATUS_FILE.exists():
        save_status(default_status())

    if args.status:
        show_status()
        return

    if args.next:
        show_next()
        return

    if args.direct_stage:
        try:
            _run_direct_stage(args.direct_stage)
        except subprocess.CalledProcessError as e:
            print(
                f"\nERROR: command failed with exit code {e.returncode}",
                file=sys.stderr,
            )
            sys.exit(e.returncode or 1)
        except Exception as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            sys.exit(1)
        return

    print(f"Project root: {ROOT}")
    total_start = time.monotonic()

    try:
        if args.resume is not None:
            if args.stage is not None or args.stage_args:
                raise RuntimeError(
                    "Use either '--resume [STAGE]' or a positional stage, not both"
                )
            start_stage = None if args.resume == "__auto__" else args.resume
            resume_workflow(start_stage)
        else:
            if args.stage is None:
                if args.stage_args:
                    raise RuntimeError(
                        "Stage arguments require an explicit stage name"
                    )
                resume_workflow()
            else:
                run_named_stage(args.stage, args.stage_args)
    except subprocess.CalledProcessError as e:
        total_elapsed = time.monotonic() - total_start
        print_timing_summary(total_elapsed)

        print(
            f"\nERROR: command failed with exit code {e.returncode}",
            file=sys.stderr,
        )
        sys.exit(e.returncode or 1)
    except Exception as e:
        total_elapsed = time.monotonic() - total_start
        print_timing_summary(total_elapsed)

        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    total_elapsed = time.monotonic() - total_start
    print_timing_summary(total_elapsed)


if __name__ == "__main__":
    main()
