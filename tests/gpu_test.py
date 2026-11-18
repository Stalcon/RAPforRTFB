"""Run a water calculation to check GPU inference and the DFT workflow."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="calcs/smoke_workspace")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-mb", type=int, default=4000)
    parser.add_argument("--thermal", action="store_true")
    parser.add_argument("--skip-uma", action="store_true", help="Start DFT from the prepared geometry.")
    args = parser.parse_args()

    # Set the workspace and thread limits before importing the calculation modules.
    os.environ["RAP_ROOT"] = str(Path(args.root).resolve())
    from core.backend import configure_threads, describe

    configure_threads(args.threads, args.threads)
    from core.cli import prepare
    from core.paths import ROOT, atomic_write_json
    from core.modeling import dft, sp, uma

    start = time.monotonic()
    summary = {"scope": "software smoke test, not research validation"}
    try:
        summary["backend"] = describe()
        print(json.dumps(summary, indent=2), flush=True)
        rows = prepare("O", "water smoke control", conformers=1)["states"]
        selection = {"ids": [rows[0]["id"]], "states": ["rest"]}
        resources = {"threads": args.threads, "memory_mb": args.memory_mb}

        stages = []
        if not args.skip_uma:
            stages.append(("uma", partial(uma.run_batch, **selection, steps=150)))
        stages.append(("dft", partial(
            dft.run_batch, **selection, **resources,
            from_library=args.skip_uma, thermal=args.thermal,
        )))
        stages.append(("sp", partial(sp.run_batch, **selection, **resources)))

        for name, run_stage in stages:
            stage_start = time.monotonic()
            outputs = run_stage()
            summary[name] = {"outputs": outputs, "seconds": time.monotonic() - stage_start}
            if any(row["status"] != "complete" for row in outputs):
                raise RuntimeError(f"{name} failed. Check its last_failure.json.")
        summary["status"] = "passed"
    except Exception as exc:
        summary.update(status="failed", error=str(exc))
    finally:
        summary["elapsed_seconds"] = time.monotonic() - start
        atomic_write_json(ROOT / "smoke_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)

    return int(summary["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
