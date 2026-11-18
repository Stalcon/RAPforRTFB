"""Serial, resumable manifest execution; one process owns each state."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import traceback
from core.paths import atomic_write_json, read_manifest, safe_name


@contextmanager
def state_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process owns {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_rows(worker, root, *, ids=None, states=None, limit=None, **kwargs):
    rows = [r for r in read_manifest() if (not ids or r["id"] in ids)
            and (not states or r["state"] in states)]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError("no matching manifest rows; prepare a library first")
    outcomes = []
    for row in rows:
        directory = Path(root) / safe_name(row["id"]) / safe_name(row["state"])
        try:
            with state_lock(directory):
                atomic_write_json(directory / 'status.json', {'status': 'running'})
                record = worker(row, directory, **kwargs)
                atomic_write_json(directory / 'status.json', {'status': record['status']})
            outcome = {"id": row["id"], "state": row["state"], "status": record["status"]}
        except Exception as exc:
            outcome = {"id": row["id"], "state": row["state"], "status": "failed",
                       "error": str(exc)}
            # A failure receipt does not overwrite an earlier scientific record.
            atomic_write_json(directory / "last_failure.json",
                              {**outcome, "traceback": traceback.format_exc()})
            if not isinstance(exc, BlockingIOError) and 'another process owns' not in str(exc):
                atomic_write_json(directory / 'status.json', outcome)
        outcomes.append(outcome)
        print(json.dumps(outcome), flush=True)
    return outcomes
