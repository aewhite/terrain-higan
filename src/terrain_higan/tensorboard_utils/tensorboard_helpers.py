import glob
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from terrain_higan.paths import DEFAULT_RUNS_DIR, setting

# Marker file name to indicate a valid tensorboard run directory
VALID_RUN_MARKER_FILE = ".tensorboard_valid"

# Marker file name to indicate a training run completed successfully
TRAINING_COMPLETE_MARKER_FILE = ".tensorboard_training_complete"

# Maximum number of TensorBoard run directories to keep (newest first)
MAX_RUN_DIRECTORIES = 50

# Never prune a run that has been written to within this period.
MIN_RUN_AGE_SECONDS = 7 * 24 * 60 * 60


def _log_run_decision(phase: str, run_dir_path: str, action: str, **evidence: object) -> None:
    evidence_text = " ".join(f"{name}={value}" for name, value in evidence.items())
    print(
        f"TensorBoard run decision phase={phase} path={run_dir_path} "
        f"action={action} {evidence_text}"
    )


def _latest_run_mtime(run_dir_path: str) -> float:
    latest_mtime = os.path.getmtime(run_dir_path)
    for child in os.scandir(run_dir_path):
        latest_mtime = max(latest_mtime, child.stat().st_mtime)
    return latest_mtime


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return False
        except OSError:
            return True


def is_tensorboard_running(port: int = 6006) -> bool:
    """Check if TensorBoard is already running on the specified port."""
    if not is_port_in_use(port):
        return False

    # Check if the port is actually being used by TensorBoard
    try:
        # Try to check running processes
        result = subprocess.run(
            ['pgrep', '-f', f'tensorboard.*{port}'],
            capture_output=True,
            text=True,
            timeout=2
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # If pgrep is not available or times out, assume it might be TensorBoard
        # since the port is in use
        return True


def cleanup_invalid_run_directories(log_dir: str) -> None:
    """Remove tensorboard run directories that don't have the valid marker file.

    Args:
        log_dir: Directory where TensorBoard logs are stored
    """
    if not os.path.exists(log_dir):
        return

    # Pattern for directory names: <name>_<YYYYMMDD>_<HHMMSS>[_<microseconds>]
    # e.g., generator_20251114_070358 or z1_20251114_070358_123456
    run_dir_pattern = re.compile(r'^.+_\d{8}_\d{6}(?:_\d{6})?$')

    removed_count = 0
    for entry in os.scandir(log_dir):
        if not entry.is_dir():
            continue

        run_dir_path = entry.path
        pattern_matches = bool(run_dir_pattern.match(entry.name))
        marker_file_path = os.path.join(run_dir_path, VALID_RUN_MARKER_FILE)
        has_valid_marker = os.path.exists(marker_file_path)
        event_files = glob.glob(os.path.join(run_dir_path, "events.out.tfevents.*"))
        event_file_count = len(event_files)
        if not pattern_matches:
            _log_run_decision(
                "invalid_cleanup",
                run_dir_path,
                "retain",
                timestamp_pattern_matches=False,
                valid_marker=has_valid_marker,
                event_file_count=event_file_count,
                reason="name_does_not_match_run_pattern",
            )
            continue
        if has_valid_marker:
            _log_run_decision(
                "invalid_cleanup",
                run_dir_path,
                "retain",
                timestamp_pattern_matches=True,
                valid_marker=True,
                event_file_count=event_file_count,
                reason="valid_marker_present",
            )
            continue
        if event_file_count == 0:
            _log_run_decision(
                "invalid_cleanup",
                run_dir_path,
                "retain",
                timestamp_pattern_matches=True,
                valid_marker=False,
                event_file_count=0,
                reason="no_event_file",
            )
            continue

        _log_run_decision(
            "invalid_cleanup",
            run_dir_path,
            "delete",
            timestamp_pattern_matches=True,
            valid_marker=False,
            event_file_count=event_file_count,
            reason="event_file_without_valid_marker",
        )

        try:
            shutil.rmtree(run_dir_path)
            print(
                f"TensorBoard run deleted phase=invalid_cleanup path={run_dir_path} "
                "reason=event_file_without_valid_marker"
            )
            removed_count += 1
        except Exception as e:
            print(
                f"TensorBoard run deletion_failed phase=invalid_cleanup path={run_dir_path} "
                f"error={e}"
            )

    if removed_count > 0:
        print(f"Cleaned up {removed_count} invalid tensorboard run director{'y' if removed_count == 1 else 'ies'}")


def prune_old_run_directories(log_dir: str) -> None:
    """Prune excess run directories only after they are old enough.

    The newest MAX_RUN_DIRECTORIES are retained by timestamp. Excess runs are
    retained too when the run directory or one of its direct files was modified
    within MIN_RUN_AGE_SECONDS.
    """
    if not os.path.exists(log_dir):
        return

    # Pattern for directory names: <name>_<YYYYMMDD>_<HHMMSS>[_<microseconds>]
    run_dir_pattern = re.compile(r'^.+_(\d{8})_(\d{6})(?:_(\d{6}))?$')

    runs_with_ts: list[tuple[str, str, str, str]] = []
    for entry in os.scandir(log_dir):
        if not entry.is_dir():
            continue

        m = run_dir_pattern.match(entry.name)
        if not m:
            _log_run_decision(
                "age_pruning",
                entry.path,
                "retain",
                timestamp_pattern_matches=False,
                valid_marker=os.path.exists(os.path.join(entry.path, VALID_RUN_MARKER_FILE)),
                reason="name_does_not_match_run_pattern",
            )
            continue

        date_part, time_part, microseconds_part = m.groups()
        runs_with_ts.append(
            (entry.path, date_part, time_part, microseconds_part or "000000")
        )

    # Sort newest first by date then time (both zero-padded numeric strings)
    runs_with_ts.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)

    # Keep the newest runs, then remove only excess runs old enough to be stale.
    cutoff_time = time.time() - MIN_RUN_AGE_SECONDS
    for rank, (run_dir_path, date_part, time_part, microseconds_part) in enumerate(
        runs_with_ts,
        start=1,
    ):
        has_valid_marker = os.path.exists(os.path.join(run_dir_path, VALID_RUN_MARKER_FILE))
        timestamp = f"{date_part}_{time_part}_{microseconds_part}"
        if rank <= MAX_RUN_DIRECTORIES:
            _log_run_decision(
                "age_pruning",
                run_dir_path,
                "retain",
                timestamp_pattern_matches=True,
                timestamp=timestamp,
                rank=rank,
                valid_marker=has_valid_marker,
                reason="within_newest_run_limit",
            )
            continue

        latest_mtime = _latest_run_mtime(run_dir_path)
        age_seconds = max(0.0, time.time() - latest_mtime)
        if latest_mtime >= cutoff_time:
            _log_run_decision(
                "age_pruning",
                run_dir_path,
                "retain",
                timestamp_pattern_matches=True,
                timestamp=timestamp,
                rank=rank,
                valid_marker=has_valid_marker,
                latest_modified=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(latest_mtime)),
                age_seconds=f"{age_seconds:.1f}",
                minimum_age_seconds=MIN_RUN_AGE_SECONDS,
                reason="recent_activity",
            )
            continue

        _log_run_decision(
            "age_pruning",
            run_dir_path,
            "delete",
            timestamp_pattern_matches=True,
            timestamp=timestamp,
            rank=rank,
            valid_marker=has_valid_marker,
            latest_modified=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(latest_mtime)),
            age_seconds=f"{age_seconds:.1f}",
            minimum_age_seconds=MIN_RUN_AGE_SECONDS,
            reason="outside_newest_run_limit_and_stale",
        )
        try:
            shutil.rmtree(run_dir_path)
            print(
                f"TensorBoard run deleted phase=age_pruning path={run_dir_path} "
                "reason=outside_newest_run_limit_and_stale"
            )
        except Exception as e:
            print(
                f"TensorBoard run deletion_failed phase=age_pruning path={run_dir_path} "
                f"error={e}"
            )


def start_tensorboard_detached(
    log_dir: str | Path = DEFAULT_RUNS_DIR,
    port: int | None = None,
    host: str | None = None,
) -> None:
    """Start TensorBoard as a detached service if it's not already running.

    Args:
        log_dir: Directory where TensorBoard logs are stored
        port: Port to bind TensorBoard to (default: 6006)
        host: Host to bind to (default: loopback)
    """
    port = int(setting("TENSORBOARD_PORT", 6006)) if port is None else port
    host = setting("TENSORBOARD_HOST", "127.0.0.1") if host is None else host
    access_host = "<your-machine-ip>" if host == "0.0.0.0" else host
    log_dir = os.fspath(log_dir)
    resolved_log_dir = os.path.abspath(log_dir)
    print(
        "TensorBoard run maintenance starting "
        f"source={__file__} log_dir={resolved_log_dir} "
        f"log_dir_exists={os.path.isdir(resolved_log_dir)}"
    )

    # Clean up invalid run directories on startup
    cleanup_invalid_run_directories(log_dir)
    # Then prune excess timestamped directories once they are old enough.
    prune_old_run_directories(log_dir)

    if is_tensorboard_running(port):
        print(f"TensorBoard is already running on port {port}")
        print(f"Access it at http://{access_host}:{port}")
        return

    print(f"Starting TensorBoard on {host}:{port} (log_dir: {log_dir})...")

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Start TensorBoard as a detached process and keep startup logs on disk
    cmd = [
        sys.executable, '-m', 'tensorboard.main',
        '--logdir', log_dir,
        '--port', str(port),
        '--host', host,
        '--reload_interval', '5'
    ]

    try:
        tensorboard_log_path = os.path.join(log_dir, "tensorboard.log")
        with open(tensorboard_log_path, "a", encoding="utf-8") as tensorboard_log_file:
            tensorboard_log_file.write(
                f"\n\n===== TensorBoard launch attempt {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
            )
            tensorboard_log_file.write("Command: " + " ".join(cmd) + "\n")
            tensorboard_log_file.flush()

            process = subprocess.Popen(
                cmd,
                stdout=tensorboard_log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

            # Give it a moment to start
            time.sleep(2)

            # Check if it's still running
            if process.poll() is None:
                print(f"TensorBoard started successfully (PID: {process.pid})")
                print(f"TensorBoard logs: {tensorboard_log_path}")
                print(f"Access TensorBoard at http://{access_host}:{port}")
            else:
                exit_code = process.returncode
                print("Warning: TensorBoard process exited immediately. It may have failed to start.")
                print(f"Exit code: {exit_code}")
                print(f"Check startup logs: {tensorboard_log_path}")
                try:
                    with open(tensorboard_log_path, "r", encoding="utf-8") as f:
                        log_lines = f.readlines()
                    tail_lines = "".join(log_lines[-20:]).rstrip()
                    if tail_lines:
                        print("Last TensorBoard log lines:")
                        print(tail_lines)
                except Exception as log_read_error:
                    print(f"Could not read TensorBoard log file: {log_read_error}")
    except FileNotFoundError:
        print("Error: tensorboard command not found. Make sure TensorBoard is installed:")
        print("  pip install tensorboard")
    except Exception as e:
        print(f"Error starting TensorBoard: {e}")
