from __future__ import annotations

from datetime import datetime
from pathlib import Path
from re import S, compile, search, subn
from shutil import copy2, move
from subprocess import CompletedProcess, TimeoutExpired, run
from time import sleep

START = 0
END = 100000
STEP = 1000
REPEATS = 3
LOG = "inline_sweep.log"
TIMEOUT = 1800

CONFIG_PATH = Path(".cargo/config.toml")
BACKUP_PATH = Path(".cargo/config.toml.bak")
RUSTFLAGS_ENTRY = r"-Cllvm-args=--inline-threshold="

RUNTIME_RE = compile(r"Total elapsed time:\s*([0-9]+(?:\.[0-9]+)?)s")
ALT_RUNTIME_RE = compile(r"Total compute time:\s*([0-9]+(?:\.[0-9]+)?)s")


def backup_config() -> None:
    """Backup existing config file, if any."""
    if CONFIG_PATH.exists():
        copy2(CONFIG_PATH, BACKUP_PATH)


def restore_config() -> None:
    """Restore config file from backup, if any."""
    if BACKUP_PATH.exists():
        move(BACKUP_PATH, CONFIG_PATH)


def set_inline_threshold_in_config(value: int) -> bool:
    """Replace existing threshold value or insert one into rustflags.
    Returns True if the file was modified, False otherwise.
    """
    content = (
        CONFIG_PATH.read_text()
        if CONFIG_PATH.exists()
        else "[build]\nrustflags = [\n]\n"
    )

    def repl(m):
        return f"{m.group(1)}{value}"

    new_content, n = subn(
        r"(--inline-threshold=)(\d+)",
        repl,
        content,
    )

    if n > 0:
        CONFIG_PATH.write_text(new_content)
        print(f"Replaced existing inline threshold -> {value}")
        return True

    match_search = search(r"(rustflags\s*=\s*\[\s*)(.*?)\s*(\])", content, S)

    if match_search:
        pre = content[: match_search.start(2)]
        inner = match_search.group(2).rstrip()
        post = content[match_search.end(2) :]
        entry = f'\n    "{RUSTFLAGS_ENTRY}{value}",'
        new_inner = inner + entry
        new_content = pre + new_inner + post
        CONFIG_PATH.write_text(new_content)
        print(f"Inserted inline threshold entry into existing rustflags -> {value}")
        return True

    content += (
        f'\n[build]\nrustflags = [\n    "{RUSTFLAGS_ENTRY}{value}",\n]\n'
    )

    CONFIG_PATH.write_text(content)
    print(f"Appended new [build] section with inline threshold -> {value}")
    return True


def _normalize_proc_output(s: bytes | str | None) -> str:
    """Normalize subprocess output to str for safe concatenation."""
    if isinstance(s, bytes):
        return s.decode(errors="replace")

    return s or ""


def run_once(timeout: int = TIMEOUT) -> tuple[str, int]:
    """Run the exact requested shell command and return combined output
    and returncode. Executes
    `cargo clean; cargo fmt; echo -e '\ny' | RUST_BACKTRACE=1 cargo run --release`
    via bash -lc so quoting/echo behavior matches the shell.
    """
    cmd = "cargo clean; cargo fmt; echo -e '\\ny' | RUST_BACKTRACE=1 cargo run --release"
    proc: CompletedProcess[str] = run(
        ["bash", "-lc", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    out = _normalize_proc_output(proc.stdout) + "\n" + _normalize_proc_output(proc.stderr)
    return out, proc.returncode


def parse_runtime(output: str) -> float | None:
    """Parse runtime in seconds from output string using the Rust print line."""
    match = RUNTIME_RE.search(output)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    match_alt = ALT_RUNTIME_RE.search(output)
    if match_alt:
        try:
            return float(match_alt.group(1))
        except ValueError:
            return None

    return None


def main() -> None:
    """Main entry point using module-level defaults (no argparse)."""
    log_path = Path(LOG)

    # ensure log exists (no CSV header)
    if not log_path.exists():
        log_path.write_text("")

    try:
        backup_config()
        thresholds = list(range(START, END + 1, STEP))

        for thr in thresholds:
            print(f"[{datetime.now().isoformat()}] Testing threshold {thr} ...")

            set_inline_threshold_in_config(thr)

            for rep in range(1, REPEATS + 1):
                print(f"  repeat {rep}/{REPEATS} ...", end=" ", flush=True)

                try:
                    out, rc = run_once(timeout=TIMEOUT)

                except TimeoutExpired as e:
                    out = (
                        _normalize_proc_output(getattr(e, "stdout", None))
                        + "\n"
                        + _normalize_proc_output(getattr(e, "stderr", None))
                    )
                    rc = -1
                    print("timeout")

                else:
                    print(f"done (rc={rc})")

                # Append raw output exactly as received, then add the required metadata lines
                with log_path.open("a") as f:
                    f.write(out)
                    if not out.endswith("\n"):
                        f.write("\n")
                    # append the inline threshold and repeat lines, then two blank lines
                    f.write(f"inline_threshold={thr}\n")
                    f.write(f"repeat={rep}\n\n")

                sleep(0.1)
    finally:
        restore_config()
        print("Restored original config (if present). Log written to", log_path)


if __name__ == "__main__":
    main()
