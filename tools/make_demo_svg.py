"""Render assets/demo.svg from real unleak output: python tools/make_demo_svg.py (needs rich)."""

import pathlib
import subprocess
import sys

from rich.console import Console
from rich.text import Text

ROOT = pathlib.Path(__file__).resolve().parents[1]
cmd = ["python3", "skills/leakage-audit/scripts/run_unleak.py", "check", "examples/demo_churn.py", "--color", "always"]
out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
console = Console(record=True, width=124, force_terminal=True, color_system="truecolor")
console.print(Text.from_markup("[bold green]$[/] [bold]unleak check examples/demo_churn.py[/]"))
console.print(Text.from_ansi(out.rstrip("\n")))
console.save_svg(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "assets" / "demo.svg"), title="unleak")
