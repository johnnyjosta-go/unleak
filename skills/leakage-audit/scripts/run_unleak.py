#!/usr/bin/env python3
"""Run unleak straight from the skill folder: no pip install needed.

    python3 scripts/run_unleak.py check path/to/code
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unleak.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
