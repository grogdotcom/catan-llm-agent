#!/usr/bin/env python3
"""SFT checkpoint batch runner (wrapper around catan_llm.llm.sft.cli)."""

import sys

sys.path.insert(0, "src")
from catan_llm.llm.sft.cli import main

if __name__ == "__main__":
    main()
