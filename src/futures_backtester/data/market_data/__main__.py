"""Module entry point for the real-data futures builder."""

from __future__ import annotations

import argparse
from pathlib import Path

from .build import build


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the real-data futures panel.")
    parser.add_argument("--out", type=Path, default=Path("data/real"), help="Output directory.")
    parser.add_argument("--end", default=None, help="Last observation date, inclusive.")
    args = parser.parse_args()
    build(args.out, end=args.end)


if __name__ == "__main__":
    main()
