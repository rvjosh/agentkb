"""Create the private deterministic benchmark snapshot under /tmp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentkb.config import paths

from experiments.modal_benchmark.common import write_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_snapshot(paths.wiki_dir(), args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
