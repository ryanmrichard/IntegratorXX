#!/usr/bin/env python3
"""Retype the tabulated solid-angle grids from `T` storage to `double` storage.

The s2 tables are declared `static constexpr`, which requires a literal type.
Types this library should support -- interval arithmetic, uncertainty-
propagating scalars -- are not literal, so a table templated on `T` cannot be
instantiated over them. Storing the tables as `double` and converting on read
(see util/copy_grid.hpp) keeps them usable from constant expressions while
leaving the quadratures type-generic.

The surrounding `template <typename T>` is deliberately left in place: `T` is
unused by the table, but keeping it means none of the ~30 dispatch branches in
each of the four s2 family headers has to change.

Re-run after syncing tables from upstream. Idempotent.
"""
import argparse
import pathlib
import re
import sys

FAMILIES = ("lebedev_laikov", "delley", "ahrens_beylkin", "womersley")

POINTS = re.compile(r"static constexpr std::array<cartesian_pt_t<T>,(\s*)(\d+)> points")
WEIGHTS = re.compile(r"static constexpr std::array<T,(\s*)(\d+)> weights")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="include/integratorxx/quadratures/s2",
                    help="directory holding the s2 family subdirectories")
    ap.add_argument("--check", action="store_true",
                    help="report what would change; do not write")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.is_dir():
        print(f"error: no such directory: {root}", file=sys.stderr)
        return 2

    files = sorted(p for fam in FAMILIES for p in (root / fam).glob("*.hpp")
                   if not p.name.endswith("_grids.hpp"))
    if not files:
        print(f"error: no table files found under {root}", file=sys.stderr)
        return 2

    changed = []
    for path in files:
        src = path.read_text()
        out = WEIGHTS.sub(r"static constexpr std::array<double,\1\2> weights",
                          POINTS.sub(r"static constexpr std::array<cartesian_pt_t<double>,\1\2> points", src))
        if out != src:
            changed.append(path)
            if not args.check:
                path.write_text(out)

    verb = "would retype" if args.check else "retyped"
    print(f"{verb} {len(changed)} of {len(files)} table files")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
