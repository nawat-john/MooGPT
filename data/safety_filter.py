#!/usr/bin/env python3
"""Phase 5 safety pass: drop any dialogue line containing a blocked keyword.

Even after controlled generation we run a final keyword filter over the persona corpus so
nothing rude/unsafe slips into training. This is intentionally simple and conservative —
it operates line-by-line (one dialogue per line, per data/PERSONA.md) and removes the whole
line on any match. Extend BLOCKLIST as needed.

Usage:
    python data/safety_filter.py data/persona.txt                 # -> data/persona.clean.txt
    python data/safety_filter.py data/persona.txt data/out.txt    # explicit output
    python data/safety_filter.py --check data/persona.txt         # report only, no write
"""
import argparse
import re
import sys

# Keep this conservative and child-appropriate. Word-boundary matched, case-insensitive.
BLOCKLIST = [
    # profanity / slurs (kept short here; expand for a real run)
    "damn", "hell", "crap", "stupid", "idiot", "shut up", "hate you",
    # violence / danger
    "kill", "die", "blood", "gun", "knife", "hurt you", "punch", "fight",
    # adult / unsafe themes
    "sex", "drug", "drunk", "beer", "wine", "cigarette", "naked",
    # scary
    "scary monster", "nightmare", "demon", "ghost will get",
]

_PATTERNS = [re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE) for w in BLOCKLIST]


def is_blocked(line: str):
    for pat, word in zip(_PATTERNS, BLOCKLIST):
        if pat.search(line):
            return word
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--check", action="store_true", help="report only; do not write")
    args = ap.parse_args()

    kept, dropped = [], []
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            stripped = line.rstrip("\n")
            if not stripped.strip():
                continue
            hit = is_blocked(stripped)
            if hit:
                dropped.append((i, hit, stripped))
            else:
                kept.append(stripped)

    print(f"scanned {len(kept) + len(dropped)} dialogues: "
          f"{len(kept)} kept, {len(dropped)} dropped")
    for ln, word, text in dropped[:20]:
        print(f"  drop line {ln} [{word}]: {text[:80]}")

    if args.check:
        return 1 if dropped else 0

    out = args.output or args.input.rsplit(".", 1)[0] + ".clean.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
