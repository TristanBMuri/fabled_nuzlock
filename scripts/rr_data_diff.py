"""TODO.md A3: the ROM dump vs the tracker sheet, boss by boss (hardcore).

The sim plays the dump's sets (rr_probe.boss_team); the tracker (official sheet export) is
the other record of the same fights. Per trainer-order entry up to --upto: species
missing on either side, then per shared species the fields that differ (level, item,
ability, nature, moves). Names go through the same resolvers the sim uses (dump_id,
tracker_id), so a row is a real disagreement, not spelling. Relative levels resolve
against the cap on both sides; the sheet's mega forms compare as the base species (the dump
gives the base species and the stone).

  uv run python scripts/rr_data_diff.py --upto 34 > results/rr_data_diff.log
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rr_probe as rp  # noqa: E402
from poke_env.data import to_id_str  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upto", type=int, default=34, help="last trainer-order index (34 = Erika)")
    args = ap.parse_args()
    order = json.load(open(rp.RR / "order.json"))["entries"]
    fields = {}
    n_mons = n_same = 0
    for t, e in enumerate(order[:args.upto + 1]):
        block = rp.block_of(e)
        try:
            packed, cap, _ = rp.boss_team(t)
        except KeyError:
            packed = None
        tag = f"#{t:<3} {e['name']:<20} cap {e['cap']}{' (optional)' if e['optional'] else ''}"
        if block is None or packed is None:
            print(f"{tag}: {'no tracker block' if block is None else ''} {'no dump trainer' if packed is None else ''}")
            continue
        dump = {}
        for m in packed.split("]"):
            f = m.split("|")
            dump[to_id_str(f[0])] = {"level": int(f[10]), "item": f[2], "ability": f[3], "nature": f[5].lower(),
                                     "moves": sorted(f[4].split(","))}
        sheet = {}
        for p in block["pokemon"]:
            sheet[rp.tracker_id(p["species"], "species").removesuffix("mega")] = {  # the dump holds the stone, the sheet the form
                "level": rp.tracker_level(p["level"], e["cap"]), "item": rp.tracker_id(p["item"], "items"),
                "ability": rp.tracker_id(p["ability"], "abilities"), "nature": p["nature"].lower(),
                "moves": sorted(rp.tracker_id(m, "moves") for m in p["moves"])}
        rows = []
        only_d, only_s = sorted(set(dump) - set(sheet)), sorted(set(sheet) - set(dump))
        if only_d or only_s:
            rows.append(f"species: dump only {only_d or '-'}, sheet only {only_s or '-'}")
        for sp in sorted(set(dump) & set(sheet)):
            n_mons += 1
            diff = [k for k in ("level", "item", "ability", "nature", "moves") if dump[sp][k] != sheet[sp][k]]
            n_same += not diff
            for k in diff:
                fields[k] = fields.get(k, 0) + 1
                a, b = dump[sp][k], sheet[sp][k]
                if k == "moves":
                    a, b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
                rows.append(f"{sp:<14} {k:<8} dump {a}  sheet {b}")
        print(f"{tag}: {'agree' if not rows else ''}")
        for r in rows:
            print(f"    {r}")
    print(f"\n{n_same}/{n_mons} shared mons identical; fields that differ: {fields}")


if __name__ == "__main__":
    main()
