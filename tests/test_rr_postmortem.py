"""Death attribution from protocol lines (pure; no engine, no Radical Red fork needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rr_postmortem import attribution  # noqa: E402


def test_attribution_counts_killers_and_kos():
    history = [
        "|switch|p1a: Jynx|Jynx, L28, F|100/100",
        "|switch|p2a: Politoed|Politoed, L27, M|100/100",
        "|move|p1a: Jynx|Ice Beam|p2a: Politoed",
        "|faint|p2a: Politoed",
        "|switch|p2a: Mantine|Mantine, L27, F|100/100",
        "|move|p2a: Mantine|Hurricane|p1a: Jynx",
        "|faint|p1a: Jynx",
        "|switch|p1a: Throh|Throh, L28, M|100/100",
        "|move|p2a: Mantine|Scald|p1a: Throh",
        "|faint|p1a: Throh",
    ]
    deaths_by, kos_on, killed = attribution(history)
    assert deaths_by == {"Mantine": 2} and kos_on == {"Politoed": 1} and killed == {"Jynx": 1, "Throh": 1}
