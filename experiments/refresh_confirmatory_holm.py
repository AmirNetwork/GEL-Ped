# Author: Amir Ghorbani
"""Refresh Holm adjustments for the five prespecified confirmatory comparisons."""

from __future__ import annotations

import json
from pathlib import Path


CONFIRMATORY = (
    "constant_velocity",
    "social_force",
    "unrestricted_ten_scalar",
    "interaction_mlp",
    "goal_stable_neural",
)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    target = project / "data" / "processed" / "small_sample_and_contact_statistics.json"
    data = json.loads(target.read_text())
    for family in data.values():
        if not isinstance(family, dict) or not all(name in family for name in CONFIRMATORY):
            continue
        for result in family.values():
            if isinstance(result, dict):
                result.pop("holm_adjusted_randomization_p", None)
        ordered = sorted(
            CONFIRMATORY,
            key=lambda name: family[name]["exact_paired_randomization_p"],
        )
        running_max = 0.0
        for rank, name in enumerate(ordered):
            raw = family[name]["exact_paired_randomization_p"]
            candidate = min(1.0, (len(ordered) - rank) * raw)
            running_max = max(running_max, candidate)
            family[name]["holm_adjusted_randomization_p"] = running_max
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
