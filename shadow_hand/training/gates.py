from __future__ import annotations

Check = tuple[str, float, str]
Milestone = tuple[int, list[Check], str]

GRASP_GATES: list[Milestone] = [
    (
        10_000_000,
        [
            ("metrics/num_finger_contacts", 0.83, "grip forms and stays formed"),
            ("reward/grasping", 0.24, "grasp reward maintained"),
        ],
        "grasp 10M: grip health",
    ),
    (
        30_000_000,
        [
            ("reward/lifting", 0.20, "lifting still improving"),
            (
                "metrics/success_hold_steps",
                0.05,
                "the cube is held at target for consecutive steps",
            ),
        ],
        "grasp 30M: lift emergence",
    ),
]

PEG_GATES: list[Milestone] = [
    (
        10_000_000,
        [
            ("reward/place", 0.20, "carrying or inserting"),
            ("metrics/stage", 1.0, "task progressed past grasp-and-sit"),
        ],
        "peg 10M: ladder acquired",
    ),
    (
        30_000_000,
        [
            ("reward/place", 0.40, "peg reaching the socket"),
            ("metrics/insertion_depth", 0.001, "first in-bore insertion"),
        ],
        "peg 30M: carry consolidated and insertion starting",
    ),
    (
        40_000_000,
        [
            ("metrics/insertion_depth", 0.003, "in-bore insertion happening"),
            ("reward/place_release", 0.03, "engaged-release occurring in-bore"),
            ("reward/success", 0.05, "some insertion succeeding"),
        ],
        "peg 40M: terminal engaged-release exists",
    ),
]

PICKPLACE_GATES: list[Milestone] = [
    (
        10_000_000,
        [
            ("metrics/num_finger_contacts", 0.50, "grip forms"),
            ("reward/lifting", 0.10, "cube is being lifted off the table"),
        ],
        "pickplace 10M: grip + lift",
    ),
    (
        30_000_000,
        [
            ("reward/placed", 0.30, "cube is set down near the goal"),
            ("reward/success", 0.30, "goal placements are being held"),
        ],
        "pickplace 30M: place + success emergence",
    ),
]
