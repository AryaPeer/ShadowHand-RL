from __future__ import annotations

import sys
from collections.abc import Callable

Command = Callable[[list[str]], None]


def _train_grasp(argv: list[str]) -> None:
    from shadow_hand.training.grasp import train

    train(argv)


def _train_peg(argv: list[str]) -> None:
    from shadow_hand.training.peg import train

    train(argv)


def _train_pickplace(argv: list[str]) -> None:
    from shadow_hand.training.pickplace import train

    train(argv)


def _resume_grasp(argv: list[str]) -> None:
    from shadow_hand.training.grasp import resume

    resume(argv)


def _resume_peg(argv: list[str]) -> None:
    from shadow_hand.training.peg import resume

    resume(argv)


def _resume_pickplace(argv: list[str]) -> None:
    from shadow_hand.training.pickplace import resume

    resume(argv)


def _evaluate(argv: list[str]) -> None:
    from shadow_hand.evaluation.evaluate import main

    main(argv)


def _render(argv: list[str]) -> None:
    from shadow_hand.evaluation.render import main

    main(argv)


COMMANDS: dict[str, tuple[Command, str]] = {
    "train grasp": (_train_grasp, "Train grasping"),
    "train peg": (_train_peg, "Train peg-in-hole"),
    "train pickplace": (_train_pickplace, "Train pick-and-place"),
    "resume grasp": (_resume_grasp, "Resume grasping from a checkpoint"),
    "resume peg": (_resume_peg, "Resume peg-in-hole from a checkpoint"),
    "resume pickplace": (_resume_pickplace, "Resume pick-and-place from a checkpoint"),
    "eval": (_evaluate, "Evaluate a checkpoint deterministically"),
    "render": (_render, "Render a deterministic rollout to mp4"),
}


def usage() -> str:
    width = max(len(name) for name in COMMANDS)
    body = "\n".join(f"  {name:<{width}}  {help_text}" for name, (_, help_text) in COMMANDS.items())
    return f"Usage: shadow-hand <command> [options]\n\n{body}"


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    for n in (2, 1):
        command = COMMANDS.get(" ".join(args[:n]))
        if command is not None:
            command[0](args[n:])
            return

    print(usage())
    if args[:1] not in (["-h"], ["--help"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
