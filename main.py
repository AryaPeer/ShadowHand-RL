import sys

USAGE = """\
Usage: python main.py <command> [options]

  train-grasp-mjx        Train grasping (SBX PPO on MJX)
  train-peg-mjx          Train peg-in-hole (SBX PPO on MJX)
  train-pickplace-mjx    Train pick-and-place (SBX PPO on MJX)
  resume-grasp-mjx       Resume grasping from a saved checkpoint
  resume-peg-mjx         Resume peg-in-hole from a saved checkpoint
"""

COMMANDS = {
    "train-grasp-mjx": "scripts.training.train_grasp",
    "train-peg-mjx": "scripts.training.train_peg",
    "train-pickplace-mjx": "scripts.training.train_pickplace",
    "resume-grasp-mjx": "scripts.training.resume_grasp",
    "resume-peg-mjx": "scripts.training.resume_peg",
}


def main() -> None:
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n{USAGE}")
        sys.exit(1)

    sys.argv = [sys.argv[0], *sys.argv[2:]]
    module = __import__(COMMANDS[command], fromlist=["main", "parse_args", "train"])

    if hasattr(module, "parse_args"):
        module.train(module.parse_args())
    else:
        module.main()


if __name__ == "__main__":
    main()
