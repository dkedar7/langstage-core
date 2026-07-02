"""``python -m langstage_core.host`` — print the resolved shared config.

Shows each ``LANGSTAGE_*`` value (legacy ``DEEPAGENT_*`` names still resolve),
where it resolved from (default / TOML / env / override), and the env var +
``langstage.toml`` key that set it — so you never have to remember the
variable names. Hosts can ship their own subclass printer for host-specific
keys; this covers the shared core.
"""
import argparse

from .config import HostConfig


def main(argv: list[str] | None = None) -> int:
    # A tiny parser so `-h/--help` works and unknown flags error, instead of
    # every arg (including --help) being a silent no-op (gh #-dogfood).
    parser = argparse.ArgumentParser(
        prog="python -m langstage_core.host",
        description="Print the resolved shared host config (value, source, env/TOML key) and exit.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved config (the default action - accepted for symmetry with langstage-agui).",
    )
    parser.parse_args(argv)
    print(HostConfig.resolve().describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
