"""Murrow CLI — `murrow <stage>`.

Stage implementations live in murrow.stages.* and are wired in as they land;
this skeleton dispatches to whichever are registered so early development can
proceed stage-by-stage without a big-bang CLI rewrite.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_discover(args: argparse.Namespace) -> int:
    from .stages import discover

    discover.run(query=args.query, event_id=args.event_id, start=args.start, end=args.end)
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    from .stages import collect

    collect.run(event_id=args.event_id)
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .stages import fetch

    fetch.run(event_id=args.event_id)
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    from .stages import baseline

    baseline.run(event_id=args.event_id)
    return 0


def _cmd_metrics(args: argparse.Namespace) -> int:
    from .stages import metrics

    metrics.run(event_id=args.event_id, model=args.model)
    return 0


def _cmd_battles(args: argparse.Namespace) -> int:
    from .stages import battles

    battles.run(event_id=args.event_id, model=args.model)
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    print("murrow build: full pipeline orchestration lands with the aggregate/publish stages.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="murrow")
    sub = parser.add_subparsers(dest="command", required=True)

    p_discover = sub.add_parser("discover", help="Snapshot GDELT coverage for one event")
    p_discover.add_argument("--event-id", required=True)
    p_discover.add_argument("--query", required=True)
    p_discover.add_argument("--start", required=True, help="YYYYMMDDHHMMSS")
    p_discover.add_argument("--end", required=True, help="YYYYMMDDHHMMSS")
    p_discover.set_defaults(func=_cmd_discover)

    p_collect = sub.add_parser("collect", help="Resolve outlet->article mapping from a GDELT snapshot")
    p_collect.add_argument("--event-id", required=True)
    p_collect.set_defaults(func=_cmd_collect)

    p_fetch = sub.add_parser("fetch", help="Transiently scrape full text for an event's articles")
    p_fetch.add_argument("--event-id", required=True)
    p_fetch.set_defaults(func=_cmd_fetch)

    p_baseline = sub.add_parser("baseline", help="Extract the neutral key-fact baseline for an event")
    p_baseline.add_argument("--event-id", required=True)
    p_baseline.set_defaults(func=_cmd_baseline)

    p_metrics = sub.add_parser("metrics", help="Extract per-article metrics for a benchmark model")
    p_metrics.add_argument("--event-id", required=True)
    p_metrics.add_argument("--model", required=True)
    p_metrics.set_defaults(func=_cmd_metrics)

    p_battles = sub.add_parser("battles", help="Run blind pairwise judging for a benchmark model")
    p_battles.add_argument("--event-id", required=True)
    p_battles.add_argument("--model", required=True)
    p_battles.set_defaults(func=_cmd_battles)

    p_build = sub.add_parser("build", help="Run the full pipeline")
    p_build.set_defaults(func=_cmd_build)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
