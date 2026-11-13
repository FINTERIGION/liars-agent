"""Generate the independent, legally replayable endgame diagnostic set."""
import argparse
import json

from liars_agent.evaluation.endgame_coverage import build_endgame_coverage, validate_endgame_coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026111301)
    parser.add_argument("--samples-per-cell", type=int, default=10)
    parser.add_argument("--max-games", type=int, default=500)
    parser.add_argument("--no-claim-per-alive", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        report = validate_endgame_coverage(args.output_dir, args.source_run)
    else:
        report = build_endgame_coverage(args.source_run, args.output_dir, seed=args.seed,
            samples_per_cell=args.samples_per_cell, max_games=args.max_games,
            no_claim_per_alive=args.no_claim_per_alive)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
