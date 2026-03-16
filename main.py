#!/usr/bin/env python3
"""
QRVS — Quantum Randomness Validation System
Main entry point.

Usage:
    python main.py                      # Run with defaults (no API)
    python main.py --anu                # Fetch real quantum data from ANU API
    python main.py --samples 10000      # Custom sample size
    python main.py --anu --samples 2000 # Both
"""

import argparse
import sys
from pathlib import Path

# Make the src package importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.analysis_pipeline import AnalysisPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantum Randomness Validation System (QRVS)"
    )
    parser.add_argument(
        "--samples", type=int, default=5000,
        help="Number of samples per synthetic source (default: 5000)"
    )
    parser.add_argument(
        "--anu", action="store_true",
        help="Fetch real quantum random numbers from the ANU QRNG API"
    )
    parser.add_argument(
        "--anu-total", type=int, default=1000,
        help="Number of ANU QRNG samples to fetch (default: 1000)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = AnalysisPipeline(
        n_samples=args.samples,
        use_anu_api=args.anu,
        anu_total=args.anu_total,
        verbose=not args.quiet,
    )

    report = pipeline.run()

    # Print a concise score table to stdout
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                    QRVS Final Scores                     │")
    print("├──────────────────┬──────────┬─────────┬──────────────────┤")
    print("│ Source           │ Score    │ Shannon │ Result           │")
    print("├──────────────────┼──────────┼─────────┼──────────────────┤")
    for name, src in report.sources.items():
        result = "✓ PASS" if src.overall_pass else "✗ FAIL"
        print(f"│ {name:<16} │ {src.score:>7.1f}  │ {src.shannon_bits:>6.3f}  │ {result:<16} │")
    print("└──────────────────┴──────────┴─────────┴──────────────────┘")
    print("\nReports saved to: results/reports/")
    print("Figures  saved to: results/figures/\n")


if __name__ == "__main__":
    main()
