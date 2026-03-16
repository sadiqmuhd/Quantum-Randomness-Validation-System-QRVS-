"""
Analysis Pipeline
Orchestrates the full QRVS analysis: data acquisition → preprocessing →
statistical tests → entropy estimation → visualization → report.
"""

import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

# ── Internal imports ─────────────────────────────────────────────────────────
from ..generators.pseudo_rng import generate_uniform, generate_biased, generate_lcg
from ..generators.simulated_quantum import generate_vacuum_fluctuation, generate_gaussian_noise
from ..preprocessing.data_cleaner import preprocess
from ..tests.chi_square import chi_square_test
from ..tests.ks_test import ks_test_uniform
from ..tests.autocorrelation import autocorrelation_test, runs_test
from ..entropy.shannon_entropy import shannon_entropy, block_entropy
from ..entropy.min_entropy import min_entropy, nist_iid_min_entropy
from ..visualization.distribution_plots import plot_distribution_comparison, plot_overlay_kde
from ..visualization.entropy_plots import (plot_entropy_comparison,
                                            plot_block_entropy,
                                            plot_autocorrelation)


REPORTS_DIR = Path.cwd() / "results" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SourceReport:
    name: str
    n_samples: int
    # Statistical tests
    chi_square_stat:   float = 0.0
    chi_square_p:      float = 0.0
    chi_square_pass:   bool  = False
    ks_stat:           float = 0.0
    ks_p:              float = 0.0
    ks_pass:           bool  = False
    max_acf:           float = 0.0
    n_significant_lags:int   = 0
    acf_pass:          bool  = False
    runs_p:            float = 0.0
    runs_pass:         bool  = False
    # Entropy
    shannon_bits:      float = 0.0
    shannon_efficiency:float = 0.0
    min_entropy_bits:  float = 0.0
    min_entropy_eff:   float = 0.0
    nist_iid_bits:     float = 0.0
    # Overall
    overall_pass:      bool  = False
    score:             float = 0.0   # 0–100


@dataclass
class PipelineReport:
    timestamp: str
    sources: Dict[str, SourceReport] = field(default_factory=dict)

    def to_dict(self) -> dict:
        import numpy as np

        def _convert(obj):
            if isinstance(obj, (bool,)):
                return bool(obj)
            if hasattr(obj, 'item'):   # numpy scalar
                return obj.item()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            return obj

        return _convert({
            "timestamp": self.timestamp,
            "sources": {k: asdict(v) for k, v in self.sources.items()},
        })


class AnalysisPipeline:
    """
    End-to-end QRVS analysis pipeline.

    Usage:
        pipeline = AnalysisPipeline(n_samples=5000)
        pipeline.run()
    """

    def __init__(self,
                 n_samples: int = 5000,
                 use_anu_api: bool = False,
                 anu_total: int = 1000,
                 verbose: bool = True):
        """
        Args:
            n_samples:    Number of samples to generate per synthetic source.
            use_anu_api:  Attempt to fetch real QRNG data from ANU (requires internet).
            anu_total:    Number of ANU samples to fetch if use_anu_api is True.
            verbose:      Print progress messages.
        """
        self.n_samples   = n_samples
        self.use_anu_api = use_anu_api
        self.anu_total   = anu_total
        self.verbose     = verbose

        self.raw_sources:    Dict[str, np.ndarray] = {}
        self.clean_sources:  Dict[str, np.ndarray] = {}
        self.report = PipelineReport(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))

    # ── Step 1: Data Acquisition ─────────────────────────────────────────────

    def acquire_data(self) -> None:
        self._log("=== Step 1: Data Acquisition ===")
        rng_seed = 42

        self.raw_sources["Pseudo RNG"]     = generate_uniform(self.n_samples, seed=rng_seed)
        self.raw_sources["Simulated QRNG"] = generate_vacuum_fluctuation(self.n_samples, seed=rng_seed)
        self.raw_sources["Biased RNG"]     = generate_biased(self.n_samples, bias_value=42,
                                                               bias_weight=0.25, seed=rng_seed)
        self.raw_sources["LCG"]            = generate_lcg(self.n_samples, seed=rng_seed)

        if self.use_anu_api:
            try:
                from ..api.anu_qrng import get_or_fetch_qrng
                self.raw_sources["ANU QRNG"] = get_or_fetch_qrng(total=self.anu_total)
                self._log(f"  ANU QRNG: {len(self.raw_sources['ANU QRNG'])} values loaded.")
            except Exception as e:
                self._log(f"  [WARNING] ANU QRNG fetch failed: {e}. Skipping.")

        self._log(f"  Sources acquired: {list(self.raw_sources.keys())}")

    # ── Step 2: Preprocessing ────────────────────────────────────────────────

    def preprocess_data(self) -> None:
        self._log("=== Step 2: Preprocessing ===")
        for name, data in self.raw_sources.items():
            whiten = name == "Simulated QRNG"
            cleaned = preprocess(data, remove_outliers=False, normalize=False, whiten=whiten)
            self.clean_sources[name] = cleaned
            self._log(f"  {name}: {len(cleaned)} samples after cleaning.")

    # ── Step 3: Statistical Tests ────────────────────────────────────────────

    def run_statistical_tests(self) -> None:
        self._log("=== Step 3: Statistical Tests ===")
        for name, data in self.clean_sources.items():
            self._log(f"\n  [{name}]")
            src_report = SourceReport(name=name, n_samples=len(data))

            # Chi-square
            chi = chi_square_test(data)
            src_report.chi_square_stat = chi.statistic
            src_report.chi_square_p    = chi.p_value
            src_report.chi_square_pass = chi.is_random
            self._log(f"    Chi²:  {chi.verdict}")

            # KS test
            ks = ks_test_uniform(data)
            src_report.ks_stat = ks.statistic
            src_report.ks_p    = ks.p_value
            src_report.ks_pass = ks.is_random
            self._log(f"    KS:    {ks.verdict}")

            # Autocorrelation
            acf_result = autocorrelation_test(data.astype(np.float64), max_lag=50)
            src_report.max_acf            = acf_result.max_acf
            src_report.n_significant_lags = len(acf_result.significant_lags)
            src_report.acf_pass           = acf_result.is_random
            self._log(f"    ACF:   {acf_result.verdict}")

            # Runs test
            runs = runs_test(data.astype(np.float64))
            src_report.runs_p    = runs["p_value"]
            src_report.runs_pass = runs["is_random"]
            self._log(f"    Runs:  {runs['verdict']}")

            self.report.sources[name] = src_report

    # ── Step 4: Entropy Estimation ───────────────────────────────────────────

    def estimate_entropy(self) -> None:
        self._log("=== Step 4: Entropy Estimation ===")
        for name, data in self.clean_sources.items():
            src = self.report.sources[name]

            sh = shannon_entropy(data)
            src.shannon_bits      = sh.entropy_bits
            src.shannon_efficiency = sh.efficiency
            self._log(f"  [{name}] Shannon: {sh.entropy_bits:.4f} bits ({sh.efficiency*100:.1f}%)")

            me = min_entropy(data)
            src.min_entropy_bits = me.min_entropy_bits
            src.min_entropy_eff  = me.efficiency
            self._log(f"  [{name}] Min-H:   {me.min_entropy_bits:.4f} bits ({me.efficiency*100:.1f}%)")

            src.nist_iid_bits = nist_iid_min_entropy(data)
            self._log(f"  [{name}] NIST IID: {src.nist_iid_bits:.4f} bits")

    # ── Step 5: Scoring ──────────────────────────────────────────────────────

    def compute_scores(self) -> None:
        self._log("=== Step 5: Scoring ===")
        for name, src in self.report.sources.items():
            tests = [src.chi_square_pass, src.ks_pass, src.acf_pass, src.runs_pass]
            n_pass = sum(tests)
            # Weighted score: 40% entropy efficiency + 60% test pass rate
            entropy_score  = src.shannon_efficiency * 100 * 0.40
            test_score     = (n_pass / len(tests)) * 100 * 0.60
            src.score      = round(entropy_score + test_score, 2)
            src.overall_pass = src.score >= 70.0
            self._log(f"  [{name}] Score: {src.score:.1f}/100  ({'PASS' if src.overall_pass else 'FAIL'})")

    # ── Step 6: Visualization ────────────────────────────────────────────────

    def generate_plots(self) -> None:
        self._log("=== Step 6: Generating Plots ===")
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for scripted runs

        # Distribution comparison
        plot_distribution_comparison(self.clean_sources)
        plot_overlay_kde(self.clean_sources)

        # Entropy comparison
        h_vals   = {n: s.shannon_bits     for n, s in self.report.sources.items()}
        hmin_vals = {n: s.min_entropy_bits for n, s in self.report.sources.items()}
        plot_entropy_comparison(h_vals, hmin_vals)

        # Block entropy
        from ..entropy.shannon_entropy import block_entropy as _block_entropy
        be_data = {n: _block_entropy(d, block_size=256)
                   for n, d in self.clean_sources.items()}
        plot_block_entropy(be_data, block_size=256)

        # Autocorrelation
        from ..tests.autocorrelation import compute_acf
        acf_data  = {n: compute_acf(d.astype(np.float64), max_lag=50)
                     for n, d in self.clean_sources.items()}
        n_samples = {n: len(d) for n, d in self.clean_sources.items()}
        plot_autocorrelation(acf_data, n_samples)

        self._log("  All plots saved to results/figures/.")

    # ── Step 7: Save Report ──────────────────────────────────────────────────

    def save_report(self) -> Path:
        self._log("=== Step 7: Saving Report ===")
        report_dict = self.report.to_dict()
        path = REPORTS_DIR / f"qrvs_report_{self.report.timestamp.replace(':', '-')}.json"
        with open(path, "w") as f:
            json.dump(report_dict, f, indent=2)
        self._log(f"  Report saved: {path}")

        # Also write a human-readable text summary
        txt_path = path.with_suffix(".txt")
        with open(txt_path, "w") as f:
            f.write(self._text_summary())
        self._log(f"  Text summary: {txt_path}")

        return path

    # ── Full run ─────────────────────────────────────────────────────────────

    def run(self) -> PipelineReport:
        """Execute all pipeline steps in order."""
        self._log("\n" + "=" * 60)
        self._log("  Quantum Randomness Validation System (QRVS)")
        self._log("=" * 60)
        self.acquire_data()
        self.preprocess_data()
        self.run_statistical_tests()
        self.estimate_entropy()
        self.compute_scores()
        self.generate_plots()
        self.save_report()
        self._log("\n=== Pipeline complete ===\n")
        return self.report

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _text_summary(self) -> str:
        lines = [
            "QRVS Statistical Summary Report",
            f"Generated: {self.report.timestamp}",
            "=" * 60,
        ]
        for name, src in self.report.sources.items():
            lines += [
                f"\nSource: {name}",
                f"  Samples            : {src.n_samples}",
                f"  Chi² p-value       : {src.chi_square_p:.4f}  ({'PASS' if src.chi_square_pass else 'FAIL'})",
                f"  KS   p-value       : {src.ks_p:.4f}  ({'PASS' if src.ks_pass else 'FAIL'})",
                f"  Max |ACF|          : {src.max_acf:.6f}  ({'PASS' if src.acf_pass else 'FAIL'})",
                f"  Runs p-value       : {src.runs_p:.4f}  ({'PASS' if src.runs_pass else 'FAIL'})",
                f"  Shannon entropy    : {src.shannon_bits:.4f} bits  ({src.shannon_efficiency*100:.1f}%)",
                f"  Min-entropy        : {src.min_entropy_bits:.4f} bits  ({src.min_entropy_eff*100:.1f}%)",
                f"  NIST IID estimate  : {src.nist_iid_bits:.4f} bits",
                f"  Overall Score      : {src.score:.1f}/100  ({'PASS' if src.overall_pass else 'FAIL'})",
            ]
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
