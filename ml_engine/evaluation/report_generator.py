"""Statistical report generator: produces Markdown reports per analysis type."""
from pathlib import Path
from datetime import datetime
import json

from loguru import logger


class ReportGenerator:
    """Generates individual Markdown report files from analysis results."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, filename: str, content: str):
        path = self.output_dir / filename
        path.write_text(content, encoding="utf-8")
        logger.info(f"Report written: {path}")
        return str(path)

    def descriptive_stats_report(self, stats: dict) -> str:
        lines = [f"# Descriptive Statistics Report\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n"]
        for col, s in stats.items():
            lines.append(f"## {col}")
            lines.append("| Metric | Value |\n|--------|-------|")
            for k, v in s.items():
                if k != "column":
                    lines.append(f"| {k} | {v} |")
            lines.append("")
        return self._write("descriptive_statistics.md", "\n".join(lines))

    def stationarity_report(self, results: list[dict]) -> str:
        lines = [f"# Stationarity Analysis Report\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n"]
        for r in results:
            series_name = r.get("series", "Unknown")
            lines.append(f"## Series: {series_name}")
            lines.append(f"**Verdict:** {r.get('verdict', 'N/A')}\n")
            adf = r.get("adf", {})
            if adf:
                lines.append(f"**ADF Test:** statistic={adf.get('statistic')}, p={adf.get('p_value')} — {adf.get('interpretation')}")
            kpss = r.get("kpss", {})
            if kpss and "error" not in kpss:
                lines.append(f"**KPSS Test:** statistic={kpss.get('statistic')}, p={kpss.get('p_value')} — {kpss.get('interpretation')}")
            lines.append("")
        return self._write("stationarity.md", "\n".join(lines))

    def residual_report(self, diagnostics: dict) -> str:
        lines = [f"# Residual Diagnostics Report\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n"]
        lb = diagnostics.get("ljung_box", {})
        dw = diagnostics.get("durbin_watson", {})
        sw = diagnostics.get("shapiro_wilk", {})
        lines += [
            f"- **N residuals:** {diagnostics.get('n')}",
            f"- **Mean (bias):** {diagnostics.get('mean')} — {diagnostics.get('bias_interpretation')}",
            f"- **Std Dev:** {diagnostics.get('std')}",
            f"- **Skewness:** {diagnostics.get('skewness')}",
            f"- **Kurtosis:** {diagnostics.get('kurtosis')}",
            "",
            f"**Ljung-Box (lag 10):** p={lb.get('lag_10', {}).get('p_value')} — {lb.get('interpretation', 'N/A')}",
            f"**Durbin-Watson:** {dw.get('statistic')} — {dw.get('interpretation')}",
            f"**Shapiro-Wilk normality:** p={sw.get('p_value')} — {'Normal' if sw.get('is_normal') else 'Non-normal'}",
        ]
        return self._write("residual_analysis.md", "\n".join(lines))

    def metrics_report(self, metrics: dict, model_name: str = "Ensemble") -> str:
        lines = [f"# Forecast Metrics Report — {model_name}\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n"]
        lines.append("| Metric | Value |\n|--------|-------|")
        for k, v in metrics.items():
            lines.append(f"| {k} | {v} |")
        return self._write("forecast_metrics.md", "\n".join(lines))

    def significance_report(self, sig_results: list[dict]) -> str:
        lines = [f"# Statistical Significance Report\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n"]
        for r in sig_results:
            lines.append(f"## {r.get('comparison', r.get('test', 'Test'))}")
            for k, v in r.items():
                lines.append(f"- **{k}:** {v}")
            lines.append("")
        return self._write("significance_tests.md", "\n".join(lines))

    def model_comparison_report(self, comparison: dict) -> str:
        lines = [f"# Model Comparison Report\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n"]
        lines.append(f"**Best Model (lowest WMAPE):** {comparison.get('best_model')}\n")
        lines.append("## Ranking\n")
        lines.append("| Model | MAE | RMSE | WMAPE | PICP | Train Time |")
        lines.append("|-------|-----|------|-------|------|------------|")
        for r in comparison.get("ranking", []):
            lines.append(
                f"| {r.get('model')} | {r.get('mae')} | {r.get('rmse')} | "
                f"{r.get('wmape')}% | {r.get('picp', 'N/A')} | {r.get('train_time_s')}s |"
            )
        lines.append("\n## Significance Tests\n")
        for t in comparison.get("significance_tests", []):
            lines.append(f"- **{t.get('comparison')}**: {t.get('interpretation')} | Effect: {t.get('magnitude')}")
        return self._write("model_comparison.md", "\n".join(lines))

