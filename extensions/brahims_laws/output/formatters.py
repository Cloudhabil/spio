"""
Output Formatters for Brahim's Laws Calculator.

Provides multiple output formats:
- JSON: Machine-readable structured output
- Table: ASCII table for terminal
- Rich: Beautiful colored console output

Author: Elias Oulad Brahim
"""

import json
from pathlib import Path
from typing import Optional, List, Any, Dict
from abc import ABC, abstractmethod

from ..models.curve_data import EllipticCurveData, Regime
from ..models.analysis_result import BrahimAnalysisResult, BatchAnalysisResult

# Try to import rich, fall back gracefully
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Table = None
    Panel = None


class BaseFormatter(ABC):
    """Abstract base class for formatters."""

    @abstractmethod
    def format_result(
        self,
        result: BrahimAnalysisResult,
        output_file: Optional[Path] = None
    ) -> str:
        """Format a single analysis result."""
        pass

    @abstractmethod
    def format_batch(
        self,
        results: List[BrahimAnalysisResult],
        output_file: Optional[Path] = None
    ) -> str:
        """Format batch of results."""
        pass


class JSONFormatter(BaseFormatter):
    """Format results as JSON."""

    def __init__(self, indent: int = 2, include_curve: bool = True):
        """
        Initialize JSON formatter.

        Args:
            indent: JSON indentation level
            include_curve: Whether to include full curve data
        """
        self.indent = indent
        self.include_curve = include_curve

    def format_result(
        self,
        result: BrahimAnalysisResult,
        output_file: Optional[Path] = None
    ) -> str:
        """Format single result as JSON."""
        data = result.to_dict()

        if not self.include_curve:
            # Reduce to summary
            data['curve'] = {
                'label': result.curve.label,
                'conductor': result.curve.conductor,
                'rank': result.curve.rank
            }

        json_str = json.dumps(data, indent=self.indent, default=str)

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)

        return json_str

    def format_batch(
        self,
        results: List[BrahimAnalysisResult],
        output_file: Optional[Path] = None
    ) -> str:
        """Format batch as JSON."""
        batch_result = BatchAnalysisResult(results=results)
        data = batch_result.to_dict()

        json_str = json.dumps(data, indent=self.indent, default=str)

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)

        return json_str


class TableFormatter(BaseFormatter):
    """Format results as ASCII tables."""

    def __init__(self, width: int = 80):
        """
        Initialize table formatter.

        Args:
            width: Maximum table width
        """
        self.width = width

    def format_result(
        self,
        result: BrahimAnalysisResult,
        output_file: Optional[Path] = None
    ) -> str:
        """Format single result as ASCII table."""
        lines = []

        # Header
        lines.append("=" * self.width)
        lines.append(f"Brahim's Laws Analysis: {result.curve.label}".center(self.width))
        lines.append("=" * self.width)

        # Curve info
        lines.append("")
        lines.append(f"Curve: {result.curve.weierstrass_equation}")
        lines.append(f"Conductor: {result.curve.conductor}  |  Rank: {result.curve.rank}  |  Torsion: {result.curve.torsion_order}")
        lines.append(f"Period (Omega): {result.curve.real_period:.6f}  |  Tamagawa: {result.curve.tamagawa_product}")
        lines.append(f"Sha (actual): {result.curve.sha_analytic or 1}")
        lines.append("")

        # Laws table
        lines.append("-" * self.width)
        lines.append(f"{'Law':<30} {'Value':<25} {'Status':<20}")
        lines.append("-" * self.width)

        # Law 1
        lines.append(f"{'1. Brahim Conjecture':<30} {'Sha_pred='+f'{result.sha_median_predicted:.4f}':<25} {'Error: '+f'{result.law1_error*100:.2f}%':<20}")

        # Law 2
        lines.append(f"{'2. Reynolds Number':<30} {'Rey='+f'{result.reynolds_number:.4f}':<25} {'':<20}")

        # Law 3
        regime_str = str(result.regime)
        lines.append(f"{'3. Phase Transition':<30} {regime_str:<25} {f'[{result.rey_c_lower},{result.rey_c_upper}]':<20}")

        # Law 4
        lines.append(f"{'4. Dynamic Scaling':<30} {'Sha_max='+f'{result.sha_max_predicted:.4f}':<25} {'Error: '+f'{result.law4_error*100:.2f}%':<20}")

        # Law 5
        lines.append(f"{'5. Cascade Law':<30} {'exp='+f'{result.p_scaling_exponent:.4f}':<25} {'Target: -0.25':<20}")

        # Law 6
        status = "VERIFIED" if result.is_consistent else "FAILED"
        lines.append(f"{'6. Consistency':<30} {'2/3 = 5/12 + 1/4':<25} {status:<20}")

        lines.append("-" * self.width)

        # Audit
        if result.vnand_hash:
            lines.append(f"VNAND Hash: {result.vnand_hash[:32]}...")

        lines.append("=" * self.width)

        output = "\n".join(lines)

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)

        return output

    def format_batch(
        self,
        results: List[BrahimAnalysisResult],
        output_file: Optional[Path] = None
    ) -> str:
        """Format batch summary as ASCII table."""
        lines = []

        lines.append("=" * self.width)
        lines.append("Brahim's Laws Batch Analysis".center(self.width))
        lines.append("=" * self.width)
        lines.append(f"Total curves: {len(results)}")
        lines.append("")

        # Regime distribution
        laminar = sum(1 for r in results if r.regime == Regime.LAMINAR)
        transition = sum(1 for r in results if r.regime == Regime.TRANSITION)
        turbulent = sum(1 for r in results if r.regime == Regime.TURBULENT)

        lines.append("Regime Distribution:")
        lines.append(f"  Laminar (Rey < 10):     {laminar:>6} ({laminar/len(results)*100:5.1f}%)")
        lines.append(f"  Transition (10-30):     {transition:>6} ({transition/len(results)*100:5.1f}%)")
        lines.append(f"  Turbulent (Rey > 30):   {turbulent:>6} ({turbulent/len(results)*100:5.1f}%)")
        lines.append("")

        # Statistics
        import numpy as np
        reynolds_vals = np.array([r.reynolds_number for r in results])
        finite_mask = np.isfinite(reynolds_vals)

        lines.append("Reynolds Statistics:")
        lines.append(f"  Mean:   {np.mean(reynolds_vals[finite_mask]):.2f}")
        lines.append(f"  Median: {np.median(reynolds_vals[finite_mask]):.2f}")
        lines.append("")

        # Error statistics
        law1_errors = [r.law1_error for r in results]
        lines.append("Law 1 (Brahim Conjecture) Error:")
        lines.append(f"  Mean:   {np.mean(law1_errors)*100:.2f}%")
        lines.append(f"  Median: {np.median(law1_errors)*100:.2f}%")

        lines.append("=" * self.width)

        output = "\n".join(lines)

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)

        return output


class RichFormatter(BaseFormatter):
    """Format results with Rich console output."""

    def __init__(self, console: Optional['Console'] = None):
        """
        Initialize Rich formatter.

        Args:
            console: Rich Console instance (creates new if None)
        """
        if not RICH_AVAILABLE:
            raise ImportError("Rich library not available. Install with: pip install rich")

        self.console = console or Console()

    def format_result(
        self,
        result: BrahimAnalysisResult,
        output_file: Optional[Path] = None
    ) -> str:
        """Format and display single result with Rich."""
        # Curve info panel
        curve_info = (
            f"Label: [bold]{result.curve.label}[/bold]\n"
            f"Equation: {result.curve.weierstrass_equation}\n"
            f"Conductor: {result.curve.conductor}  |  Rank: {result.curve.rank}\n"
            f"Period (Omega): {result.curve.real_period:.6f}  |  Tamagawa: {result.curve.tamagawa_product}\n"
            f"Sha (actual): {result.curve.sha_analytic or 1}"
        )

        self.console.print(Panel(curve_info, title="Elliptic Curve", border_style="blue"))

        # Laws table
        table = Table(title="Brahim's Laws Analysis")
        table.add_column("Law", style="cyan", width=28)
        table.add_column("Value", style="green", width=20)
        table.add_column("Status", style="yellow", width=18)

        # Law 1
        table.add_row(
            "1. Brahim Conjecture",
            f"Sha_pred = {result.sha_median_predicted:.4f}",
            f"Error: {result.law1_error*100:.2f}%"
        )

        # Law 2
        table.add_row(
            "2. Reynolds Number",
            f"Rey = {result.reynolds_number:.4f}",
            ""
        )

        # Law 3
        regime_color = {"LAMINAR": "green", "TRANSITION": "yellow", "TURBULENT": "red"}
        regime_str = str(result.regime)
        table.add_row(
            "3. Phase Transition",
            f"[{regime_color.get(regime_str, 'white')}]{regime_str}[/]",
            f"[{result.rey_c_lower}, {result.rey_c_upper}]"
        )

        # Law 4
        table.add_row(
            "4. Dynamic Scaling",
            f"Sha_max = {result.sha_max_predicted:.4f}",
            f"Error: {result.law4_error*100:.2f}%"
        )

        # Law 5
        table.add_row(
            "5. Cascade Law",
            f"exp = {result.p_scaling_exponent:.4f}",
            "Target: -0.25"
        )

        # Law 6
        status = "[green]VERIFIED[/green]" if result.is_consistent else "[red]FAILED[/red]"
        table.add_row(
            "6. Consistency",
            "2/3 = 5/12 + 1/4",
            status
        )

        self.console.print(table)

        # Audit hash
        if result.vnand_hash:
            self.console.print(f"\n[dim]VNAND Hash: {result.vnand_hash[:32]}...[/dim]")

        # Return summary string for file output
        return result.summary()

    def format_batch(
        self,
        results: List[BrahimAnalysisResult],
        output_file: Optional[Path] = None
    ) -> str:
        """Format and display batch summary with Rich."""
        n = len(results)

        # Header
        self.console.print(Panel(
            f"[bold]Batch Analysis Complete[/bold]\n"
            f"Total curves: {n}",
            title="Brahim's Laws Batch",
            border_style="blue"
        ))

        # Regime table
        laminar = sum(1 for r in results if r.regime == Regime.LAMINAR)
        transition = sum(1 for r in results if r.regime == Regime.TRANSITION)
        turbulent = sum(1 for r in results if r.regime == Regime.TURBULENT)

        regime_table = Table(title="Regime Distribution")
        regime_table.add_column("Regime", style="cyan")
        regime_table.add_column("Count", style="green")
        regime_table.add_column("Percentage", style="yellow")

        regime_table.add_row(
            "[green]Laminar[/green] (Rey < 10)",
            str(laminar),
            f"{laminar/n*100:.1f}%"
        )
        regime_table.add_row(
            "[yellow]Transition[/yellow] (10-30)",
            str(transition),
            f"{transition/n*100:.1f}%"
        )
        regime_table.add_row(
            "[red]Turbulent[/red] (Rey > 30)",
            str(turbulent),
            f"{turbulent/n*100:.1f}%"
        )

        self.console.print(regime_table)

        # Statistics
        import numpy as np
        reynolds_vals = np.array([r.reynolds_number for r in results])
        finite_mask = np.isfinite(reynolds_vals)
        law1_errors = [r.law1_error for r in results]

        stats_table = Table(title="Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Mean Reynolds", f"{np.mean(reynolds_vals[finite_mask]):.2f}")
        stats_table.add_row("Median Reynolds", f"{np.median(reynolds_vals[finite_mask]):.2f}")
        stats_table.add_row("Mean Law 1 Error", f"{np.mean(law1_errors)*100:.2f}%")
        stats_table.add_row("Sha > 1 Count", str(sum(1 for r in results if r.curve.has_nontrivial_sha)))

        self.console.print(stats_table)

        # Return text summary
        return TableFormatter().format_batch(results)


def get_formatter(
    format_type: str,
    console: Optional['Console'] = None
) -> BaseFormatter:
    """
    Get formatter by type name.

    Args:
        format_type: 'json', 'table', or 'rich'
        console: Rich console (for rich formatter)

    Returns:
        Formatter instance
    """
    if format_type == 'json':
        return JSONFormatter()
    elif format_type == 'table':
        return TableFormatter()
    elif format_type == 'rich':
        if RICH_AVAILABLE:
            return RichFormatter(console)
        else:
            return TableFormatter()  # Fallback
    else:
        return JSONFormatter()  # Default
