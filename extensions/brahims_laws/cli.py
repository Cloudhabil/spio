"""
Brahim's Laws Calculator - Command Line Interface.

Typer-based CLI for analyzing elliptic curves against Brahim's Laws.

Commands:
    analyze         Analyze a single curve by Cremona label
    batch           Batch process curves from file
    lmfdb-query     Query LMFDB API with filters
    coefficients    Analyze from Weierstrass coefficients
    verify-consistency  Verify the 2/3 = 5/12 + 1/4 relation
    regime-stats    Compute Reynolds regime statistics

Usage:
    brahims-laws analyze 11a1
    brahims-laws batch curves.json --gpu
    brahims-laws verify-consistency

Author: Elias Oulad Brahim
"""

import typer
from pathlib import Path
from typing import Optional
import json
import sys

# Create Typer app
app = typer.Typer(
    name="brahims-laws",
    help="Brahim's Laws Calculator for Elliptic Curves",
    add_completion=False,
    no_args_is_help=True
)

# Try to import rich for pretty output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_output(message: str, style: str = ""):
    """Print message with optional rich styling."""
    if RICH_AVAILABLE and console:
        console.print(message, style=style)
    else:
        print(message)


def print_error(message: str):
    """Print error message."""
    if RICH_AVAILABLE and console:
        console.print(f"[red]Error:[/red] {message}")
    else:
        print(f"Error: {message}", file=sys.stderr)


# ==========================================================================
# ANALYZE COMMAND
# ==========================================================================

@app.command("analyze")
def analyze_curve(
    label: str = typer.Argument(..., help="Cremona/LMFDB label (e.g., '11a1', '37.a1')"),
    source: str = typer.Option(
        "auto", "--source", "-s",
        help="Data source: cremona, lmfdb, auto"
    ),
    output_format: str = typer.Option(
        "rich", "--format", "-f",
        help="Output format: json, table, rich"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output file path"
    ),
):
    """
    Analyze a single elliptic curve against all 6 Brahim's Laws.

    Examples:
        brahims-laws analyze 11a1
        brahims-laws analyze 37.a1 --source lmfdb
        brahims-laws analyze 389a1 --format json --output result.json
    """
    from .data.cremona_loader import CremonaLoader
    from .data.lmfdb_client import LMFDBClient
    from .core.brahim_laws import BrahimLawsEngine
    from .audit.vnand_hasher import VNANDHasher
    from .output.formatters import get_formatter

    print_output(f"[bold blue]Analyzing curve: {label}[/bold blue]")

    # Load curve data
    curve = None

    if source in ["auto", "cremona"]:
        loader = CremonaLoader()
        curve = loader.load_by_label(label)
        if curve:
            print_output("[dim]Loaded from Cremona database[/dim]")

    if curve is None and source in ["auto", "lmfdb"]:
        print_output("[dim]Querying LMFDB...[/dim]")
        client = LMFDBClient()
        curve = client.fetch_by_label(label)
        if curve:
            print_output("[dim]Loaded from LMFDB[/dim]")

    if curve is None:
        print_error(f"Could not find curve {label}")
        raise typer.Exit(1)

    # Run analysis
    engine = BrahimLawsEngine()
    result = engine.analyze(curve)

    # Generate VNAND hash
    hasher = VNANDHasher()
    hasher.hash_and_attach(result)

    # Format and output
    formatter = get_formatter(output_format, console if RICH_AVAILABLE else None)
    output = formatter.format_result(result, output_file)

    if output_format == "json" and not output_file:
        print(output)

    if output_file:
        print_output(f"\n[green]Results saved to {output_file}[/green]")


# ==========================================================================
# BATCH COMMAND
# ==========================================================================

@app.command("batch")
def batch_analyze(
    source_file: Path = typer.Argument(..., help="Input JSON/JSONL file with curves"),
    output_dir: Path = typer.Option(
        "outputs/brahims_laws", "--output-dir", "-o",
        help="Output directory"
    ),
    batch_size: int = typer.Option(
        2000, "--batch-size", "-b",
        help="GPU batch size"
    ),
    use_gpu: bool = typer.Option(
        True, "--gpu/--no-gpu",
        help="Use CUDA acceleration"
    ),
    output_format: str = typer.Option(
        "all", "--format", "-f",
        help="Output format: json, table, all"
    ),
):
    """
    Batch analyze curves from file using GPU acceleration.

    Examples:
        brahims-laws batch curves.json
        brahims-laws batch data.jsonl --batch-size 5000
        brahims-laws batch curves.json --no-gpu --format json
    """
    from .data.cremona_loader import CremonaLoader
    from .gpu.batch_processor import CUDABatchProcessor
    from .audit.vnand_hasher import VNANDHasher
    from .output.formatters import JSONFormatter, TableFormatter

    if not source_file.exists():
        print_error(f"File not found: {source_file}")
        raise typer.Exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load curves
    print_output(f"[bold]Loading curves from {source_file}...[/bold]")

    loader = CremonaLoader()
    if source_file.suffix == ".jsonl":
        curves = list(loader.load_from_jsonl(source_file))
    else:
        curves = loader.load_from_json(source_file)

    if not curves:
        print_error("No curves loaded from file")
        raise typer.Exit(1)

    print_output(f"Loaded [green]{len(curves)}[/green] curves")

    # Process with GPU/CPU
    device = "cuda" if use_gpu else "cpu"
    processor = CUDABatchProcessor(device=device)

    print_output(f"\n[bold]Processing on {device}...[/bold]")

    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing curves", total=len(curves))
            results = processor.process_batch(curves, batch_size=batch_size)
            progress.update(task, completed=len(curves))
    else:
        results = processor.process_batch(curves, batch_size=batch_size)

    # Generate VNAND audit trail
    hasher = VNANDHasher()
    batch_audit = hasher.hash_batch(results, attach=True)
    hasher.save_audit_trail(output_dir)

    # Compute and display statistics
    stats = processor.compute_statistics(results)

    if RICH_AVAILABLE and console:
        from .output.formatters import RichFormatter
        formatter = RichFormatter(console)
        formatter.format_batch(results)
    else:
        print(TableFormatter().format_batch(results))

    # Save results
    if output_format in ["json", "all"]:
        json_file = output_dir / "results.json"
        JSONFormatter().format_batch(results, json_file)
        print_output(f"[green]JSON saved to {json_file}[/green]")

    if output_format in ["table", "all"]:
        table_file = output_dir / "summary.txt"
        TableFormatter().format_batch(results, table_file)
        print_output(f"[green]Summary saved to {table_file}[/green]")

    print_output("\n[bold green]Analysis complete![/bold green]")
    print_output(f"Results: {output_dir}")
    print_output(f"Batch hash: {batch_audit['master_hash'][:32]}...")


# ==========================================================================
# LMFDB QUERY COMMAND
# ==========================================================================

@app.command("lmfdb-query")
def lmfdb_query(
    rank: Optional[int] = typer.Option(
        None, "--rank", "-r",
        help="Filter by rank"
    ),
    conductor_min: Optional[int] = typer.Option(
        None, "--cond-min",
        help="Minimum conductor"
    ),
    conductor_max: Optional[int] = typer.Option(
        None, "--cond-max",
        help="Maximum conductor"
    ),
    limit: int = typer.Option(
        100, "--limit", "-l",
        help="Maximum curves to fetch"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output file"
    ),
    analyze: bool = typer.Option(
        True, "--analyze/--no-analyze",
        help="Run Brahim analysis on results"
    ),
):
    """
    Query LMFDB API and optionally analyze results.

    Examples:
        brahims-laws lmfdb-query --rank 0 --cond-max 1000
        brahims-laws lmfdb-query --rank 0 --limit 500 --output curves.json
    """
    from .data.lmfdb_client import LMFDBClient
    from .gpu.batch_processor import CUDABatchProcessor
    from .output.formatters import JSONFormatter, get_formatter

    print_output("[bold]Querying LMFDB...[/bold]")

    client = LMFDBClient()

    # Test connection
    if not client.test_connection():
        print_error("Cannot connect to LMFDB API")
        raise typer.Exit(1)

    curves = client.fetch_batch(
        rank=rank,
        conductor_min=conductor_min,
        conductor_max=conductor_max,
        limit=limit
    )

    print_output(f"Fetched [green]{len(curves)}[/green] curves from LMFDB")

    if not curves:
        print_output("[yellow]No curves found matching criteria[/yellow]")
        raise typer.Exit(0)

    if analyze:
        processor = CUDABatchProcessor()
        results = processor.process_batch(curves)

        if RICH_AVAILABLE and console:
            from .output.formatters import RichFormatter
            RichFormatter(console).format_batch(results)
        else:
            from .output.formatters import TableFormatter
            print(TableFormatter().format_batch(results))

        if output_file:
            JSONFormatter().format_batch(results, output_file)
            print_output(f"[green]Results saved to {output_file}[/green]")
    else:
        # Just save raw curves
        if output_file:
            data = {"curves": [c.to_dict() for c in curves]}
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print_output(f"[green]Curves saved to {output_file}[/green]")


# ==========================================================================
# COEFFICIENTS COMMAND
# ==========================================================================

@app.command("coefficients")
def analyze_from_coefficients(
    a1: int = typer.Argument(..., help="a1 coefficient"),
    a2: int = typer.Argument(..., help="a2 coefficient"),
    a3: int = typer.Argument(..., help="a3 coefficient"),
    a4: int = typer.Argument(..., help="a4 coefficient"),
    a6: int = typer.Argument(..., help="a6 coefficient"),
    output_format: str = typer.Option("rich", "--format", "-f"),
):
    """
    Analyze curve from Weierstrass coefficients [a1, a2, a3, a4, a6].

    The curve is y^2 + a1*xy + a3*y = x^3 + a2*x^2 + a4*x + a6

    Note: Full BSD data requires SageMath or LMFDB lookup.
    This provides partial analysis with user-provided coefficients.

    Examples:
        brahims-laws coefficients 0 -1 1 -10 -20  # Curve 11a1
        brahims-laws coefficients 0 0 1 -1 0      # Curve 37a1
    """
    from .models.curve_data import EllipticCurveData
    from .core.brahim_laws import BrahimLawsEngine
    from .output.formatters import get_formatter

    # Create curve from coefficients
    curve = EllipticCurveData.from_coefficients(a1, a2, a3, a4, a6)

    print_output(f"[bold]Curve: {curve.weierstrass_equation}[/bold]")
    print_output("")
    print_output("[yellow]Note: Full analysis requires BSD invariants from SageMath or LMFDB.[/yellow]")
    print_output("[yellow]Using placeholder values for demonstration.[/yellow]")
    print_output("")

    # Set some placeholder values for demonstration
    # In production, these would come from SageMath computation
    curve.conductor = 11  # Placeholder
    curve.real_period = 1.0
    curve.tamagawa_product = 1
    curve.im_tau = 1.0
    curve.sha_analytic = 1

    # Run analysis
    engine = BrahimLawsEngine()
    result = engine.analyze(curve)

    # Format output
    formatter = get_formatter(output_format, console if RICH_AVAILABLE else None)
    formatter.format_result(result)


# ==========================================================================
# VERIFY CONSISTENCY COMMAND
# ==========================================================================

@app.command("verify-consistency")
def verify_consistency():
    """
    Verify the mathematical consistency relation: 2/3 = 5/12 + 1/4

    This fundamental relation connects Laws 1, 4, and 5:
    - 2/3:  Geometric exponent (Brahim Conjecture)
    - 5/12: Dynamic exponent (Reynolds scaling)
    - 1/4:  Cascade exponent (prime variance decay)

    The exact equality suggests these laws are manifestations
    of a single underlying structure.
    """
    from .core.constants import CONSTANTS
    from fractions import Fraction

    if RICH_AVAILABLE and console:
        table = Table(title="Law 6: Consistency Relation Verification")
        table.add_column("Component", style="cyan")
        table.add_column("Symbol", style="green")
        table.add_column("Value", style="yellow")
        table.add_column("Fraction", style="magenta")

        table.add_row(
            "Geometric (Law 1)",
            "alpha",
            f"{CONSTANTS.ALPHA_IMTAU:.10f}",
            str(Fraction(2, 3))
        )
        table.add_row(
            "Dynamic (Law 4)",
            "gamma",
            f"{CONSTANTS.GAMMA_REY:.10f}",
            str(Fraction(5, 12))
        )
        table.add_row(
            "Cascade (Law 5)",
            "|delta|",
            f"{abs(CONSTANTS.DELTA_CASCADE):.10f}",
            str(Fraction(1, 4))
        )

        console.print(table)
        console.print()

        # Verification
        lhs = Fraction(2, 3)
        rhs = Fraction(5, 12) + Fraction(1, 4)

        console.print(Panel(
            f"[bold]LHS:[/bold] alpha = 2/3 = {float(lhs):.10f}\n"
            f"[bold]RHS:[/bold] gamma + |delta| = 5/12 + 1/4 = {float(rhs):.10f}\n\n"
            f"[bold]Difference:[/bold] {abs(float(lhs) - float(rhs)):.2e}\n\n"
            f"{'[green]VERIFIED: 2/3 = 5/12 + 1/4 exactly![/green]' if lhs == rhs else '[red]X Relation failed[/red]'}",
            title="Consistency Check",
            border_style="green" if lhs == rhs else "red"
        ))
    else:
        print("Law 6: Consistency Relation Verification")
        print("=" * 50)
        print(f"Alpha (2/3):     {CONSTANTS.ALPHA_IMTAU:.10f}")
        print(f"Gamma (5/12):    {CONSTANTS.GAMMA_REY:.10f}")
        print(f"|Delta| (1/4):   {abs(CONSTANTS.DELTA_CASCADE):.10f}")
        print()
        print(f"LHS: 2/3 = {2/3:.10f}")
        print(f"RHS: 5/12 + 1/4 = {5/12 + 1/4:.10f}")
        print()
        is_consistent = abs(2/3 - (5/12 + 1/4)) < 1e-10
        print(f"Verified: {'YES' if is_consistent else 'NO'}")


# ==========================================================================
# REGIME STATS COMMAND
# ==========================================================================

@app.command("regime-stats")
def regime_statistics(
    source_file: Path = typer.Argument(..., help="Input data file"),
):
    """
    Compute Reynolds regime distribution statistics.

    Analyzes the distribution of curves across:
    - Laminar (Rey < 10): Sha = 1 expected
    - Transition (10-30): Mixed behavior
    - Turbulent (Rey > 30): Sha > 1 possible

    Examples:
        brahims-laws regime-stats curves.json
    """
    from .data.cremona_loader import CremonaLoader
    from .core.reynolds import ReynoldsAnalyzer

    if not source_file.exists():
        print_error(f"File not found: {source_file}")
        raise typer.Exit(1)

    loader = CremonaLoader()
    curves = loader.load_from_json(source_file)

    if not curves:
        print_error("No curves loaded")
        raise typer.Exit(1)

    analyzer = ReynoldsAnalyzer()
    stats = analyzer.statistics(curves)

    if RICH_AVAILABLE and console:
        table = Table(title=f"Reynolds Regime Distribution (n={stats.total_curves})")
        table.add_column("Regime", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")
        table.add_column("P(Sha > 1)", style="magenta")

        table.add_row(
            "[green]Laminar[/green] (Rey < 10)",
            str(stats.laminar_count),
            f"{stats.laminar_percent:.1f}%",
            f"{stats.sha_nontrivial_by_regime.get('laminar', 0)*100:.1f}%"
        )
        table.add_row(
            "[yellow]Transition[/yellow] (10-30)",
            str(stats.transition_count),
            f"{stats.transition_percent:.1f}%",
            f"{stats.sha_nontrivial_by_regime.get('transition', 0)*100:.1f}%"
        )
        table.add_row(
            "[red]Turbulent[/red] (Rey > 30)",
            str(stats.turbulent_count),
            f"{stats.turbulent_percent:.1f}%",
            f"{stats.sha_nontrivial_by_regime.get('turbulent', 0)*100:.1f}%"
        )

        console.print(table)

        console.print(Panel(
            f"Mean Reynolds: {stats.mean_reynolds:.2f}\n"
            f"Median Reynolds: {stats.median_reynolds:.2f}\n"
            f"Std Dev: {stats.std_reynolds:.2f}",
            title="Reynolds Statistics"
        ))
    else:
        print(f"Reynolds Regime Distribution (n={stats.total_curves})")
        print("=" * 50)
        print(f"Laminar (Rey < 10):   {stats.laminar_count} ({stats.laminar_percent:.1f}%)")
        print(f"Transition (10-30):   {stats.transition_count} ({stats.transition_percent:.1f}%)")
        print(f"Turbulent (Rey > 30): {stats.turbulent_count} ({stats.turbulent_percent:.1f}%)")
        print()
        print(f"Mean Reynolds: {stats.mean_reynolds:.2f}")
        print(f"Median Reynolds: {stats.median_reynolds:.2f}")


# ==========================================================================
# ML TRAINING COMMAND
# ==========================================================================

@app.command("ml-train")
def ml_train(
    data_source: str = typer.Option(
        "lmfdb",
        "--source", "-s",
        help="Data source: lmfdb or file path"
    ),
    n_curves: int = typer.Option(
        5000,
        "--n-curves", "-n",
        help="Number of curves to fetch from LMFDB"
    ),
    model_type: str = typer.Option(
        "all",
        "--model", "-m",
        help="Model type: gbm, rf, neural, ensemble, or all"
    ),
    output_dir: Path = typer.Option(
        Path("models/sha_predictor"),
        "--output", "-o",
        help="Output directory for trained model"
    ),
):
    """
    Train ML models to predict Sha from curve invariants.

    Uses Brahim's Laws scaling relationships as feature engineering basis.
    Trains on curves with known Sha values from LMFDB.

    Examples:
        brahims-laws ml-train --n-curves 10000
        brahims-laws ml-train --model gbm --output models/gbm
        brahims-laws ml-train --source curves.json
    """
    try:
        from .ml.trainer import ShaModelTrainer
    except ImportError as e:
        print_error(f"ML dependencies not installed: {e}")
        print_output("[yellow]Install with: pip install scikit-learn torch[/yellow]")
        raise typer.Exit(1)

    trainer = ShaModelTrainer(output_dir=output_dir)

    # Load data
    if data_source == "lmfdb":
        print_output(f"[bold]Fetching {n_curves} curves from LMFDB...[/bold]")
        n_loaded = trainer.collect_data(n_curves=n_curves, verbose=True)
        if n_loaded == 0:
            print_error("Could not fetch data from LMFDB")
            raise typer.Exit(1)
    else:
        print_output(f"[bold]Loading data from {data_source}...[/bold]")
        n_loaded = trainer.load_data(Path(data_source))

    print_output(f"[green]Loaded {n_loaded} curves with known Sha[/green]\n")

    # Train
    if model_type == "all":
        print_output("[bold]Training all model types...[/bold]\n")
        results = trainer.train_all_models(verbose=True)
    else:
        results = {model_type: trainer.train_model(model_type, verbose=True)}

    # Save best model
    save_path = trainer.save_best_model()
    print_output(f"\n[green]Best model saved to {save_path}[/green]")

    # Generate report
    report = trainer.generate_report()
    report_path = output_dir / "training_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print_output(f"[green]Report saved to {report_path}[/green]")


@app.command("ml-predict")
def ml_predict(
    label: str = typer.Argument(..., help="Curve label to predict Sha for"),
    model_path: Path = typer.Option(
        Path("models/sha_predictor/gbm"),
        "--model", "-m",
        help="Path to trained model"
    ),
):
    """
    Predict Sha for a curve using trained ML model.

    Examples:
        brahims-laws ml-predict 389a1
        brahims-laws ml-predict 5077a1 --model models/sha_predictor/ensemble
    """
    try:
        from .ml.sha_predictor import ShaPredictor
        from .data.lmfdb_client import LMFDBClient
    except ImportError as e:
        print_error(f"ML dependencies not installed: {e}")
        raise typer.Exit(1)

    # Load model
    if not model_path.exists():
        print_error(f"Model not found at {model_path}")
        print_output("[yellow]Train a model first with: brahims-laws ml-train[/yellow]")
        raise typer.Exit(1)

    predictor = ShaPredictor()
    predictor.load(model_path)

    # Fetch curve
    print_output(f"[dim]Fetching curve {label}...[/dim]")
    client = LMFDBClient()
    curve = client.fetch_by_label(label)

    if curve is None:
        print_error(f"Could not find curve {label}")
        raise typer.Exit(1)

    # Predict
    sha_pred, uncertainty = predictor.predict(curve, return_uncertainty=True)
    regime = predictor.predict_regime(curve)[0]

    # Display results
    actual_sha = curve.sha_analytic if curve.sha_analytic else "Unknown"

    if RICH_AVAILABLE and console:
        from rich.panel import Panel

        console.print(Panel(
            f"[bold]Curve:[/bold] {label}\n"
            f"[bold]Conductor:[/bold] {curve.conductor}\n"
            f"[bold]Rank:[/bold] {curve.rank}\n"
            f"[bold]Regime:[/bold] {regime}\n\n"
            f"[bold green]Predicted Sha:[/bold green] {int(sha_pred[0])}\n"
            f"[bold]Actual Sha:[/bold] {actual_sha}\n"
            f"[dim]Uncertainty: +/- {uncertainty[0]:.2f}[/dim]",
            title="ML Sha Prediction",
            border_style="green"
        ))
    else:
        print(f"Curve: {label}")
        print(f"Predicted Sha: {int(sha_pred[0])}")
        print(f"Actual Sha: {actual_sha}")
        print(f"Regime: {regime}")


# ==========================================================================
# PINN COMMAND
# ==========================================================================

@app.command("ml-pinn")
def ml_pinn_train(
    data_source: str = typer.Option(
        "lmfdb",
        "--source", "-s",
        help="Data source"
    ),
    n_curves: int = typer.Option(
        5000,
        "--n-curves", "-n",
        help="Number of curves"
    ),
    physics_weight: float = typer.Option(
        1.0,
        "--physics-weight", "-w",
        help="Weight for physics constraint loss"
    ),
    epochs: int = typer.Option(
        200,
        "--epochs",
        help="Training epochs"
    ),
    output_dir: Path = typer.Option(
        Path("models/pinn"),
        "--output", "-o",
        help="Output directory"
    ),
):
    """
    Train Physics-Informed Neural Network respecting Law 6.

    The PINN learns exponents (alpha, gamma, delta) as parameters
    while enforcing the constraint: alpha = gamma + |delta|
    (i.e., 2/3 = 5/12 + 1/4)

    This validates whether the mathematical identity emerges
    from data-driven learning.

    Examples:
        brahims-laws ml-pinn --n-curves 10000
        brahims-laws ml-pinn --physics-weight 2.0 --epochs 300
    """
    try:
        from .ml.pinn import PhysicsInformedPredictor
        from .data.lmfdb_client import LMFDBClient
    except ImportError as e:
        print_error(f"ML dependencies not installed: {e}")
        raise typer.Exit(1)

    print_output("[bold]Training Physics-Informed Neural Network[/bold]")
    print_output(f"  Physics constraint weight: {physics_weight}")
    print_output(f"  Epochs: {epochs}")
    print_output("")

    print_output("[dim]Target: Learn exponents satisfying 2/3 = 5/12 + 1/4[/dim]\n")

    # Fetch data
    client = LMFDBClient()
    print_output("[dim]Fetching training data...[/dim]")

    curves = []
    for rank in [0, 1, 2]:
        batch = client.fetch_batch(rank=rank, limit=n_curves // 3)
        curves.extend(batch)

    print_output(f"[green]Loaded {len(curves)} curves[/green]\n")

    # Train PINN
    pinn = PhysicsInformedPredictor(learn_exponents=True)
    metrics = pinn.fit(
        curves,
        epochs=epochs,
        physics_weight=physics_weight,
        verbose=True
    )

    # Display learned exponents
    learned = pinn.get_exponents()
    theoretical = {
        'alpha': 2/3,
        'gamma': 5/12,
        'delta': -1/4
    }

    if RICH_AVAILABLE and console:
        table = Table(title="Learned vs Theoretical Exponents")
        table.add_column("Exponent", style="cyan")
        table.add_column("Learned", style="green")
        table.add_column("Theoretical", style="yellow")
        table.add_column("Error", style="magenta")

        for name in ['alpha', 'gamma', 'delta']:
            learned_val = learned[name]
            theo_val = theoretical[name]
            error = abs(learned_val - theo_val)
            table.add_row(
                name,
                f"{learned_val:.6f}",
                f"{theo_val:.6f}",
                f"{error:.6f}"
            )

        console.print(table)

        # Verify consistency
        lhs = learned['alpha']
        rhs = learned['gamma'] + abs(learned['delta'])
        consistency_error = abs(lhs - rhs)

        console.print(Panel(
            f"[bold]Learned Consistency Check:[/bold]\n"
            f"alpha = {lhs:.6f}\n"
            f"gamma + |delta| = {rhs:.6f}\n"
            f"Error: {consistency_error:.6f}\n\n"
            f"{'[green]Physics constraint satisfied![/green]' if consistency_error < 0.01 else '[yellow]Constraint approximately satisfied[/yellow]'}",
            title="Law 6 Verification",
            border_style="green" if consistency_error < 0.01 else "yellow"
        ))
    else:
        print("Learned Exponents:")
        for name in ['alpha', 'gamma', 'delta']:
            print(f"  {name}: {learned[name]:.6f} (theo: {theoretical[name]:.6f})")

    # Validate
    validation = pinn.validate_brahim_laws()
    print_output("\n[bold]Validation:[/bold]")
    print_output(f"  Consistency satisfied: {validation['consistency_satisfied']}")
    print_output(f"  Sha prediction R2: {validation.get('sha_r2', 'N/A')}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pinn.save(output_dir)
    print_output(f"\n[green]PINN saved to {output_dir}[/green]")


# ==========================================================================
# AGENT COMMAND
# ==========================================================================

@app.command("agent")
def run_agent_interactive(
    query: Optional[str] = typer.Argument(
        None,
        help="Single query to run (omit for interactive mode)"
    ),
    model: str = typer.Option(
        "gpt-4o",
        "--model", "-m",
        help="OpenAI model to use"
    ),
):
    """
    Launch the AI-powered Brahim's Laws analysis agent.

    In interactive mode, chat with an AI that can analyze curves,
    explain BSD conjecture, and explore mathematical concepts.

    Requires OPENAI_API_KEY environment variable to be set.

    Examples:
        brahims-laws agent
        brahims-laws agent "Analyze curve 11a1"
        brahims-laws agent "Explain the BSD conjecture" --model gpt-4o-mini
    """
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print_error("OPENAI_API_KEY environment variable not set")
        print_output("[yellow]Set it with: set OPENAI_API_KEY=your-key[/yellow]")
        raise typer.Exit(1)

    try:
        from .agent.curve_agent import run_agent, interactive_session
    except ImportError as e:
        print_error(f"Agent dependencies not installed: {e}")
        print_output("[yellow]Install with: pip install openai-agents[/yellow]")
        raise typer.Exit(1)

    if query:
        # Single query mode
        print_output(f"[bold blue]Query:[/bold blue] {query}")
        print_output("[dim]Processing...[/dim]\n")

        try:
            response = run_agent(query, model=model)
            print_output(response)
        except Exception as e:
            print_error(f"Agent error: {e}")
            raise typer.Exit(1)
    else:
        # Interactive mode
        try:
            interactive_session()
        except Exception as e:
            print_error(f"Session error: {e}")
            raise typer.Exit(1)


# ==========================================================================
# MAIN ENTRY POINT
# ==========================================================================

def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
