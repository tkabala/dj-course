#!/usr/bin/env python3
"""
Compare tokenizers by counting tokens produced for different text files.
For each text, tokenizers are sorted by number of tokens (fewer = better compression).
"""

from pathlib import Path
from tokenizers import Tokenizer
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich import box

console = Console()


def load_tokenizers(tokenizer_dir: Path) -> Dict[str, Tokenizer]:
    """Load all tokenizer JSON files from the specified directory."""
    tokenizers = {}

    with Progress(
        TextColumn("[bold blue]Loading tokenizers..."),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=True
    ) as progress:
        tokenizer_files = list(tokenizer_dir.glob("*.json"))
        task = progress.add_task("Loading", total=len(tokenizer_files))

        for tokenizer_file in sorted(tokenizer_files):
            try:
                tokenizer_name = tokenizer_file.stem
                tokenizers[tokenizer_name] = Tokenizer.from_file(str(tokenizer_file))
                progress.update(task, advance=1)
            except Exception as e:
                console.print(f"[red]✗[/red] Error loading {tokenizer_file.name}: {e}")

    return tokenizers


def load_text_file(file_path: Path) -> str:
    """Load text content from a file."""
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        console.print(f"[red]Error loading {file_path}: {e}[/red]")
        return ""


def tokenize_and_count(tokenizer: Tokenizer, text: str) -> int:
    """Tokenize text and return the number of tokens."""
    encoded = tokenizer.encode(text)
    return len(encoded.ids)


def get_color_for_rank(rank: int, total: int) -> str:
    """Return color based on ranking (green=best, red=worst)."""
    ratio = rank / total
    if ratio < 0.2:
        return "bright_green"
    elif ratio < 0.4:
        return "green"
    elif ratio < 0.6:
        return "yellow"
    elif ratio < 0.8:
        return "orange3"
    else:
        return "red"


def create_bar(value: int, max_value: int, width: int = 40) -> str:
    """Create a visual bar representation."""
    filled = int((value / max_value) * width) if max_value > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    return bar


def compare_tokenizers_on_text(
    tokenizers: Dict[str, Tokenizer],
    text: str,
    text_name: str
) -> List[Tuple[str, int]]:
    """
    Tokenize text with all tokenizers and return results sorted by token count.
    Returns list of (tokenizer_name, token_count) tuples.
    """
    results = []

    with Progress(
        TextColumn("[bold cyan]Tokenizing with {task.fields[tokenizer]}..."),
        BarColumn(bar_width=30),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Tokenizing", total=len(tokenizers), tokenizer="")

        for tokenizer_name, tokenizer in tokenizers.items():
            progress.update(task, tokenizer=tokenizer_name[:30])
            token_count = tokenize_and_count(tokenizer, text)
            results.append((tokenizer_name, token_count))
            progress.update(task, advance=1)

    # Sort by token count (ascending - fewer tokens is better)
    results.sort(key=lambda x: x[1])

    return results


def display_results_table(
    results: List[Tuple[str, int]],
    text_name: str,
    text_length: int
):
    """Display results in a beautiful table with bars."""

    if not results:
        return

    min_tokens = results[0][1]
    max_tokens = results[-1][1]

    # Create table
    table = Table(
        title=f"[bold cyan]{text_name}[/bold cyan] [dim]({text_length:,} chars | best: {min_tokens:,} | worst: {max_tokens:,})[/dim]",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold magenta",
        border_style="bright_blue",
        title_style="bold cyan",
        padding=(0, 1),
    )

    table.add_column("Rank", justify="right", style="cyan", width=5)
    table.add_column("Tokenizer", justify="left", style="white", width=35)
    table.add_column("Tokens", justify="right", style="yellow", width=12)
    table.add_column("Performance", justify="center", style="green", width=50)

    for rank, (tokenizer_name, token_count) in enumerate(results, 1):
        # Color based on rank
        color = get_color_for_rank(rank - 1, len(results))

        # Create visual bar
        bar = create_bar(token_count, max_tokens, width=40)

        # Calculate percentage difference from best
        diff_percent = ((token_count - min_tokens) / min_tokens * 100) if min_tokens > 0 else 0

        # Format difference with consistent width
        if rank == 1:
            diff_str = "[bold bright_green]🏆 BEST  [/bold bright_green]"
        else:
            diff_str = f"[{color}]+{diff_percent:5.1f}%[/{color}]"

        # Add row with colored bar
        table.add_row(
            f"[{color}]{rank}[/{color}]",
            f"[bold]{tokenizer_name}[/bold]",
            f"[{color}]{token_count:,}[/{color}]",
            f"[{color}]{bar}[/{color}] {diff_str}"
        )

    # Display the table in a panel
    panel = Panel(
        table,
        border_style="bright_blue",
        padding=(0, 1)
    )
    console.print(panel)


def main():
    # Display header
    console.print("[bold bright_cyan]🚀 TOKENIZER PERFORMANCE COMPARISON[/bold bright_cyan]")

    # Define paths
    tokenizer_dir = Path("tokenizers")
    text_files = [
        Path("../korpus-wolnelektury/pan-tadeusz-ksiega-1.txt"),
        Path("../korpus-mini/fryderyk-chopin-wikipedia.txt"),
        Path("../korpus-mini/the-pickwick-papers-gutenberg.txt"),
    ]

    # Load all tokenizers
    tokenizers = load_tokenizers(tokenizer_dir)
    if not tokenizers:
        console.print("[bold red]✗ No tokenizers found![/bold red]")
        return
    console.print(f"[bold green]✓[/bold green] Loaded {len(tokenizers)} tokenizers\n")

    for text_file in text_files:
        if not text_file.exists():
            console.print(f"[red]✗ File not found: {text_file}[/red]")
            continue

        text = load_text_file(text_file)
        if not text:
            console.print("[red]✗ Empty or unreadable file[/red]")
            continue

        # Compare tokenizers
        results = compare_tokenizers_on_text(tokenizers, text, text_file.name)

        # Display results
        display_results_table(results, text_file.name, len(text))

    console.print("[bold green]✓ Analysis complete![/bold green]")


if __name__ == "__main__":
    main()
