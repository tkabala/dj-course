from tokenizers import Tokenizer
import tiktoken
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.layout import Layout
from rich import box

# ============================================================================
# CONSTANTS
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_DIR = os.path.join(SCRIPT_DIR, 'tokenizers')
SAMPLES_DIR = os.path.join(SCRIPT_DIR, 'samples')

FORMATS = ['json', 'nows-json', 'yaml', 'toon', 'tron']
SAMPLE_NAMES = ['placeholder', 'recipe', 'models', 'arch', 'photos']

# Color mapping for each format (consistent across all charts)
FORMAT_COLORS = {
    'json': 'blue',
    'nows-json': 'cyan',
    'yaml': 'magenta',
    'toon': 'yellow',
    'tron': 'red'
}

# ============================================================================
# TOKENIZER ADAPTER CLASSES
# ============================================================================

class TokenizerAdapter(ABC):
    """Abstract base class for tokenizer adapters"""

    @abstractmethod
    def encode(self, text: str) -> list:
        """Encode text and return token IDs"""
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Get number of tokens without returning IDs"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return tokenizer name"""
        pass


class HuggingFaceTokenizerAdapter(TokenizerAdapter):
    """Adapter for HuggingFace tokenizers"""

    def __init__(self, tokenizer, name: str):
        self._tokenizer = tokenizer
        self._name = name

    def encode(self, text: str) -> list:
        return self._tokenizer.encode(text).ids

    def get_token_count(self, text: str) -> int:
        return len(self.encode(text))

    @property
    def name(self) -> str:
        return self._name


class TikTokenAdapter(TokenizerAdapter):
    """Adapter for OpenAI's tiktoken tokenizers"""

    def __init__(self, encoding_name: str):
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._name = f"tiktoken-{encoding_name}"

    def encode(self, text: str) -> list:
        return self._encoding.encode(text)

    def get_token_count(self, text: str) -> int:
        return len(self.encode(text))

    @property
    def name(self) -> str:
        return self._name


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BenchmarkResult:
    """Stores the result of a single tokenization benchmark"""
    tokenizer_name: str
    sample_name: str
    format_name: str
    token_count: int
    file_size_bytes: int
    error: Optional[str] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_sample(sample_name: str, samples_dir: str) -> dict:
    """Load all format variants of a sample

    Returns dict with format names as keys and file contents as values.
    Missing files will have None as value.
    """
    data = {}

    for format_name in FORMATS:
        # Determine file path
        if format_name == 'nows-json':
            file_path = os.path.join(samples_dir, f"{sample_name}-nows.json")
        else:
            file_path = os.path.join(samples_dir, f"{sample_name}.{format_name}")

        # Try to load file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data[format_name] = f.read()
        except FileNotFoundError:
            data[format_name] = None

    return data


def load_all_tokenizers(tokenizer_dir: str) -> list:
    """Load all HuggingFace tokenizers + tiktoken tokenizers

    Returns list of TokenizerAdapter instances
    """
    tokenizers = []
    console = Console()

    # Load HuggingFace tokenizers
    if os.path.isdir(tokenizer_dir):
        for filename in sorted(os.listdir(tokenizer_dir)):
            if filename.endswith('.json'):
                name = filename[:-5]  # remove .json
                full_path = os.path.join(tokenizer_dir, filename)
                try:
                    hf_tokenizer = Tokenizer.from_file(full_path)
                    tokenizers.append(HuggingFaceTokenizerAdapter(hf_tokenizer, name))
                    console.print(f"[green]✓[/green] Loaded HF tokenizer: {name}")
                except Exception as e:
                    console.print(f"[red]✗[/red] Failed to load {name}: {e}")

    # Add tiktoken tokenizers
    tiktoken_models = [
        "o200k_base",   # GPT-4o
        "cl100k_base",  # GPT-4, GPT-3.5-turbo
    ]

    for encoding_name in tiktoken_models:
        try:
            tokenizers.append(TikTokenAdapter(encoding_name))
            console.print(f"[green]✓[/green] Loaded tiktoken: {encoding_name}")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load tiktoken {encoding_name}: {e}")

    return tokenizers


# ============================================================================
# BENCHMARK ENGINE
# ============================================================================

def run_benchmark(tokenizers: list, sample_names: list, samples_dir: str) -> list:
    """Run comprehensive benchmark across all tokenizers, samples, and formats

    Returns list of BenchmarkResult instances
    """
    results = []
    console = Console()

    total_tests = len(tokenizers) * len(sample_names) * len(FORMATS)
    current = 0

    with console.status("[bold green]Running benchmark...") as status:
        for sample_name in sample_names:
            sample_data = load_sample(sample_name, samples_dir)

            for tokenizer in tokenizers:
                for format_name in FORMATS:
                    current += 1
                    status.update(f"[bold green]Processing {current}/{total_tests}: {tokenizer.name} × {sample_name} × {format_name}")

                    content = sample_data.get(format_name)

                    # Handle missing files
                    if content is None:
                        results.append(BenchmarkResult(
                            tokenizer.name,
                            sample_name,
                            format_name,
                            0,
                            0,
                            error="File not found"
                        ))
                        continue

                    # Tokenize and measure
                    try:
                        token_count = tokenizer.get_token_count(content)
                        file_size = len(content.encode('utf-8'))

                        results.append(BenchmarkResult(
                            tokenizer.name,
                            sample_name,
                            format_name,
                            token_count,
                            file_size
                        ))
                    except Exception as e:
                        results.append(BenchmarkResult(
                            tokenizer.name,
                            sample_name,
                            format_name,
                            0,
                            0,
                            error=str(e)
                        ))

    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def calculate_format_averages(results: list) -> dict:
    """Calculate average token count per format across all tokenizers and samples"""
    format_totals = {fmt: [] for fmt in FORMATS}

    for result in results:
        if result.error is None and result.token_count > 0:
            format_totals[result.format_name].append(result.token_count)

    format_averages = {}
    for fmt, counts in format_totals.items():
        if counts:
            format_averages[fmt] = sum(counts) / len(counts)
        else:
            format_averages[fmt] = 0

    return format_averages


def calculate_tokenizer_averages(results: list) -> dict:
    """Calculate average token count per tokenizer across all samples and formats"""
    tokenizer_totals = {}

    for result in results:
        if result.error is None and result.token_count > 0:
            if result.tokenizer_name not in tokenizer_totals:
                tokenizer_totals[result.tokenizer_name] = []
            tokenizer_totals[result.tokenizer_name].append(result.token_count)

    tokenizer_averages = {}
    for tok_name, counts in tokenizer_totals.items():
        if counts:
            tokenizer_averages[tok_name] = sum(counts) / len(counts)

    return tokenizer_averages


def create_bar_chart(title: str, data: dict, max_bar_width: int = 50,
                     highlight_min: bool = True) -> Table:
    """Create a horizontal bar chart as a Rich Table

    Args:
        title: Chart title
        data: Dict of {label: value}
        max_bar_width: Maximum width of bars in characters
        highlight_min: If True, highlight the minimum value as winner
    """
    table = Table(title=title, show_header=False, box=box.ROUNDED,
                  title_style="bold cyan")

    table.add_column("Label", style="yellow", width=20)
    table.add_column("Bar", no_wrap=False)
    table.add_column("Value", justify="right", style="white", width=20)

    if not data:
        return table

    # Sort data by value (descending - winner at bottom)
    sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)

    max_value = max(data.values()) if data.values() else 1
    min_value = min(data.values()) if data.values() else 0

    for label, value in sorted_data:
        # Calculate bar length
        if max_value > 0:
            bar_length = int((value / max_value) * max_bar_width)
        else:
            bar_length = 0

        # Calculate percentage relative to max
        percentage = (value / max_value * 100) if max_value > 0 else 0

        # Determine if this is the winner
        is_winner = highlight_min and (value == min_value) and (value > 0)

        # Get format-specific color or default
        format_color = FORMAT_COLORS.get(label.lower(), 'cyan')

        # Format value with space as thousand separator
        value_formatted = f"{value:,.0f}".replace(',', ' ')

        # Create bar with format-specific color
        if is_winner:
            bar = f"[bold {format_color}]{'█' * bar_length}[/bold {format_color}]"
            value_str = f"[bold {format_color}]{value_formatted} ({percentage:.1f}%)[/bold {format_color}]"
            label_str = f"[bold {format_color}]{label.upper()}[/bold {format_color}]"
        else:
            bar = f"[{format_color}]{'█' * bar_length}[/{format_color}]"
            value_str = f"[{format_color}]{value_formatted} ({percentage:.1f}%)[/{format_color}]"
            label_str = f"[{format_color}]{label.upper()}[/{format_color}]"

        table.add_row(label_str, bar, value_str)

    return table


def create_tokenizer_sample_chart(tokenizer_name: str, sample_name: str, results: list) -> Table:
    """Create a bar chart comparing formats for a specific tokenizer and sample"""
    # Filter results for this tokenizer and sample
    filtered_results = [r for r in results
                       if r.tokenizer_name == tokenizer_name
                       and r.sample_name == sample_name
                       and r.error is None
                       and r.token_count > 0]

    # Get token count per format
    format_counts = {}
    for result in filtered_results:
        format_counts[result.format_name] = result.token_count

    return create_bar_chart(f"{sample_name}", format_counts,
                           max_bar_width=30, highlight_min=True)


def display_overall_format_efficiency(results: list, console: Console):
    """Display overall format efficiency chart"""
    format_averages = calculate_format_averages(results)
    chart = create_bar_chart("Overall Format Efficiency (Lower = Better)",
                            format_averages, max_bar_width=50, highlight_min=True)
    console.print()
    console.print(chart)
    console.print()


def display_sample_comparisons(results: list, console: Console):
    """Display per-tokenizer per-sample format comparison charts"""
    # Get unique tokenizer names
    tokenizer_names = sorted(set(r.tokenizer_name for r in results))

    console.print(Panel("Per-Tokenizer Per-Sample Format Comparison", style="bold magenta"))
    console.print()

    for tokenizer_name in tokenizer_names:
        # Create header for this tokenizer
        console.print(f"[bold yellow]{'=' * 80}[/bold yellow]")
        console.print(f"[bold cyan]{tokenizer_name}[/bold cyan]")
        console.print(f"[bold yellow]{'=' * 80}[/bold yellow]")
        console.print()

        # Create charts for all samples for this tokenizer
        charts = []
        for sample_name in SAMPLE_NAMES:
            chart = create_tokenizer_sample_chart(tokenizer_name, sample_name, results)
            charts.append(chart)

        # Display charts in pairs (2 per row)
        for i in range(0, len(charts), 2):
            if i + 1 < len(charts):
                console.print(Columns([charts[i], charts[i + 1]], equal=True, expand=True))
            else:
                console.print(charts[i])
            console.print()

        console.print()


def display_tokenizer_ranking(results: list, console: Console):
    """Display tokenizer efficiency ranking"""
    tokenizer_averages = calculate_tokenizer_averages(results)
    chart = create_bar_chart("Tokenizer Efficiency Ranking (Lower = Better)",
                            tokenizer_averages, max_bar_width=50, highlight_min=True)
    console.print()
    console.print(chart)
    console.print()


def display_errors(results: list, console: Console):
    """Display error summary if any errors occurred"""
    errors = [r for r in results if r.error is not None]

    if not errors:
        console.print("[bold green]✓ No errors encountered[/bold green]")
        return

    table = Table(title="Errors Encountered", show_header=True,
                 header_style="bold red", box=box.ROUNDED)
    table.add_column("Tokenizer", style="yellow")
    table.add_column("Sample", style="cyan")
    table.add_column("Format", style="magenta")
    table.add_column("Error", style="red")

    for error_result in errors:
        table.add_row(
            error_result.tokenizer_name,
            error_result.sample_name,
            error_result.format_name,
            error_result.error
        )

    console.print()
    console.print(Panel(table, border_style="red"))


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main benchmark execution"""
    console = Console()

    # Print header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Tokenizer Benchmark Tool[/bold cyan]\n"
        "Testing 7 tokenizers × 5 samples × 5 formats",
        border_style="cyan"
    ))
    console.print()

    # Check directories
    if not os.path.isdir(TOKENIZER_DIR):
        console.print(f"[red]Error: Tokenizer directory not found at {TOKENIZER_DIR}[/red]")
        return

    if not os.path.isdir(SAMPLES_DIR):
        console.print(f"[red]Error: Samples directory not found at {SAMPLES_DIR}[/red]")
        return

    # Load tokenizers
    console.print("[bold cyan]Loading tokenizers...[/bold cyan]")
    tokenizers = load_all_tokenizers(TOKENIZER_DIR)

    if not tokenizers:
        console.print("[red]Error: No tokenizers loaded[/red]")
        return

    console.print(f"[green]✓ Loaded {len(tokenizers)} tokenizers[/green]")
    console.print()

    # Run benchmark
    results = run_benchmark(tokenizers, SAMPLE_NAMES, SAMPLES_DIR)

    # Display visualizations
    console.print()
    console.print("=" * 80)
    console.print()

    display_overall_format_efficiency(results, console)

    console.print("=" * 80)
    console.print()

    display_sample_comparisons(results, console)

    console.print("=" * 80)

    display_tokenizer_ranking(results, console)

    console.print("=" * 80)
    console.print()

    display_errors(results, console)

    console.print()
    console.print("[bold green]✓ Benchmark complete![/bold green]")
    console.print()


if __name__ == "__main__":
    main()
