import numpy as np
import logging
from gensim.models import Word2Vec
from tokenizers import Tokenizer
import os
import glob
import re
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns

# Ustawienie logowania dla gensim (wyłącz szczegółowe logi)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

console = Console()

# --- FUNKCJA: WYODRĘBNIANIE NAZWY TOKENIZERA Z PLIKU MODELU ---

def extract_tokenizer_name(model_filename: str) -> str:
    """
    Wyodrębnia nazwę tokenizera z nazwy pliku modelu.

    Wzorzec: embedding_word2vec_cbow_{corpora}_{tokenizer_name}_v{vector_length}_w{window_size}_e{epochs}_s{sample_rate}.model

    Args:
        model_filename: Nazwa pliku modelu (np. 'embedding_word2vec_cbow_ALL_tokenizer-wolnelektury-64k_v20_w6_e20_s0.01.model')

    Returns:
        Nazwa tokenizera (np. 'tokenizer-wolnelektury-64k')
    """
    # Pattern: embedding_word2vec_cbow_{corpora}_t-{tokenizer_name}_v{vector_length}_w{window_size}_e{epochs}_s{sample_rate}.model
    # Używamy regex do wyodrębnienia tokenizer_name po prefiksie 't-'
    pattern = r'embedding_word2vec_cbow_.+?_t-(.+?)_v\d+'
    match = re.search(pattern, model_filename)

    if match:
        return match.group(1)
    else:
        raise ValueError(f"Nie można wyodrębnić nazwy tokenizera z pliku: {model_filename}")


def load_tokenizer_for_model(model_filename: str) -> Tokenizer:
    """
    Wczytuje odpowiedni tokenizer dla danego pliku modelu.

    Args:
        model_filename: Nazwa pliku modelu

    Returns:
        Obiekt Tokenizer
    """
    tokenizer_name = extract_tokenizer_name(model_filename)
    tokenizer_path = f"../tokenizer/tokenizers/{tokenizer_name}.json"

    try:
        console.print(f"[cyan]Ładowanie tokenizera:[/cyan] {tokenizer_name}.json")
        return Tokenizer.from_file(tokenizer_path)
    except FileNotFoundError:
        console.print(f"[red]BŁĄD:[/red] Nie znaleziono pliku tokenizera '{tokenizer_path}'")
        raise

# --- FUNKCJA: OBLICZANIE WEKTORA DLA CAŁEGO SŁOWA ---

def get_word_vector_and_similar(word: str, tokenizer: Tokenizer, model: Word2Vec, topn: int = 20):
    # Tokenizacja słowa na tokeny podwyrazowe
    # Używamy .encode(), aby otoczyć słowo spacjami, co imituje kontekst w zdaniu
    # Ważne: tokenizator BPE/SentencePiece musi widzieć spację, by dodać prefiks '_'
    encoding = tokenizer.encode(" " + word + " ")
    word_tokens = [t.strip() for t in encoding.tokens if t.strip()] # Usuń puste tokeny

    # Usuwamy tokeny początku/końca sekwencji, jeśli zostały dodane przez tokenizator
    if word_tokens and word_tokens[0] in ['[CLS]', '<s>', '<s>', 'Ġ']:
        word_tokens = word_tokens[1:]
    if word_tokens and word_tokens[-1] in ['[SEP]', '</s>', '</s>']:
        word_tokens = word_tokens[:-1]

    valid_vectors = []
    missing_tokens = []

    # 1. Zbieranie wektorów dla każdego tokenu
    for token in word_tokens:
        if token in model.wv:
            # Użycie tokenu ze spacją (np. '_ryż') lub bez (np. 'szlach')
            valid_vectors.append(model.wv[token])
        else:
            # W tym miejscu token może być zbyt rzadki i pominięty przez MIN_COUNT
            missing_tokens.append(token)

    if not valid_vectors:
        # Kod do obsługi, gdy żaden token nie ma wektora
        return None, None

    # 2. Uśrednianie wektorów
    # Wektor dla całego słowa to średnia wektorów jego tokenów składowych
    word_vector = np.mean(valid_vectors, axis=0)

    # 3. Znalezienie najbardziej podobnych tokenów
    similar_words = model.wv.most_similar(
        positive=[word_vector],
        topn=topn
    )

    return word_vector, similar_words

# --- WERYFIKACJA MODELI ---

def verify_model(model_path: str, tokenizer: Tokenizer):
    """Weryfikuje pojedynczy model Word2Vec i zwraca wyniki."""
    results = {
        'model_path': model_path,
        'model_name': os.path.basename(model_path),
        'error': None,
        'vocab_size': 0,
        'vector_size': 0,
        'word_similarities': {},
        'analogy_results': []
    }

    try:
        model = Word2Vec.load(model_path)
        results['vocab_size'] = len(model.wv.index_to_key)
        results['vector_size'] = model.wv.vector_size
    except Exception as e:
        results['error'] = str(e)
        return results

    # Przykłady słów do testowania
    words_to_test = ['wojsko', 'szlachta', 'choroba', 'król']

    for word in words_to_test:
        word_vector, similar_tokens = get_word_vector_and_similar(word, tokenizer, model, topn=10)

        if word_vector is not None:
            results['word_similarities'][word] = {
                'vector': word_vector,
                'similar_tokens': similar_tokens
            }
        else:
            results['word_similarities'][word] = None

    # Analogia wektorowa
    tokens_analogy = ['dziecko', 'kobieta']

    if tokens_analogy[0] in model.wv and tokens_analogy[1] in model.wv:
        similar_to_combined = model.wv.most_similar(
            positive=tokens_analogy,
            topn=10
        )
        results['analogy_results'] = similar_to_combined

    return results


def create_verification_panel(title, similarities, word_vector=None, panel_width=25):
    """Tworzy panel dla pojedynczej weryfikacji (słowo lub analogia)."""
    content = []

    if similarities:
        for i, (token, score) in enumerate(similarities[:10], 1):
            # Skróć długie tokeny
            token_display = token[:15] if len(token) > 15 else token
            content.append(f"{i:2}. {token_display:15} {score:.3f}")
    else:
        content.append("[dim]Brak danych[/dim]")

    panel_content = "\n".join(content)

    return Panel(
        panel_content,
        title=f"[bold cyan]{title}[/bold cyan]",
        border_style="blue",
        padding=(0, 1),
        width=panel_width
    )


def display_results(all_results):
    """Wyświetla wyniki weryfikacji - każdy model w osobnym wierszu, weryfikacje w kolumnach."""

    console.print("\n")

    terminal_width = console.width
    # 5 weryfikacji + padding
    panel_width = (terminal_width - 15) // 5

    words_to_test = ['wojsko', 'szlachta', 'choroba', 'król']

    # Dla każdego modelu wyświetl wiersz z 5 weryfikacjami
    for idx, result in enumerate(all_results):
        # Nagłówek modelu
        if result['error']:
            console.print(f"\n[red]{result['model_name']}[/red]")
            console.print(f"[red]BŁĄD: {result['error']}[/red]")
            continue

        console.print(f"\n[bold yellow]{result['model_name']}[/bold yellow]")
        console.print(f"[dim]Słownik: {result['vocab_size']:,} | Wymiar: {result['vector_size']}[/dim]")

        # Utwórz 5 paneli dla tego modelu (4 słowa + 1 analogia)
        verification_panels = []

        # Panele dla słów testowych
        for word in words_to_test:
            word_data = result['word_similarities'].get(word)
            if word_data is not None and isinstance(word_data, dict):
                similarities = word_data.get('similar_tokens')
                word_vector = word_data.get('vector')
            else:
                similarities = None
                word_vector = None
            panel = create_verification_panel(word, similarities, word_vector, panel_width)
            verification_panels.append(panel)

        # Panel dla analogii
        analogy_panel = create_verification_panel(
            "dziecko+kobieta",
            result['analogy_results'],
            word_vector=None,
            panel_width=panel_width
        )
        verification_panels.append(analogy_panel)

        # Wyświetl wszystkie 5 paneli w jednym wierszu
        console.print(Columns(verification_panels, equal=False, expand=False, padding=(0, 1)))

        # Separator między modelami
        if idx < len(all_results) - 1:
            console.print("")


# --- MAIN: Znajdź i zweryfikuj wszystkie modele ---

if __name__ == "__main__":
    # Szukamy wszystkich plików .model w bieżącym katalogu
    model_files = sorted(glob.glob("*.model"))

    if not model_files:
        console.print("[red]BŁĄD:[/red] Nie znaleziono żadnych plików .model w bieżącym katalogu.")
        exit(1)

    console.print(f"\n[bold green]Znaleziono {len(model_files)} modeli do weryfikacji[/bold green]")
    for model_file in model_files:
        console.print(f"  [cyan]•[/cyan] {model_file}")

    console.print("\n[yellow]Weryfikacja modeli w toku...[/yellow]")

    # Weryfikacja każdego modelu z odpowiednim tokenizerem
    all_results = []
    for model_file in model_files:
        try:
            # Wczytaj odpowiedni tokenizer dla tego modelu
            tokenizer = load_tokenizer_for_model(model_file)
            result = verify_model(model_file, tokenizer)
        except Exception as e:
            # Jeśli nie udało się wczytać tokenizera, zapisz błąd w wynikach
            console.print(f"[red]Błąd przy przetwarzaniu {model_file}: {str(e)}[/red]")
            result = {
                'model_path': model_file,
                'model_name': os.path.basename(model_file),
                'error': f"Błąd tokenizera: {str(e)}",
                'vocab_size': 0,
                'vector_size': 0,
                'word_similarities': {},
                'analogy_results': []
            }
        all_results.append(result)

    # Wyświetlenie wyników
    display_results(all_results)

    console.print(f"\n[bold green]✓ WERYFIKACJA ZAKOŃCZONA[/bold green] - Zweryfikowano {len(model_files)} modeli\n")
