import numpy as np
import json
import logging
import argparse
from gensim.models import Word2Vec
from tokenizers import Tokenizer
import os
import glob
# import z corpora (zakładam, że jest to plik pomocniczy)
from corpora import CORPORA_FILES # type: ignore 

# Ustawienie logowania dla gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- PARSOWANIE PARAMETRÓW WIERSZA POLECEŃ ---
parser = argparse.ArgumentParser(description='Train Word2Vec CBOW embeddings')
parser.add_argument('--corpora', type=str, default='ALL',
                    help='Corpus to use (e.g., ALL, PAN_TADEUSZ, WOLNELEKTURY)')
parser.add_argument('--tokenizer', type=str, default='tokenizer-wolnelektury-64k',
                    help='Tokenizer filename without extension (will be loaded from ../tokenizer/tokenizers/)')
parser.add_argument('--vector-length', type=int, default=20,
                    help='Embedding vector dimension (default: 20)')
parser.add_argument('--window-size', type=int, default=6,
                    help='Context window size (default: 6)')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of training epochs (default: 20)')
parser.add_argument('--sample-rate', type=float, default=0.01,
                    help='Subsampling rate for frequent words (default: 0.01)')

args = parser.parse_args()

# --- KONFIGURACJA ŚCIEŻEK I PARAMETRÓW ---
files = CORPORA_FILES[args.corpora]

TOKENIZER_FILE = f"../tokenizer/tokenizers/{args.tokenizer}.json"

# Use tokenizer name as-is (already without extension)
tokenizer_name = args.tokenizer

# Build descriptive output filenames
output_base = f"cbow_{args.corpora}_t-{tokenizer_name}_v{args.vector_length}_w{args.window_size}_e{args.epochs}_s{args.sample_rate}"
OUTPUT_TENSOR_FILE = f"embedding_tensor_{output_base}.npy"
OUTPUT_MAP_FILE = f"embedding_token_to_index_map_{output_base}.json"
OUTPUT_MODEL_FILE = f"embedding_word2vec_{output_base}.model"

# Parametry treningu Word2Vec (CBOW)
VECTOR_LENGTH = args.vector_length
WINDOW_SIZE = args.window_size
MIN_COUNT = 2
WORKERS = 4
EPOCHS = args.epochs
SAMPLE_RATE = args.sample_rate
SG_MODE = 0 # 0 dla CBOW, 1 dla Skip-gram

try:
    print(f"Ładowanie tokenizera z pliku: {TOKENIZER_FILE}")
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
except Exception as e:
    print(f"BŁĄD: Nie znaleziono pliku '{TOKENIZER_FILE}'.")
    tokenizer_dir = "../tokenizer/tokenizers/"
    if os.path.exists(tokenizer_dir):
        available_tokenizers = glob.glob(os.path.join(tokenizer_dir, "*.json"))
        if available_tokenizers:
            print("\nDostępne tokenizery:")
            for tok_path in sorted(available_tokenizers):
                tok_name = os.path.basename(tok_path).replace('.json', '')
                print(f"  - {tok_name}")
            print("\nUwaga: Podaj nazwę tokenizera BEZ rozszerzenia .json")
        else:
            print(f"Brak tokenierów w katalogu '{tokenizer_dir}'.")
    else:
        print(f"Katalog '{tokenizer_dir}' nie istnieje.")
    raise

# loading r& aggregating aw sentences from files
def aggregate_raw_sentences(files):
    raw_sentences = []
    print("Wczytywanie tekstu z plików...")
    print(f"Liczba plików do wczytania: {len(files)}")
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                raw_sentences.extend(lines)
        except FileNotFoundError:
            print(f"OSTRZEŻENIE: Nie znaleziono pliku '{file}'. Pomijam.")
            continue

    if not raw_sentences:
        print("BŁĄD: Pliki wejściowe są puste lub nie zostały wczytane.")
        exit()
    return raw_sentences

raw_sentences = aggregate_raw_sentences(files)

# Tokenizacja całej partii zdań przy użyciu tokenizera BPE
print(f"Tokenizacja {len(raw_sentences)} zdań...")
encodings = tokenizer.encode_batch(raw_sentences)

# Konwersja obiektów Encoding na listę list stringów (tokenów)
tokenized_sentences = [
    encoding.tokens for encoding in encodings
]
print(f"Przygotowano {len(tokenized_sentences)} sekwencji do treningu.")

# --- ETAP 2: Trening Word2Vec (CBOW) ---

print("\n--- Rozpoczynanie Treningu Word2Vec (CBOW) ---")
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=VECTOR_LENGTH,
    window=WINDOW_SIZE,
    min_count=MIN_COUNT,
    workers=WORKERS,
    sg=SG_MODE,  # 0: CBOW
    epochs=EPOCHS,
    sample=SAMPLE_RATE,
)
print("Trening zakończony pomyślnie.")

# --- ETAP 3: Eksport i Zapis Wyników ---

# Eksport tensora embeddingowego
embedding_matrix_np = model.wv.vectors
embedding_matrix_tensor = np.array(embedding_matrix_np, dtype=np.float32)

print(f"\nKształt finalnego tensora: {embedding_matrix_tensor.shape} (Tokeny x Wymiar)")

# 1. Zapisanie tensora NumPy (.npy)
np.save(OUTPUT_TENSOR_FILE, embedding_matrix_tensor)
print(f"Tensor embeddingowy zapisany jako: '{OUTPUT_TENSOR_FILE}'.")

# 2. Zapisanie mapowania tokenów na indeksy
token_to_index = {token: model.wv.get_index(token) for token in model.wv.index_to_key}
with open(OUTPUT_MAP_FILE, "w", encoding="utf-8") as f:
    json.dump(token_to_index, f, ensure_ascii=False, indent=4)
print(f"Mapa tokenów do indeksów zapisana jako: '{OUTPUT_MAP_FILE}'.")

# 3. Zapisanie całego modelu gensim (opcjonalne, ale zalecane)
model.save(OUTPUT_MODEL_FILE)
print(f"Pełny model Word2Vec zapisany jako: '{OUTPUT_MODEL_FILE}'.")