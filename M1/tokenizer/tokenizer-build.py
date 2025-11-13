#!/usr/bin/env python3
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from corpora import get_corpus_file

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Build a BPE tokenizer from corpus files")
    parser.add_argument("name", help="Name for the tokenizer (used for output path)")
    parser.add_argument("--corpora", required=True, help="Corpora name to use (e.g., WOLNE_LEKTURY)")
    parser.add_argument("--filter", required=True, help="File filter pattern (e.g., *.txt)")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size (default: 32000)")

    args = parser.parse_args()

    # Build output path
    tokenizer_output_file = f"tokenizers/{args.name}.json"

    # 1. Initialize the Tokenizer (BPE model)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # 2. Set the pre-tokenizer (e.g., split on spaces)
    tokenizer.pre_tokenizer = Whitespace()

    # 3. Set the Trainer
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        vocab_size=args.vocab_size,
        min_frequency=2
    )

    # Get corpus files based on parameters
    FILES = [str(f) for f in get_corpus_file(args.corpora, args.filter)]
    print(f"Training on files: {FILES}")

    # 4. Train the Tokenizer
    tokenizer.train(FILES, trainer=trainer)

    # 5. Save the vocabulary and tokenization rules
    tokenizer.save(tokenizer_output_file)
    print(f"Tokenizer saved to: {tokenizer_output_file}")

    # Test the tokenizer
    for txt in [
        "Litwo! Ojczyzno moja! ty jesteś jak zdrowie.",
        "Jakże mi wesoło!",
        "Jeśli wolisz mieć pełną kontrolę nad tym, które listy są łączone (a to jest bezpieczniejsze, gdy słownik może zawierać inne klucze), po prostu prześlij listę list do spłaszczenia.",
    ]:
        encoded = tokenizer.encode(txt)
        print("Zakodowany tekst:", encoded.tokens)
        print("ID tokenów:", encoded.ids)

if __name__ == "__main__":
    main()
