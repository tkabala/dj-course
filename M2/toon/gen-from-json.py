import json
import yaml
import sys
import os
import subprocess
from typing import Any, Dict

# --- Funkcje pomocnicze do wczytywania danych ---

def load_json_data(input_path: str) -> Dict[str, Any] | None:
    """
    Wczytuje dane z pliku JSON i obsługuje błędy.
    
    :param input_path: Ścieżka do pliku wejściowego JSON.
    :return: Wczytane dane lub None w przypadku błędu.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"Błąd: Plik wejściowy '{input_path}' nie został znaleziony.", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        print(f"Błąd: Plik '{input_path}' zawiera niepoprawny format JSON.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas czytania pliku JSON: {e}", file=sys.stderr)
        return None

# --- Funkcje konwertujące i zapisujące ---

def json_to_yaml(data: Dict[str, Any], output_path: str, input_path: str) -> bool:
    """
    Zapisuje dane w formacie YAML.
    
    :param data: Dane do zapisania.
    :param output_path: Ścieżka do pliku wyjściowego YAML.
    :param input_path: Ścieżka do pliku wejściowego JSON (tylko do komunikatów).
    :return: True jeśli zapis się powiódł, False w przeciwnym razie.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as yaml_file:
            # Użycie safe_dump z odpowiednimi opcjami formatowania
            yaml.safe_dump(
                data, 
                yaml_file, 
                allow_unicode=True, 
                default_flow_style=False, 
                sort_keys=False, 
                indent=2
            )
        print(f"INFO: Successfully created YAML file: '{output_path}'.")
        return True
    except Exception as e:
        print(f"Błąd podczas zapisu do pliku YAML '{output_path}': {e}", file=sys.stderr)
        return False

def json_to_nows_json(data: Dict[str, Any], output_path: str) -> bool:
    """
    Zapisuje dane w formacie JSON bez białych znaków (jako jedna linia).
    
    :param data: Dane do zapisania.
    :param output_path: Ścieżka do pliku wyjściowego -nows.json.
    :return: True jeśli zapis się powiódł, False w przeciwnym razie.
    """
    try:
        # json.dumps z separatorami (',', ':') oraz ensure_ascii=False usuwa zbędne białe znaki
        compact_json = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(compact_json)
        
        print(f"INFO: Successfully created no-whitespace JSON file: '{output_path}'.")
        return True
    except Exception as e:
        print(f"Błąd podczas zapisu do pliku no-whitespace JSON '{output_path}': {e}", file=sys.stderr)
        return False

def json_to_toon_cli(input_json_path: str, output_toon_path: str) -> bool:
    """
    Konwertuje plik JSON do TOON za pomocą narzędzia npx @toon-format/cli.

    Uwaga: Ta funkcja wymaga zainstalowanego Node.js i dostępności 'npx' w ścieżce systemowej.
    
    :param input_json_path: Ścieżka do pliku wejściowego JSON.
    :param output_toon_path: Ścieżka do pliku wyjściowego .toon.
    :return: True jeśli konwersja się powiodła, False w przeciwnym razie.
    """
    # Polecenie dla CLI: npx @toon-format/cli <input> --no-strict -o <output>
    # Format TOON jest domyślnie tworzony, gdy wejściem jest JSON, a wyjściem nie jest JSON.
    command = [
        'npx', 
        '@toon-format/cli', 
        input_json_path, 
        '--no-strict', # Opcja z Twojego zapytania
        '-o', 
        output_toon_path
    ]
    
    try:
        # Wywołanie komendy systemowej. capture_output=True przechwytuje stdout/stderr.
        result = subprocess.run(
            command, 
            check=True,  # Wywołuje wyjątek CalledProcessError, jeśli kod powrotu != 0
            capture_output=True, 
            text=True
        )
        print(f"INFO: Successfully created TOON file: '{output_toon_path}'.")
        # print(f"DEBUG CLI Output:\n{result.stdout.strip()}") # Uncomment for debugging
        return True
    except subprocess.CalledProcessError as e:
        print(f"Błąd podczas konwersji do TOON (CLI zwróciło błąd):", file=sys.stderr)
        print(f"  Kod powrotu: {e.returncode}", file=sys.stderr)
        print(f"  Stderr:\n{e.stderr.strip()}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("Błąd: Polecenie 'npx' nie zostało znalezione.", file=sys.stderr)
        print("Upewnij się, że masz zainstalowane Node.js i 'npx' jest dostępne w PATH.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas wywoływania CLI TOON: {e}", file=sys.stderr)
        return False

def json_to_tron_cli(input_json_path: str, output_tron_path: str) -> bool:
    """
    Konwertuje plik JSON do TRON za pomocą lokalnego wrappera Node.js.

    Uwaga: Ta funkcja wymaga zainstalowanego Node.js, 'node' w ścieżce systemowej,
    oraz zainstalowanych zależności Node.js (npm install).

    :param input_json_path: Ścieżka do pliku wejściowego JSON.
    :param output_tron_path: Ścieżka do pliku wyjściowego .tron.
    :return: True jeśli konwersja się powiodła, False w przeciwnym razie.
    """
    # Polecenie dla naszego własnego wrappera CLI
    # Użycie node dla lokalnego skryptu
    wrapper_script = os.path.join(os.path.dirname(__file__), 'tron-cli-wrapper.js')

    command = [
        'node',
        wrapper_script,
        input_json_path,
        '-o',
        output_tron_path
    ]

    try:
        # Wywołanie komendy systemowej
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"INFO: Successfully created TRON file: '{output_tron_path}'.")
        # print(f"DEBUG CLI Output:\n{result.stdout.strip()}") # Uncomment for debugging
        return True
    except subprocess.CalledProcessError as e:
        print(f"Błąd podczas konwersji do TRON (CLI zwróciło błąd):", file=sys.stderr)
        print(f"  Kod powrotu: {e.returncode}", file=sys.stderr)
        print(f"  Stderr:\n{e.stderr.strip()}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("Błąd: Polecenie 'node' nie zostało znalezione.", file=sys.stderr)
        print("Upewnij się, że masz zainstalowane Node.js i 'node' jest dostępne w PATH.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas wywoływania CLI TRON: {e}", file=sys.stderr)
        return False

# --- Główna logika ---

def process_file(json_file_name: str, base_dir: str) -> None:
    """
    Przetwarza pojedynczy plik JSON, tworząc pliki YAML, -nows.json, .toon i .tron.

    :param json_file_name: Nazwa pliku JSON (np. 'arch.json').
    :param base_dir: Katalog bazowy.
    """
    print(f"\n--- Processing '{json_file_name}' ---")
    
    input_path = os.path.join(base_dir, json_file_name)
    
    # 1. Wczytanie danych (potrzebne do YAML i -nows.json)
    data = load_json_data(input_path)
    if data is None:
        print(f"SKIP: Could not load data from '{input_path}'.")
        return

    # 2. Tworzenie ścieżek wyjściowych
    name_without_ext = json_file_name.replace('.json', '')
    
    output_yaml_path = os.path.join(base_dir, f"{name_without_ext}.yaml")
    output_nows_path = os.path.join(base_dir, f"{name_without_ext}-nows.json")
    output_toon_path = os.path.join(base_dir, f"{name_without_ext}.toon")
    output_tron_path = os.path.join(base_dir, f"{name_without_ext}.tron")

    json_to_yaml(data, output_yaml_path, input_path)
    json_to_nows_json(data, output_nows_path)
    json_to_toon_cli(input_path, output_toon_path)
    json_to_tron_cli(input_path, output_tron_path)

if __name__ == "__main__":
    BASE_DIR = 'samples/'
    JSON_FILES = ['arch.json', 'models.json', 'placeholder.json', 'photos.json', 'recipe.json']
    
    if not os.path.isdir(BASE_DIR):
        print(f"OSTRZEŻENIE: Katalog bazowy '{BASE_DIR}' nie istnieje. Próbuję go utworzyć.")
        try:
            os.makedirs(BASE_DIR)
        except OSError as e:
            print(f"Błąd: Nie udało się utworzyć katalogu '{BASE_DIR}': {e}", file=sys.stderr)
            sys.exit(1)

    for json_file in JSON_FILES:
        process_file(json_file, BASE_DIR)
    
    print("\n--- Done 💪💪💪 ---")