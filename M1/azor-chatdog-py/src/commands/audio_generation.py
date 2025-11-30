"""Audio generation commands for Azor Chatdog."""

import os
import tempfile
import shutil
from typing import List, Dict
from cli import console
from files.config import LOG_DIR


def _extract_text_from_entry(entry: Dict) -> str:
    """
    Helper function to safely extract text from history entry.

    Args:
        entry: History entry dictionary

    Returns:
        str: Extracted text or empty string if not found
    """
    try:
        return entry['parts'][0].get('text', '')
    except (KeyError, IndexError):
        return ''


def _validate_voice_samples() -> tuple[str, str]:
    """
    Validates that voice sample files exist and returns their paths.

    Returns:
        tuple: (azor_sample_path, user_sample_path)

    Raises:
        FileNotFoundError: If voice sample files are missing
    """
    # Get base path for samples
    base_path = os.path.dirname(os.path.dirname(__file__))
    azor_sample = os.path.join(base_path, 'sample', 'sample-azor.mp3')
    user_sample = os.path.join(base_path, 'sample', 'sample-user.mp3')

    # Validate files exist
    if not os.path.exists(azor_sample):
        raise FileNotFoundError(f"Brak pliku próbki głosu Azora: {azor_sample}")

    if not os.path.exists(user_sample):
        raise FileNotFoundError(f"Brak pliku próbki głosu użytkownika: {user_sample}")

    return azor_sample, user_sample


def generate_last_response_audio(history: List[Dict], session_id: str, assistant_name: str):
    """
    Generates WAV file with the last assistant response.

    Args:
        history: Conversation history in universal format
        session_id: Unique session identifier (used for filename)
        assistant_name: Name of the assistant (for display purposes)
    """
    # Validation
    if not history:
        console.print_error("❌ Historia sesji jest pusta. Brak audio do wygenerowania.")
        return

    # Find last model response
    last_model_response = None
    for entry in reversed(history):
        if entry.get('role') == 'model':
            last_model_response = entry
            break

    if not last_model_response:
        console.print_error("❌ Brak odpowiedzi asystenta w historii.")
        return

    # Extract text
    text = _extract_text_from_entry(last_model_response)
    if not text or text.strip() == '':
        console.print_error("❌ Ostatnia odpowiedź nie zawiera tekstu.")
        return

    # Generate audio
    console.print_info(f"\n🎙️  Generowanie pliku audio dla ostatniej odpowiedzi {assistant_name}...")

    try:
        from audio.tts_service import TTSService

        # Validate voice samples
        azor_sample, _ = _validate_voice_samples()

        # Get TTS service (will initialize on first call)
        tts_service = TTSService()
        tts_model = tts_service.get_model()

        # Output file path
        output_filename = f"{session_id}-last-response.wav"
        output_path = os.path.join(LOG_DIR, output_filename)

        # Generate TTS
        console.print_info(f"📝 Przetwarzanie tekstu ({len(text)} znaków)...")
        tts_model.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=azor_sample,
            language="pl"
        )

        # Get file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        console.print_info(f"\n✅ Plik audio zapisany: {output_path}")
        console.print_info(f"   Rozmiar: {file_size_mb:.2f} MB")

    except FileNotFoundError as e:
        console.print_error(f"❌ {e}")
    except ImportError:
        console.print_error("❌ Błąd: Biblioteka TTS nie jest zainstalowana.")
        console.print_info("   Zainstaluj wymagane biblioteki: pip install -r requirements.txt")
    except Exception as e:
        console.print_error(f"❌ Błąd podczas generowania audio: {e}")


def generate_full_conversation_audio(history: List[Dict], session_id: str, assistant_name: str):
    """
    Generates WAV file with the entire conversation using alternating voices.

    Args:
        history: Conversation history in universal format
        session_id: Unique session identifier (used for filename)
        assistant_name: Name of the assistant (for display purposes)
    """
    if not history or len(history) < 2:
        console.print_error("❌ Historia sesji jest zbyt krótka. Potrzebne są co najmniej 2 wpisy.")
        return

    console.print_info(f"\n🎙️  Generowanie pliku audio dla całej konwersacji ({len(history)} wpisów)...")
    console.print_info("⏳ To może potrwać kilka minut dla długich konwersacji...")

    temp_dir = None

    try:
        from audio.tts_service import TTSService
        from pydub import AudioSegment

        # Validate voice samples
        azor_sample, user_sample = _validate_voice_samples()

        # Get TTS service (will initialize on first call)
        tts_service = TTSService()
        tts_model = tts_service.get_model()

        # Create temporary directory for segments
        temp_dir = tempfile.mkdtemp()
        audio_segments = []

        # Generate each message
        for i, entry in enumerate(history):
            role = entry.get('role', '')
            text = _extract_text_from_entry(entry)

            if not text or text.strip() == '':
                console.print_info(f"   ⏭️  Pomijam pusty wpis [{i+1}/{len(history)}]...")
                continue

            # Choose voice sample
            speaker_sample = azor_sample if role == 'model' else user_sample
            speaker_label = assistant_name if role == 'model' else 'Użytkownik'

            console.print_info(f"   🔊 Generowanie [{i+1}/{len(history)}] - {speaker_label} ({len(text)} znaków)...")

            # Generate segment
            segment_path = os.path.join(temp_dir, f"segment_{i}.wav")
            tts_model.tts_to_file(
                text=text,
                file_path=segment_path,
                speaker_wav=speaker_sample,
                language="pl"
            )

            # Load segment
            segment = AudioSegment.from_wav(segment_path)
            audio_segments.append(segment)

            # Add silence between messages (800ms)
            if i < len(history) - 1:  # Don't add silence after last message
                silence = AudioSegment.silent(duration=800)
                audio_segments.append(silence)

        if not audio_segments:
            console.print_error("❌ Brak segmentów audio do wygenerowania.")
            return

        # Concatenate all segments
        console.print_info("   🔗 Łączenie segmentów audio...")
        final_audio = sum(audio_segments)

        # Save final file
        output_filename = f"{session_id}-full-conversation.wav"
        output_path = os.path.join(LOG_DIR, output_filename)
        final_audio.export(output_path, format="wav")

        # Calculate metrics
        duration_seconds = len(final_audio) / 1000
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        console.print_info(f"\n✅ Plik audio zapisany: {output_path}")
        console.print_info(f"   Czas trwania: {duration_seconds:.1f} sekund")
        console.print_info(f"   Rozmiar: {file_size_mb:.2f} MB")

    except FileNotFoundError as e:
        console.print_error(f"❌ {e}")
    except ImportError as e:
        if 'pydub' in str(e):
            console.print_error("❌ Błąd: Biblioteka pydub nie jest zainstalowana.")
            console.print_info("   Zainstaluj wymagane biblioteki: pip install -r requirements.txt")
            console.print_info("   UWAGA: pydub wymaga również instalacji ffmpeg w systemie!")
        else:
            console.print_error("❌ Błąd: Brakujące biblioteki.")
            console.print_info("   Zainstaluj wymagane biblioteki: pip install -r requirements.txt")
    except PermissionError:
        console.print_error(f"❌ Brak uprawnień do zapisu pliku: {output_path}")
    except Exception as e:
        console.print_error(f"❌ Błąd podczas generowania audio: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup temporary files
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                console.print_error(f"⚠️  Nie udało się usunąć plików tymczasowych: {e}")
