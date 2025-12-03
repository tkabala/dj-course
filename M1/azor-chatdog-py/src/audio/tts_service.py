"""TTS Service for managing text-to-speech model lifecycle."""

import warnings
from cli import console


class TTSService:
    """
    Singleton service for managing TTS model lifecycle.
    Handles model initialization, GPU detection, and provides thread-safe access.
    """

    _instance = None
    _tts_model = None
    _device = None
    _initialized = False

    def __new__(cls):
        """Ensures only one instance exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(TTSService, cls).__new__(cls)
        return cls._instance

    def _detect_device(self) -> str:
        """
        Detects CUDA availability and returns appropriate device string.

        Returns:
            str: 'cuda' if GPU is available, 'cpu' otherwise
        """
        try:
            import torch
            if torch.cuda.is_available():
                console.print_info("🚀 GPU (CUDA) wykryty - używam akceleracji GPU dla TTS")
                return "cuda"
            else:
                console.print_info("⚠️  CUDA niedostępne - używam CPU dla TTS (będzie wolniej)")
                return "cpu"
        except ImportError:
            console.print_info("⚠️  PyTorch nie znaleziony - używam CPU dla TTS")
            return "cpu"

    def _initialize_model(self):
        """
        Lazy initialization of TTS model with progress feedback.
        Loads the XTTS v2 model and configures it for the detected device.
        """
        if self._initialized:
            return

        console.print_info("\n🔄 Inicjalizacja modelu TTS...")

        # Detect device first
        self._device = self._detect_device()

        try:
            # Suppress torchaudio deprecation warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")
            warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.ffmpeg")

            from TTS.api import TTS

            console.print_info("📥 Ładowanie modelu XTTS v2 (to może potrwać 10-30 sekund przy pierwszym uruchomieniu)...")

            # Initialize TTS model
            self._tts_model = TTS(
                "tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=True
            ).to(self._device)

            self._initialized = True
            console.print_info("✅ Model TTS gotowy do użycia!")

        except ImportError as e:
            console.print_error(f"❌ Błąd: Biblioteka TTS nie jest zainstalowana.")
            console.print_info("   Zainstaluj wymagane biblioteki: pip install -r requirements.txt")
            raise
        except Exception as e:
            console.print_error(f"❌ Błąd podczas inicjalizacji modelu TTS: {e}")
            raise

    def get_model(self):
        """
        Returns TTS model, initializing it if necessary (lazy loading).

        Returns:
            TTS: Initialized TTS model instance
        """
        if not self._initialized:
            self._initialize_model()
        return self._tts_model

    def get_device(self) -> str:
        """
        Returns the device being used for TTS generation.

        Returns:
            str: 'cuda' or 'cpu'
        """
        if self._device is None:
            self._detect_device()
        return self._device
