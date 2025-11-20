from pydantic import BaseModel, Field, validator
from typing import Optional, Literal

class GeminiConfig(BaseModel):
    engine: Literal["GEMINI"] = Field(default="GEMINI")
    model_name: str = Field(..., description="Nazwa modelu Gemini")
    gemini_api_key: str = Field(..., min_length=1, description="Klucz API Google Gemini")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Kontroluje losowość (0.0-2.0)")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling (0.0-1.0)")
    top_k: Optional[int] = Field(default=40, ge=1, description="Liczba najlepszych tokenów do próbkowania")

    @validator('gemini_api_key')
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("GEMINI_API_KEY nie może być pusty")
        return v.strip()
