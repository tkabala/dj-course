from pydantic import BaseModel, Field, validator
from typing import Optional, Literal

class OpenAIConfig(BaseModel):
    engine: Literal["OPENAI"] = Field(default="OPENAI")
    model_name: str = Field(..., description="Nazwa modelu OpenAI (np. gpt-4, gpt-3.5-turbo)")
    openai_api_key: str = Field(..., min_length=1, description="Klucz API OpenAI")
    openai_base_url: Optional[str] = Field(default=None, description="Opcjonalny URL bazowy dla kompatybilnych serwisów (np. Ollama)")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Kontroluje losowość (0.0-2.0)")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling (0.0-1.0)")

    @validator('openai_api_key')
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("OPENAI_API_KEY nie może być pusty")
        return v.strip()

    @validator('openai_base_url')
    def validate_base_url(cls, v):
        if v and v.strip() == "":
            return None
        return v.strip() if v else None
