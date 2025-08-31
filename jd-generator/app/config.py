"""Configuration settings for the JD Generator application."""

from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # API Keys
    groq_api_key: Optional[str] = Field(None, description="Groq API key")

    # Application Settings
    app_env: Literal["development", "staging", "production"] = Field(
        "development", description="Application environment"
    )
    debug: bool = Field(True, description="Debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", description="Logging level"
    )

    # Streamlit Configuration
    streamlit_server_port: int = Field(8501, description="Streamlit server port")
    streamlit_server_address: str = Field(
        "localhost", description="Streamlit server address"
    )

    # Groq Model Configuration
    llm_model: str = Field(
        "openai/gpt-oss-120b", description="Default Groq model to use"
    )
    llm_temperature: float = Field(0.7, description="LLM temperature setting")
    llm_max_tokens: int = Field(2000, description="Maximum tokens for LLM response")
    llm_timeout: int = Field(30, description="LLM request timeout in seconds")

    # Groq Voice Settings
    whisper_model: str = Field(
        "whisper-large-v3", description="Groq Whisper model for transcription"
    )
    audio_sample_rate: int = Field(16000, description="Audio sample rate")
    max_audio_duration: int = Field(60, description="Maximum audio duration in seconds")

    # Session Settings
    session_timeout_minutes: int = Field(30, description="Session timeout in minutes")
    max_conversation_length: int = Field(50, description="Maximum conversation turns")

    # File Paths
    @property
    def base_dir(self) -> Path:
        """Get the base directory of the application."""
        return Path(__file__).parent.parent

    @property
    def logs_dir(self) -> Path:
        """Get the logs directory."""
        return self.base_dir / "logs"

    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        return self.base_dir / "data"

    @property
    def templates_dir(self) -> Path:
        """Get the templates directory."""
        return self.base_dir / "templates"

    @field_validator("groq_api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate Groq API key format."""
        if v and v.startswith("your_") and v.endswith("_here"):
            return None
        return v

    def get_llm_api_key(self) -> str:
        """Get the Groq API key."""
        if not self.groq_api_key:
            raise ValueError("Groq API key not configured")
        return self.groq_api_key

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env == "development"

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env == "production"


# Create a singleton instance
settings = Settings()

# Export the settings instance
__all__ = ["settings", "Settings"]
