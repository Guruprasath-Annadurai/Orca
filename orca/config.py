import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

ORCA_HOME = Path(os.environ.get("ORCA_HOME", Path.home() / ".orca"))
ORCA_HOME.mkdir(parents=True, exist_ok=True)

MEMORY_DIR = ORCA_HOME / "memory"
MEMORY_DIR.mkdir(exist_ok=True)

CACHE_DIR = ORCA_HOME / "cache"
CACHE_DIR.mkdir(exist_ok=True)

VAULT_DIR = ORCA_HOME / "vault"
VAULT_DIR.mkdir(exist_ok=True)


class OllamaConfig(BaseModel):
    host: str = os.environ.get("ORCA_OLLAMA_HOST", "http://localhost:11434")
    # Priority: your fine-tuned model → best available open model
    model_nano: str = os.environ.get("ORCA_NANO_MODEL", "orca-nano")
    model_core: str = os.environ.get("ORCA_CORE_MODEL", "orca")
    model_ultra: str = os.environ.get("ORCA_ULTRA_MODEL", "orca")
    fallback_models: list[str] = [
        "llama3.1:8b",
        "llama3:8b",
        "mistral:7b",
        "qwen2:7b",
        "gemma2:9b",
    ]


class BrainConfig(BaseModel):
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    context_length: int = 8192
    stream_by_default: bool = True


class OrcaConfig(BaseModel):
    ollama: OllamaConfig = OllamaConfig()
    brain: BrainConfig = BrainConfig()

    # Optional: only used for seeding training data, never for inference
    seed_api_key: str = os.environ.get("SEED_API_KEY", "")


CONFIG = OrcaConfig()
