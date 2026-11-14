"""Configuratie: leest de API-key uit .env en maakt de client om modellen aan te roepen."""

import os  # omgevingsvariabelen lezen, zoals OPENROUTER_API_KEY

from dotenv import load_dotenv  # laadt de variabelen uit het .env-bestand in os.environ
from openai import OpenAI  # OpenAI-client; werkt ook met OpenRouter

# Lees .env in. override=True: waarden uit .env winnen van wat al in je shell staat.
load_dotenv(override=True)

# OpenRouter is één toegangspunt tot modellen van veel aanbieders (OpenAI, Google, ...).
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Eén gedeelde client voor het hele programma.
# timeout=120: maximaal 2 minuten wachten op een antwoord.
# max_retries=5: bij tijdelijke fouten (bv. rate limits) automatisch opnieuw proberen.
openrouter = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    timeout=120,
    max_retries=5,
)
