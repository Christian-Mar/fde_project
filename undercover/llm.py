"""LLM: laat één model één claim beoordelen.

Hier komen de vorige blokken samen:
- config.py   levert de client (de verbinding met OpenRouter)
- prompts.py  levert de messages (system prompt + user prompt voor deze claim)
- schemas.py  levert ClaimDecision (de vorm van het antwoord)
"""

import time  # meten hoe lang een aanroep duurt

from undercover.config import openrouter
from undercover.prompts import make_messages
from undercover.schemas import ClaimDecision


def decide(claim, model, effort):
    """Laat één model één claim beoordelen en geef het antwoord terug als dict.

    claim:  één claim uit de data (een dict)
    model:  de OpenRouter-naam van het model, in de vorm "aanbieder/model",
            bv. "openai/gpt-6-luna" of "google/gemini-3.5-flash-lite"
    effort: de reasoning effort: hoe lang het model mag nadenken voor het antwoordt.
            "low" is snel en goedkoop, "medium" en "high" zijn grondiger maar duurder.
    """
    start = time.time()

    # .parse() in plaats van .create(): hetzelfde API-verzoek, maar de client geeft
    # ClaimDecision mee als JSON-schema (Structured Outputs) en zet het antwoord
    # meteen om naar een ClaimDecision-object.
    response = openrouter.chat.completions.parse(
        model=model,
        messages=make_messages(claim),
        response_format=ClaimDecision,
        reasoning_effort=effort,
        # OpenRouter stuurt een verzoek naar één van meerdere aanbieders (providers)
        # van hetzelfde model. require_parameters=True: kies alleen een provider die
        # ALLE parameters ondersteunt (Structured Outputs en reasoning_effort). Anders
        # kan een provider ze stilletjes negeren.
        extra_body={"provider": {"require_parameters": True}},
    )

    # Het antwoord staat in choices[0].message. Bij .parse() is .parsed al een
    # ClaimDecision, dus we kunnen meteen answer.decision en answer.payout gebruiken.
    answer = response.choices[0].message.parsed

    # We geven een gewone dict terug met het antwoord én gegevens over de aanroep.
    # Zo kunnen we later veel resultaten in één pandas-tabel zetten en vergelijken.
    return {
        "model": model,
        "effort": effort,
        "technique": "LLM",  # één LLM-aanroep; later vergelijken we met een agent
        "claim_id": claim["claim_id"],
        "decision": answer.decision,
        "payout": answer.payout,
        "rationale": answer.rationale,
        # OpenRouter zet de kosten van deze aanroep (in dollar) in usage.cost.
        # Dat veld is een extra van OpenRouter; de gewone OpenAI-API heeft het niet.
        "cost": response.usage.cost,
        "seconds": round(time.time() - start, 1),
    }
