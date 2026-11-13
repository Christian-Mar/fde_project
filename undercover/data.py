"""Data: laadt de claims, de beslissingen van de behandelaars, de polissen en de polisvoorwaarden.

Inhoud van de map data/:
- claims.json             de 25 claims in de wachtrij (wat de klant indient)
- evals/decisions.json    de beslissing van de behandelaar per claim: het "juiste antwoord"
                          waarmee we de modellen straks beoordelen
- evals/held_out_*.json   25 andere claims + beslissingen, om later eerlijk te testen
                          op claims waarop je je prompt niet hebt afgestemd
- policies.json           de polisgegevens per klant (plan, add-ons, reis, reizigers)
- policy_wording.md       de polisvoorwaarden: de tekst die zegt wat gedekt is en wat niet
"""

import json  # JSON-tekst omzetten naar Python-lijsten en -dicts
from pathlib import Path  # bestandspaden opbouwen, los van het besturingssysteem

# De map data/ ligt naast de package-map undercover/.
# We rekenen vanaf de locatie van dit bestand (__file__) en niet vanaf de map waarin
# je het programma start. Zo werkt het pad overal: in main.py, in tests en op een server.
DATA = Path(__file__).resolve().parent.parent / "data"


def load(name):
    """Lees één JSON-bestand uit de map data/, bv. load("claims.json")."""
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def get_sample_data(tiny=False, held_out=False):
    """Geef de claims als lijst, en de beslissingen van de behandelaars als dict per claim_id.

    held_out=True: gebruik de 25 achtergehouden claims in plaats van de wachtrij.
    tiny=True: houd alleen de 5 claims over die als "tiny" gemarkeerd zijn
               (handig om goedkoop en snel te testen).
    """
    claims = load("evals/held_out_claims.json" if held_out else "claims.json")
    decisions = load("evals/held_out_decisions.json" if held_out else "evals/decisions.json")

    if tiny:
        # Een set (de {...}) met de claim_ids die tiny zijn; daarmee filteren we beide lijsten.
        chosen = {decision["claim_id"] for decision in decisions if decision["tiny"]}
        claims = [claim for claim in claims if claim["claim_id"] in chosen]
        decisions = [decision for decision in decisions if decision["tiny"]]

    # Van een lijst naar een dict met claim_id als sleutel,
    # zodat we later snel labels["CLM-1015"] kunnen opzoeken.
    labels = {decision["claim_id"]: decision for decision in decisions}
    return claims, labels


def get_policies():
    """Geef alle polissen als dict per policy_id, zodat je de polis van een claim snel vindt."""
    return {policy["policy_id"]: policy for policy in load("policies.json")}


def get_policy_wording():
    """Geef de polisvoorwaarden als één stuk tekst (Markdown). Die gaat straks mee in de prompt."""
    return (DATA / "policy_wording.md").read_text(encoding="utf-8")
