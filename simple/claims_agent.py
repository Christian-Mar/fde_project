"""Eén-bestand versie van de claims-agent, met LangChain en OOP. Leest alleen uit data/."""

# Uitleg bij de imports:
# - json: JSON-bestanden lezen (loads) en de policy als tekst aan het model geven (dumps)
# - os: alleen voor os.getenv(), de API-key uit de omgeving halen
# - sys: sys.argv, zodat je `python claims_agent.py 10` kunt draaien
# - Path: paden bouwen die op elk besturingssysteem werken (met de / operator)
# - Literal: type dat alleen vaste waarden toestaat (zie record_decision)
# - load_dotenv: leest .env in, zodat de key in os.environ staat
# - create_agent: bouwt de agent-lus (model aanroepen, tool uitvoeren, resultaat terugsturen, herhalen)
# - tool: decorator die van een Python-functie een tool voor het model maakt
# - ChatOpenAI: client voor het OpenAI-API-formaat (/v1/chat/completions), niet per se voor
#   OpenAI zelf. Via base_url kun je er elke OpenAI-compatibele aanbieder mee aanspreken.
import json
import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# override=True: een waarde in .env wint van een variabele die al in je shell staat.
load_dotenv(override=True)  # zoekt .env omhoog in de mappen en zet de API-key klaar
# parent.parent = een map omhoog vanaf dit bestand; resolve() maakt het pad absoluut, zodat het
# ook werkt als je het script vanuit een andere map start.
DATA = Path(__file__).resolve().parent.parent / "data"  # data/ ligt naast deze map

INSTRUCTIONS = """You are a claims handler at UnderCover Travel Insurance.
Decide the claim using the policy wording and the customer's policy record.
Use your tools to look up the policy and to record your decision.
Decisions: approve (covered, documents attached, payout per section 10),
decline (not covered), refer (document missing, story/documents disagree, or unclear).
"""


class ClaimsAgent:
    def __init__(self, model="openai/gpt-6-luna", effort="low"):
        # De data één keer inlezen; alles komt uit de data-map, niets uit undercover/.
        # Policies als dict met policy_id als sleutel, zodat opzoeken snel gaat.
        self.policies = {p["policy_id"]: p for p in self.load("policies.json")}
        self.decisions = {}  # hier schrijft de tool de beslissingen weg: claim_id -> dict
        wording = (DATA / "policy_wording.md").read_text(encoding="utf-8")

        # Tools zijn geneste functies binnen __init__ (closure): ze "zien" self, zodat ze bij
        # self.policies/decisions kunnen, zonder dat het model self hoeft mee te geven.
        # Het model ziet alleen de parameters. LangChain gebruikt de docstring als beschrijving
        # van de tool en de type hints als schema.
        @tool
        def get_policy(policy_id: str) -> str:
            """Look up a customer's policy record: plan, add-ons, purchase date, trip, travelers."""
            policy = self.policies.get(policy_id)
            # Ook bij een fout geven we een string terug, zodat het model zelf kan reageren
            # in plaats van dat de lus crasht.
            return json.dumps(policy, indent=2) if policy else f"No policy {policy_id}"

        @tool
        def record_decision(
            claim_id: str,
            # Literal dwingt af dat het model alleen deze drie waarden kan kiezen.
            decision: Literal["approve", "decline", "refer"],
            payout: float,
            rationale: str,
        ) -> str:
            """Record the decision on a claim. payout in USD (0 for decline/refer); rationale 2-3 sentences citing the policy section."""
            # Dit is vooral een manier om gestructureerde output uit de agent te halen:
            # de beslissing komt in self.decisions terecht, niet in het gesprek met het model.
            self.decisions[claim_id] = {
                "decision": decision,
                "payout": payout,
                "rationale": rationale,
            }
            return f"Recorded {decision} on {claim_id}"

        # ChatOpenAI wijst naar OpenRouter; create_agent bouwt de tool-lus voor ons.
        # Ander model (DeepSeek, Mistral, ...)? Via OpenRouter pas je alleen model= aan.
        # Rechtstreeks bij de aanbieder: andere base_url + key (als die OpenAI-compatibel is),
        # of een eigen pakket zoals langchain_mistralai.ChatMistralAI.
        llm = ChatOpenAI(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            # Let op: reasoning_effort is OpenAI-achtig; niet elk model of elke aanbieder
            # ondersteunt het (kan genegeerd worden of een fout geven).
            reasoning_effort=effort,
            max_retries=5,
        )
        # System prompt = onze instructie plus de volledige polisvoorwaarden.
        self.agent = create_agent(
            llm, [get_policy, record_decision], system_prompt=f"{INSTRUCTIONS}\n{wording}"
        )

    @staticmethod  # gebruikt geen self, maar hoort wel bij de klasse
    def load(name):
        """Lees één JSON-bestand uit data/."""
        return json.loads((DATA / name).read_text(encoding="utf-8"))

    def decide(self, claim):
        """Laat de agent één claim afhandelen en geef zijn opgeslagen beslissing terug."""
        # De claim gaat als JSON-tekst in een gebruikersbericht; de agent draait zijn lus
        # (policy opzoeken, beslissing vastleggen).
        self.agent.invoke(
            {"messages": [{"role": "user", "content": json.dumps(claim, indent=2)}]}
        )
        # pop (geen get): er blijft niets achter voor de volgende claim. De fallback geldt als
        # het model de tool nooit aanriep.
        return self.decisions.pop(claim["claim_id"], {"decision": "nothing recorded"})

    def evaluate(self, limit=5):
        """Vergelijk de beslissingen met die van de behandelaars (juist = beslissing + bedrag ±$1)."""
        claims = self.load("claims.json")[:limit]
        labels = {d["claim_id"]: d for d in self.load("evals/decisions.json")}
        right = 0
        for claim in claims:
            answer = self.decide(claim)
            label = labels[claim["claim_id"]]
            ok = answer["decision"] == label["decision"] and abs(
                answer.get("payout", 0) - label["payout"]
            ) <= 1
            right += ok  # True telt als 1, False als 0
            print(claim["claim_id"], answer["decision"], label["decision"], "OK" if ok else "FOUT")
        print(f"{right} van {len(claims)} juist")


# Draait alleen als je het bestand direct start, niet bij een import.
if __name__ == "__main__":
    # Aantal claims van de command line, met 5 als standaard.
    ClaimsAgent().evaluate(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
