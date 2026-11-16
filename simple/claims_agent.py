"""Eén-bestand versie van de claims-agent, met LangChain en OOP. Leest alleen uit data/."""

import json
import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv(override=True)  # zoekt .env omhoog in de mappen en zet de API-key klaar
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
        self.policies = {p["policy_id"]: p for p in self.load("policies.json")}
        self.decisions = {}  # hier schrijft de tool de beslissingen weg: claim_id -> dict
        wording = (DATA / "policy_wording.md").read_text(encoding="utf-8")

        # Tools als methodes van de klasse, zodat ze bij self.policies/decisions kunnen.
        @tool
        def get_policy(policy_id: str) -> str:
            """Look up a customer's policy record: plan, add-ons, purchase date, trip, travelers."""
            policy = self.policies.get(policy_id)
            return json.dumps(policy, indent=2) if policy else f"No policy {policy_id}"

        @tool
        def record_decision(
            claim_id: str,
            decision: Literal["approve", "decline", "refer"],
            payout: float,
            rationale: str,
        ) -> str:
            """Record the decision on a claim. payout in USD (0 for decline/refer); rationale 2-3 sentences citing the policy section."""
            self.decisions[claim_id] = dict(
                decision=decision, payout=payout, rationale=rationale
            )
            return f"Recorded {decision} on {claim_id}"

        # ChatOpenAI wijst naar OpenRouter; create_agent bouwt de tool-lus voor ons.
        llm = ChatOpenAI(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            reasoning_effort=effort,
            max_retries=5,
        )
        self.agent = create_agent(
            llm, [get_policy, record_decision], system_prompt=f"{INSTRUCTIONS}\n{wording}"
        )

    @staticmethod
    def load(name):
        """Lees één JSON-bestand uit data/."""
        return json.loads((DATA / name).read_text(encoding="utf-8"))

    def decide(self, claim):
        """Laat de agent één claim afhandelen en geef zijn opgeslagen beslissing terug."""
        self.agent.invoke(
            {"messages": [{"role": "user", "content": json.dumps(claim, indent=2)}]}
        )
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
            right += ok
            print(claim["claim_id"], answer["decision"], label["decision"], "OK" if ok else "FOUT")
        print(f"{right} van {len(claims)} juist")


if __name__ == "__main__":
    ClaimsAgent().evaluate(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
