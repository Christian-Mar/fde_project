"""Agent: een LLM met tools, in een lus, om een doel te bereiken.

Tot nu toe deden we één LLM-aanroep per claim: alles zat in de prompt en het model
antwoordde met Structured Outputs. Een agent werkt anders:

1. Het model krijgt alleen de claim, plus een lijst tools die het mag gebruiken.
2. Het model beslist zelf welke tool het aanroept, bv. get_policy("POL-2001").
3. Ons programma voert die tool uit en stuurt het resultaat terug naar het model.
4. Dat herhaalt zich tot het model klaar is en geen tool meer aanroept.

Per claim verwachten we drie modelaanroepen: polis opvragen, beslissing opslaan,
en een kort afsluitend bericht.

LangChain regelt die lus voor ons met create_agent. Achter de schermen bouwt die
een LangGraph-graaf; daarover meer in week 2.
"""

import json  # de polis als JSON-tekst teruggeven aan het model
import time  # meten hoe lang een claim duurt
from typing import Literal  # beperkt "decision" tot de drie geldige waarden

from langchain.agents import create_agent  # bouwt de agent (model + tools + lus)
from langchain_core.tools import tool  # decorator: maakt van een functie een tool
from langchain_openai import ChatOpenAI  # LangChain-model voor OpenAI-compatibele API's

from undercover.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from undercover.prompts import AGENT_SYSTEM_PROMPT, POLICIES, make_claim_prompt

# Hier slaat de tool record_decision de beslissingen op: claim_id -> beslissing.
# In een echt systeem zou dit een database zijn; hier volstaat een dict.
recorded = {}


# --- De tools ---
#
# @tool maakt van een gewone functie een tool. LangChain gebruikt de naam, de
# type hints en de docstring om het model te vertellen wat de tool doet en welke
# argumenten hij verwacht. De docstring is dus een instructie voor het model,
# en staat daarom in het Engels.


@tool
def get_policy(policy_id: str) -> str:
    """Look up a customer's policy record: the plan, the add-ons, the purchase date, the trip and the travelers."""
    policy = POLICIES.get(policy_id)
    # Een tool geeft tekst terug aan het model. Bestaat de polis niet, dan zeggen
    # we dat, in plaats van een fout te geven: zo kan het model daarop reageren.
    return json.dumps(policy, indent=2) if policy else f"There is no policy {policy_id}"


@tool
def record_decision(
    claim_id: str,
    decision: Literal["approve", "decline", "refer"],
    payout: float,
    rationale: str,
) -> str:
    """Record the decision on a claim in the claims system.

    payout is the amount to pay in US dollars, or 0 for a decline or a referral.
    rationale is 2-3 sentences to explain the decision along with the relevant section of the policy.
    """
    # Dezelfde drie velden als ClaimDecision in schemas.py, maar nu als argumenten
    # van een tool. De agent "antwoordt" dus door deze tool aan te roepen.
    recorded[claim_id] = {
        "decision": decision,
        "payout": payout,
        "rationale": rationale,
    }
    return f"Recorded {decision} on {claim_id}"


# --- De agent bouwen ---


def make_agent(model, effort):
    """Bouw een agent met het gegeven model en de gegeven reasoning effort.

    ChatOpenAI is LangChain's eigen client. We richten hem op OpenRouter, net als
    de OpenAI-client in config.py, met dezelfde key en base_url.
    """
    llm = ChatOpenAI(
        model=model,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        reasoning_effort=effort,
        timeout=120,
        max_retries=5,
    )
    return create_agent(
        model=llm,
        tools=[get_policy, record_decision],
        system_prompt=AGENT_SYSTEM_PROMPT,
    )


def invoke_agent(agent, claim):
    """Laat de agent één claim afhandelen en geef alle berichten van de lus terug.

    invoke() start de lus met één user-bericht (de claim) en loopt tot de agent klaar
    is. Het resultaat bevat alle berichten: user, AI, tool, AI, tool, AI, ...
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": make_claim_prompt(claim)}]}
    )
    return result["messages"]


def decide_with_agent(claim, model, effort):
    """De agent-versie van decide() in llm.py: zelfde argumenten, zelfde soort antwoord.

    Daardoor kan run() in evaluate.py de agent op precies dezelfde manier evalueren
    als de gewone LLM-aanroep.
    """
    start = time.time()
    messages = invoke_agent(make_agent(model, effort), claim)

    # De beslissing staat niet in het antwoord, maar in wat de tool heeft opgeslagen.
    # pop() haalt hem eruit. Heeft de agent niets opgeslagen, dan telt dat als fout.
    answer = recorded.pop(
        claim["claim_id"],
        {"decision": "nothing recorded", "payout": 0.0, "rationale": ""},
    )

    # Elke AI-boodschap is één modelaanroep. OpenRouter zet de kosten ervan in de
    # metadata; samen zijn dat de kosten van deze claim.
    cost = sum(
        message.response_metadata["token_usage"]["cost"]
        for message in messages
        if message.type == "ai"
    )

    return {
        "model": model,
        "effort": effort,
        "technique": "Agent",
        "claim_id": claim["claim_id"],
        **answer,
        "cost": cost,
        "seconds": round(time.time() - start, 1),
    }
