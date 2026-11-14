"""Startpunt van het programma. Kies wat je wilt draaien met een commando:

    uv run main.py explore   de data, het schema en de prompts bekijken (gratis)
    uv run main.py llm       de claims laten beoordelen met één LLM-aanroep per claim
    uv run main.py agent     de claims laten beoordelen door een agent met tools

llm en agent roepen echte modellen aan en kosten dus (een beetje) geld.
"""

import sys  # sys.argv: de woorden die je na "main.py" typt

from undercover.agent import decide_with_agent, invoke_agent, make_agent
from undercover.config import OPENROUTER_API_KEY
from undercover.data import get_policies, get_policy_wording, get_sample_data
from undercover.evaluate import cost_summary, run, score_by_difficulty, show_misses
from undercover.explore import (
    DIFFICULTY_ORDER,
    claims_overview,
    count_by,
    print_json,
    show_claim,
)
from undercover.llm import decide
from undercover.prompts import make_messages
from undercover.schemas import ClaimDecision

# TINY = True werkt met 5 claims in plaats van 25: goedkoper en sneller tijdens het testen.
TINY = False

# Welk model de claims beoordeelt, en hoe lang het mag nadenken ("low", "medium", "high").
# Andere modellen vind je op https://openrouter.ai/models (naam in de vorm aanbieder/model).
MODEL = "openai/gpt-6-luna"
EFFORT = "low"

# De experimenten voor "llm": elk paar (model, effort) beoordeelt alle claims.
# Haal het # weg om een experiment toe te voegen; elk experiment kost geld.
EXPERIMENTS = [
    (MODEL, EFFORT),
    # ("openai/gpt-6-luna", "medium"),
    # ("google/gemini-3.5-flash-lite", "low"),
]

# Het model voor "agent". Standaard hetzelfde als MODEL, zodat je de agent eerlijk
# kunt vergelijken met de gewone LLM-aanroep. Opdracht uit de les: wat is het
# goedkoopste model waarmee de agent steeds minstens 15 van de 25 haalt?
AGENT_MODEL = MODEL
AGENT_EFFORT = EFFORT


def explore(claims, labels):
    """De data, het antwoordformaat en de prompts bekijken. Roept geen model aan."""
    policies = get_policies()
    policy_wording = get_policy_wording()
    print(f"{len(claims)} claims, {len(labels)} labels, {len(policies)} policies")
    print(f"The policy wording is {len(policy_wording.split()):,} words")

    # 1. Eén claim zoals de klant hem indient, en de beslissing van de behandelaar erover.
    #    Dat label gebruiken we later om de antwoorden van de modellen te controleren.
    print("\n--- First claim ---")
    print_json(claims[0])
    print("\n--- Handler's decision on that claim ---")
    print_json(labels[claims[0]["claim_id"]])

    # 2. Overzicht van alle claims in één tabel.
    data = claims_overview(claims, labels)
    print("\n--- All claims ---")
    print(data.to_string(index=False))

    # 3. Twee bekende voorbeelden in detail. We laden hier altijd de volledige wachtrij,
    #    want bij TINY = True zitten deze claims er misschien niet in.
    queue, queue_labels = get_sample_data()
    by_id = {claim["claim_id"]: claim for claim in queue}
    for claim_id in ["CLM-1015", "CLM-1010"]:
        show_claim(by_id[claim_id], queue_labels[claim_id])

    # 4. Aantallen per claimtype en per moeilijkheid, gesplitst naar beslissing.
    print("\n--- Claims by type ---")
    print(count_by(data, "claim_type"))
    print("\n--- Claims by difficulty ---")
    print(count_by(data, "difficulty").reindex(DIFFICULTY_ORDER + ["total"]))

    # 5. Het JSON-schema dat Pydantic maakt van ClaimDecision. Dit is wat het model
    #    meekrijgt als beschrijving van het antwoord dat we verwachten.
    print("\n--- JSON schema for ClaimDecision ---")
    print_json(ClaimDecision.model_json_schema())

    # 6. De messages voor de eerste claim: dit sturen we naar het model.
    #    De system prompt is lang (opdracht + polisvoorwaarden), dus we tonen alleen
    #    het begin. De user prompt tonen we volledig.
    messages = make_messages(claims[0])
    system, user = messages[0]["content"], messages[1]["content"]
    print(f"\n--- System prompt ({len(system.split()):,} words), first 600 chars ---")
    print(system[:600] + "...")
    print(f"\n--- User prompt ({len(user.split()):,} words) ---")
    print(user)


def show_results(all_results, labels, misses_of):
    """De drie resultaattabellen: per moeilijkheid, de fouten, en juist/kosten/tijd."""
    print("\n--- Right by difficulty ---")
    print(score_by_difficulty(all_results))

    print("\n--- Misses ---")
    show_misses(misses_of, labels)

    print("\n--- Right, cost and seconds ---")
    print(cost_summary(all_results))


def llm(claims, labels):
    """De claims laten beoordelen met één LLM-aanroep per claim (Structured Outputs)."""
    # 7. Eén claim laten beoordelen, en het antwoord naast dat van de behandelaar zetten.
    first = decide(claims[0], MODEL, EFFORT)
    label = labels[claims[0]["claim_id"]]
    print(f"\n--- {MODEL} at {EFFORT} on {claims[0]['claim_id']} ---")
    print_json(first)
    print(f"\nHandler: {label['decision']} ${label['payout']:,.2f}")

    # 8. Elk experiment uit EXPERIMENTS: alle claims door dat model op die effort.
    print("\n--- Evaluation ---")
    runs = [run(model, effort, claims, labels) for model, effort in EXPERIMENTS]

    # 9-11. De resultaten, met de fouten van het eerste experiment.
    all_results = [result for results in runs for result in results]
    show_results(all_results, labels, misses_of=runs[0])


def agent(claims, labels):
    """De claims laten beoordelen door een agent die tools gebruikt."""
    # 12. Eén claim, met alle berichten van de lus. Zo zie je hoe een agent werkt:
    #     het model vraagt de polis op (tool call), krijgt hem terug (tool message),
    #     slaat zijn beslissing op (tool call) en sluit af met een kort bericht.
    print(
        f"\n--- Agent with {AGENT_MODEL} at {AGENT_EFFORT} on {claims[0]['claim_id']} ---"
    )
    messages = invoke_agent(make_agent(AGENT_MODEL, AGENT_EFFORT), claims[0])
    for message in messages[1:]:  # [1:] slaat het eerste bericht (de claim) over
        message.pretty_print()

    # 13. De hele set: de agent naast de gewone LLM-aanroep met hetzelfde model.
    #     Zo zie je wat de agent-aanpak oplevert of kost. Per claim doet de agent
    #     drie modelaanroepen, dus reken op ongeveer drie keer de kosten.
    print("\n--- Evaluation ---")
    llm_results = run(AGENT_MODEL, AGENT_EFFORT, claims, labels)
    agent_results = run(
        AGENT_MODEL, AGENT_EFFORT, claims, labels, decide_fn=decide_with_agent
    )

    # 14. De resultaten, met de fouten van de agent.
    show_results(llm_results + agent_results, labels, misses_of=agent_results)


# De commando's die je kunt kiezen, met de functie die erbij hoort.
COMMANDS = {"explore": explore, "llm": llm, "agent": agent}


def main():
    # sys.argv is de lijst met woorden op de commandoregel: ["main.py", "agent"].
    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command not in COMMANDS:
        print(__doc__)  # de uitleg bovenaan dit bestand
        return

    # Controle: is de key gevonden? (de key zelf tonen we niet)
    if not OPENROUTER_API_KEY:
        print("No OpenRouter API key found. Add OPENROUTER_API_KEY to the .env file")
        return

    claims, labels = get_sample_data(tiny=TINY)
    COMMANDS[command](claims, labels)


if __name__ == "__main__":
    main()
