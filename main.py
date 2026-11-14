from undercover.config import OPENROUTER_API_KEY
from undercover.data import get_policies, get_policy_wording, get_sample_data
from undercover.explore import (
    DIFFICULTY_ORDER,
    claims_overview,
    count_by,
    print_json,
    show_claim,
)
from undercover.schemas import ClaimDecision

# TINY = True werkt met 5 claims in plaats van 25: goedkoper en sneller tijdens het testen.
TINY = False


def main():
    # Controle: is de key gevonden? (de key zelf tonen we niet)
    if OPENROUTER_API_KEY:
        print("OpenRouter API key found")
    else:
        print("No OpenRouter API key found. Add OPENROUTER_API_KEY to the .env file")

    # De data laden en kort samenvatten.
    claims, labels = get_sample_data(tiny=TINY)
    policies = get_policies()
    policy_wording = get_policy_wording()

    print(f"{len(claims)} claims, {len(labels)} labels, {len(policies)} policies")
    print(f"The policy wording is {len(policy_wording.split()):,} words")

    # --- De data verkennen ---

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

    # --- Het antwoordformaat (Structured Outputs) ---

    # 5. Het JSON-schema dat Pydantic maakt van ClaimDecision. Dit is wat het model
    #    straks meekrijgt als beschrijving van het antwoord dat we verwachten.
    print("\n--- JSON schema for ClaimDecision ---")
    print_json(ClaimDecision.model_json_schema())

    # Zo ziet een geldig antwoord eruit als Python-object (hier met de hand gemaakt).
    example = ClaimDecision(
        rationale="Example only: the airline canceled the flight and the delay is covered.",
        decision="approve",
        payout=420.0,
    )
    print("\n--- Example ClaimDecision ---")
    print(example)


if __name__ == "__main__":
    main()
