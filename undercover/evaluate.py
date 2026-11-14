"""Evalueren: laat een model alle claims beoordelen en vergelijk met de behandelaars.

Een "eval" is een vaste set vragen waarvan je het juiste antwoord al kent. Je laat
een model (of een agent) de hele set beantwoorden en telt hoeveel er goed zijn.
Omdat de set telkens hetzelfde is, kun je modellen, reasoning efforts en prompts
eerlijk met elkaar vergelijken: op juistheid, kosten en snelheid.
"""

import pandas as pd  # resultaten in tabellen zetten, groeperen en optellen

from undercover.llm import decide


def is_right(answer, label):
    """True als de beslissing klopt EN het bedrag maximaal een dollar afwijkt.

    Allebei moeten kloppen: "approve" met het verkeerde bedrag is in de praktijk
    ook fout (je betaalt te veel of te weinig uit).
    Kleine verschillen door afronding (bv. 419.99 tegenover 420.00) laten we toe.
    """
    return (
        answer["decision"] == label["decision"]
        and abs(answer["payout"] - label["payout"]) <= 1
    )


def run(model, effort, claims, labels):
    """Laat het model elke claim beoordelen en controleer elk antwoord met het label.

    Geeft een lijst met één dict per claim terug: het antwoord van decide(), plus de
    moeilijkheid van de claim en of het antwoord juist is ("right").
    """
    results = []
    for claim in claims:
        answer = decide(claim, model, effort)
        label = labels[claim["claim_id"]]

        # {**answer, ...} maakt een nieuwe dict: alle velden van answer, met twee
        # velden erbij. "difficulty" komt van het label, zodat we straks per
        # moeilijkheid kunnen tellen.
        results.append(
            {
                **answer,
                "difficulty": label["difficulty"],
                "right": is_right(answer, label),
            }
        )

        # Krijg je van OpenRouter de fout "new accounts are limited to 20 requests per
        # minute"? Zet dan hier time.sleep(3) (en import time bovenaan).

    # Totalen voor de hele set. sum() over True/False telt het aantal True.
    right = sum(result["right"] for result in results)
    cost = sum(result["cost"] for result in results)
    seconds = sum(result["seconds"] for result in results)
    print(
        f"{model} at {effort}: {right} of {len(results)} right "
        f"for ${cost:.4f} in {seconds:.0f} seconds"
    )
    return results


# --- De resultaten bekijken ---
#
# De functies hieronder krijgen de resultaten van één of meer runs, samengevoegd
# in één lijst, bv. luna_low + gemini. Elke rij heeft "model" en "effort", dus
# pandas kan per combinatie groeperen.


def score_by_difficulty(results):
    """Tel per model en effort hoeveel claims juist zijn, per moeilijkheid.

    pivot_table maakt een kruistabel: rijen = (model, effort), kolommen = difficulty,
    cellen = som van "right" (True telt als 1). Verwacht bij "manual" bijna niets:
    die claims hangen af van informatie die het model niet krijgt.
    """
    return pd.DataFrame(results).pivot_table(
        index=["model", "effort"],
        columns="difficulty",
        values="right",
        aggfunc="sum",
        sort=False,  # volgorde van de runs aanhouden
    )


def cost_summary(results):
    """Tel per model en effort het aantal juiste claims, de kosten en de tijd op.

    Dit zijn de drie cijfers die een klant wil weten: hoe goed, hoe duur, hoe snel.
    """
    return (
        pd.DataFrame(results)
        .groupby(["model", "effort"], sort=False)[["right", "cost", "seconds"]]
        .sum()
    )


def show_misses(results, labels):
    """Toon elke claim die het model fout had, met de uitleg van model en behandelaar.

    Zo zie je WAAROM het misging. Het veld "tests" in het label zegt wat de claim
    eigenlijk test, bv. een uitsluiting of een ontbrekend document.
    """
    for result in results:
        if not result["right"]:
            label = labels[result["claim_id"]]
            print(
                f"\n{result['claim_id']} ({label['difficulty']}): "
                f"model {result['decision']} ${result['payout']:,.2f}, "
                f"handler {label['decision']} ${label['payout']:,.2f}"
            )
            print(f"- Model:   {result['rationale']}")
            print(f"- Handler: {label['reason']}")
            print(f"- Tests:   {label['tests']}")
