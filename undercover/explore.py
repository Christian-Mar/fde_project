"""Verkennen: de data bekijken voordat we er modellen op loslaten.

Altijd eerst naar je data kijken: welke soorten claims zijn er, wat besliste de
behandelaar, en welke claims zijn moeilijk? Dan begrijp je later beter waarom een
model iets goed of fout doet.
"""

import json  # een claim leesbaar afdrukken (met inspringing)

import pandas as pd  # tabellen (DataFrames) maken, groeperen en tellen

# pandas kort lange tabellen standaard in met "...". Voor de terminal tonen we alles.
pd.set_option("display.width", 200)  # breedte van de uitvoer in tekens
pd.set_option("display.max_columns", None)  # alle kolommen tonen
pd.set_option("display.max_colwidth", 60)  # lange tekst (summary) afkappen op 60 tekens

# Volgorde van moeilijkheid, van makkelijk naar moeilijk:
# - routine: de polisvoorwaarden zijn genoeg om te beslissen
# - hard:    ook met de polisvoorwaarden, maar lastiger te lezen
# - manual:  hangt af van het claims manual of de geschiedenis van de klant,
#            informatie die het model nu nog NIET krijgt (verwacht hier fouten)
DIFFICULTY_ORDER = ["routine", "hard", "manual"]


def print_json(item):
    """Druk een dict (een claim of een beslissing) leesbaar af."""
    print(json.dumps(item, indent=2, ensure_ascii=False))


def claims_overview(claims, labels):
    """Maak één tabel met per claim de belangrijkste velden en de beslissing van de behandelaar.

    De claim zelf (wat de klant indient) en het label (wat de behandelaar besliste)
    zitten in aparte bestanden. Via claim_id voegen we ze samen tot één rij.
    """
    rows = []
    for claim in claims:
        label = labels[claim["claim_id"]]
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "summary": claim["summary"],
                "claim_type": claim["claim_type"],
                "amount_claimed": claim["amount_claimed"],  # wat de klant vraagt
                "difficulty": label["difficulty"],
                "decision": label["decision"],  # approve / decline / refer
                "payout": label["payout"],  # wat de behandelaar uitbetaalt
            }
        )
    return pd.DataFrame(rows)


def show_claim(claim, label):
    """Toon één claim als verhaal, met daaronder de beslissing en de reden van de behandelaar."""
    print(f"\n=== {claim['claim_id']}: {claim['summary']} ===\n")
    print(claim["story"])  # het verhaal van de klant, in eigen woorden
    print(f"\nHandler: {label['decision']} ${label['payout']:,.2f}. {label['reason']}")


def count_by(data, column):
    """Tel de claims per waarde van een kolom, gesplitst naar de beslissing van de behandelaar.

    Dit is de tabelversie van de histogrammen in de les: rijen = column,
    kolommen = decision, cellen = aantal claims. margins=True voegt totalen toe.
    """
    return pd.crosstab(
        data[column], data["decision"], margins=True, margins_name="total"
    )
