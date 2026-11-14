"""Schema's: de vorm van het antwoord dat we van het model terug willen.

Structured Outputs
------------------
Normaal antwoordt een LLM met vrije tekst. Daarmee kan een programma weinig: je
moet zelf uitzoeken waar de beslissing en het bedrag staan. Met Structured Outputs
geven we het model een schema mee, en antwoordt het met JSON die exact dat schema
volgt.

We beschrijven het schema als een Pydantic-class. De OpenAI-client zet die class om
naar een JSON-schema voor het model, en zet het JSON-antwoord weer om naar een
object van die class. Je krijgt dan gewoon answer.decision en answer.payout, met de
juiste types.
"""

from typing import Literal  # beperkt een veld tot een vaste lijst waarden

# BaseModel = basis voor het schema, Field = extra info per veld (zoals een description)
from pydantic import BaseModel, Field


class ClaimDecision(BaseModel):
    """A claims handler's decision on one claim."""

    # De volgorde van de velden is bewust: het model schrijft de JSON van boven naar
    # beneden. Door de rationale (de redenering) EERST te laten schrijven, denkt het
    # model na voordat het de beslissing neemt. Andersom zou het eerst kiezen en
    # daarna een uitleg verzinnen die bij die keuze past.
    #
    # De description gaat mee naar het model: het is een instructie per veld.
    # Daarom staat die in het Engels, net als de rest van de prompt.
    rationale: str = Field(
        description="2-3 sentences to explain the decision along with the relevant section of the policy"
    )

    # Literal: het model MOET één van deze drie woorden kiezen, iets anders is ongeldig.
    # - approve: gedekt en de documenten zijn compleet, dus uitbetalen
    # - decline: niet gedekt door de polisvoorwaarden
    # - refer:   doorsturen naar een collega (document ontbreekt, verhaal en documenten
    #            spreken elkaar tegen, of de voorwaarden geven geen duidelijk antwoord)
    decision: Literal["approve", "decline", "refer"]

    # float: een kommagetal. Bij decline of refer hoort dit 0 te zijn.
    payout: float = Field(
        description="The amount to pay in US dollars, or 0 for a decline or a referral"
    )
