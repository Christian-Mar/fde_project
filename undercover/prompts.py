"""Prompts: de tekst die we voor elke claim naar het model sturen.

Een aanroep van de Chat Completions API krijgt een lijst "messages" mee. Elk bericht
heeft een role en content:

- system: hoe het model zich moet gedragen. Hier: de rol van claimbehandelaar, de
          drie mogelijke beslissingen en de volledige polisvoorwaarden.
          Dit is voor elke claim hetzelfde.
- user:   de vraag zelf. Hier: de polisgegevens van de klant en de claim, als JSON.
          Dit verschilt per claim.

De prompts staan in het Engels, net als de data en de polisvoorwaarden.
"""

import json  # de polis en de claim als JSON-tekst in de prompt zetten

from undercover.data import get_policies, get_policy_wording

# We laden de polisvoorwaarden en de polissen één keer, als deze module geïmporteerd
# wordt, en niet opnieuw bij elke claim.
POLICY_WORDING = get_policy_wording()
POLICIES = get_policies()

# De opdracht. "Section 10" verwijst naar het deel van de polisvoorwaarden dat
# uitlegt hoe je het uit te betalen bedrag berekent.
INSTRUCTIONS = """You are a claims handler at UnderCover Travel Insurance.
Decide the claim you are given, using the policy wording below and the customer's policy record.

Choose one decision:
- approve: the wording covers the claim and the required documents are attached. Work out the payout using section 10.
- decline: the wording does not cover the claim.
- refer: a colleague needs to look at it, because a required document is missing, or the story and the documents disagree, or the wording does not settle the question.
"""

# De system prompt = de opdracht + de volledige polisvoorwaarden (ca. 1.400 woorden).
# Zo heeft het model alle regels die het nodig heeft om te beslissen.
SYSTEM_PROMPT = f"""{INSTRUCTIONS}
{POLICY_WORDING}
"""


def make_user_prompt(claim):
    """De polisgegevens van de klant en de claim, als JSON.

    De claim bevat een policy_id; daarmee zoeken we de bijbehorende polis op.
    Met de koppen "# Policy record" en "# Claim" ziet het model waar elk deel begint.
    """
    policy = POLICIES[claim["policy_id"]]
    return f"""# Policy record

{json.dumps(policy, indent=2)}

# Claim

{json.dumps(claim, indent=2)}
"""


def make_messages(claim):
    """Bouw de messages voor één claim: eerst de system prompt, dan de user prompt.

    Dit is precies de lijst die we straks aan de API meegeven.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_user_prompt(claim)},
    ]


# --- Prompts voor de agent (deel 4) ---
#
# De agent krijgt dezelfde opdracht en polisvoorwaarden, met één verschil: hij krijgt
# de polis van de klant NIET in de prompt. Die moet hij zelf opzoeken met de tool
# get_policy, en zijn beslissing slaat hij op met de tool record_decision.

AGENT_SYSTEM_PROMPT = f"""{INSTRUCTIONS}
Use your tools to look up the customer's policy record and to record your decision.

{POLICY_WORDING}
"""


def make_claim_prompt(claim):
    """Alleen de claim, als JSON. De agent zoekt de polis zelf op via policy_id."""
    return f"""# Claim

{json.dumps(claim, indent=2)}
"""
