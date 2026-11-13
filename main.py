from undercover.config import OPENROUTER_API_KEY
from undercover.data import get_policies, get_policy_wording, get_sample_data

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


if __name__ == "__main__":
    main()
