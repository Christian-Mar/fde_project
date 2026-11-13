from undercover.config import OPENROUTER_API_KEY


def main():
    # Controle: is de key gevonden? (de key zelf tonen we niet)
    if OPENROUTER_API_KEY:
        print("OpenRouter API key found")
    else:
        print("No OpenRouter API key found. Add OPENROUTER_API_KEY to the .env file")


if __name__ == "__main__":
    main()
