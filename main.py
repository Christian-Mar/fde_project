import os

from dotenv import load_dotenv

load_dotenv(override=True)


def main():
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        print(f"OpenRouter-key gevonden: {key[:10]}...")
    else:
        print("Nog geen OPENROUTER_API_KEY in .env")


if __name__ == "__main__":
    main()
