import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    api_keys = {
        "xai": os.getenv("XAI-API-KEY") or os.getenv("XAI_API_KEY") or "",
        "sarvam": os.getenv("SARVAM_API_KEY") or "",
        "google": os.getenv("GOOGLE_API_KEY") or "",
        "deepgram": os.getenv("DEEPGRAM_API_KEY") or "",
    }

    if not any(api_keys.values()):
        print("Error: no API key set in .env", file=sys.stderr)
        sys.exit(1)

    raw_data = Path("raw-data")
    companies = [d.name for d in raw_data.iterdir() if d.is_dir()]

    if not companies:
        print("No company folders found in raw-data/", file=sys.stderr)
        sys.exit(1)

    # For now pick first company; later TUI can have company selector
    company = companies[0]
    if len(companies) > 1:
        print("Multiple companies found, using:", company)

    from tui import AgentSchoolApp
    app = AgentSchoolApp(company=company, api_keys=api_keys)
    app.run()


if __name__ == "__main__":
    main()
