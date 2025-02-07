import os
import sys
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, SecretStr

from browser_use import Agent, Controller, BrowserConfig, Browser

load_dotenv()


class Album(BaseModel):
    title: str
    year: Optional[str] = None


class Disco(BaseModel):
    list_album: List[Album]


async def main():
    # Configuration du navigateur
    browser_config = BrowserConfig(
        headless=True,
        disable_security=True,
        extra_chromium_args=[
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--no-sandbox",
            "--user-agent=BrowserUse-DiscogsCrawler/1.0",
        ],
    )

    browser = Browser(config=browser_config)

    task = """Go to https://www.discogs.com/fr/ page and give me all albums from The Flower Kings band.
    Make sure to:
    1. Search for 'The Flower Kings'
    2. Go through at least the first 3 pages of results
    3. For each page, collect all albums information
    4. Use the pagination controls at the bottom of the page to navigate
    5. Between each page navigation, wait for 5 seconds to respect rate limits
    6. Stop when you've processed 3 pages or when there are no more pages"""

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", api_key=SecretStr(os.getenv("GEMINI_API_KEY"))
    )

    controller = Controller(output_model=Disco)

    agent = Agent(task=task, llm=model, controller=controller, browser=browser)

    try:
        history = await agent.run()
        result = history.final_result()

        if result:
            albums = Disco.model_validate_json(result)

            for album in albums.list_album:
                print("\n--------------------------------")
                print(f"Title:            {album.title}")
                print(f"Year:             {album.year}")

            print(f"\nTotal albums found: {len(albums.list_album)}")

            with open("flower_kings_albums.json", "w", encoding="utf-8") as f:
                f.write(albums.model_dump_json(indent=2))
        else:
            print("No results found")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
