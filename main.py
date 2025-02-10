import os
import sys
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from browser_use import Agent, Controller, BrowserConfig, Browser

load_dotenv()


class Format(BaseModel):
    format: str
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    avg_price: Optional[float] = None


class Album(BaseModel):
    title: str
    year: Optional[str] = None
    formats: List[Format]


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

    task = """Go to https://www.discogs.com/fr/ page and search for album Fire and Ice from Yngwie Malmsteen, formats and prices.
    Make sure to:
    1. Search for the band Yngwie Malmsteen
    2. Search for the album Fire and Ice
    3. Click on the album link to access its details page
    4. For each format:
        a. Click on the 'Marketplace' tab or look for a section showing prices
        b. Wait for the price information to load completely (at least 3 seconds)
        c. Look for elements containing price information (usually prefixed with € or $)
        d. If prices are found:
            - Collect all visible prices
            - Calculate minimum price
            - Calculate maximum price 
            - Calculate average (mean) price
    5. Add a 3-second delay between each interaction to ensure content is loaded
    6. If no prices are visible, try looking for a 'Show Marketplace' or similar button and click it"""

    model = ChatOpenAI(model="gpt-4o", api_key=SecretStr(os.getenv("OPENAI_API_KEY")))

    controller = Controller(output_model=Album)

    agent = Agent(task=task, llm=model, controller=controller, browser=browser)

    try:
        history = await agent.run()
        result = history.final_result()

        if result:
            album = Album.model_validate_json(result)

            with open("flower_kings_albums.json", "w", encoding="utf-8") as f:
                f.write(album.model_dump_json(indent=2))
        else:
            print("No results found")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
