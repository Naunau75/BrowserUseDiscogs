import os
import sys
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browser_use import Agent, Controller

load_dotenv()


class Album(BaseModel):
    title: str
    year: Optional[str] = None


class Disco(BaseModel):
    list_album: List[Album]


controller = Controller(output_model=Disco)

MAX_RETRIES = 3
INITIAL_DELAY = 2  # secondes


async def navigate_with_retry(
    page, url, max_retries=MAX_RETRIES, initial_delay=INITIAL_DELAY
):
    for attempt in range(max_retries):
        try:
            print(f"Attempting to navigate to {url}, attempt {attempt + 1}")
            await page.goto(url)
            return  # Navigation successful
        except Exception as e:
            print(f"Navigation failed on attempt {attempt + 1}: {e}")
            if "net::ERR_TOO_MANY_REDIRECTS" in str(e):  # handle redirect loops
                raise e
            if attempt == max_retries - 1:
                raise  # Re-raise exception if max retries reached

            delay = initial_delay * (2**attempt)  # Exponential backoff
            print(f"Waiting {delay} seconds before retrying...")
            await asyncio.sleep(delay)


async def main():
    task = """Go to https://www.discogs.com/fr/ page and give me all albums from The Flower Kings band.
    Make sure to:
    1. Search for 'The Flower Kings'
    2. Go through at least the first 3 pages of results
    3. For each page, collect all albums information
    4. Use the pagination controls at the bottom of the page to navigate
    5. Between each page navigation, wait for a random amount of time (between 3 and 7 seconds) to respect rate limits
    6. Stop when you've processed 3 pages or when there are no more pages"""

    model = ChatOpenAI(model="gpt-4o")
    agent = Agent(
        task=task,
        llm=model,
        controller=controller,
    )

    history = await agent.run(max_steps=200)

    result = history.final_result()
    if result:
        albums: Disco = Disco.model_validate_json(result)

        for album in albums.list_album:
            print("\n--------------------------------")
            print(f"Title:            {album.title}")
            print(f"Year:             {album.year}")

        print(f"\nTotal albums found: {len(albums.list_album)}")
    else:
        print("No result")

    # Sauvegarder
    with open("data.json", "w") as f:
        f.write(albums.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
