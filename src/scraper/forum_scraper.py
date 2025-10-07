"""
PyTorch Forum Scraper

This module scrapes topics from the PyTorch discussion forum and saves them to disk.
Handles rate limiting, pagination, and robust error handling.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyTorchForumScraper:
    """Scraper for PyTorch discussion forum"""

    BASE_URL = "https://discuss.pytorch.org"
    RATE_LIMIT_DELAY = 0.1

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Educational Research Project)",
            "Accept": "application/json",
        })

    def _make_request(self, url: str, max_retries: int = 3) -> Optional[Dict]:
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                time.sleep(self.RATE_LIMIT_DELAY)
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch {url}: {e}")
                    return None
                time.sleep(2 ** attempt)
        return None

    def scrape_latest_topics(self, max_pages: Optional[int] = None) -> List[Dict]:
        topics, page = [], 0

        with tqdm(desc="Pages scraped") as pbar:
            while True:
                data = self._make_request(f"{self.BASE_URL}/latest.json?page={page}")
                if not data or "topic_list" not in data:
                    break

                topic_list = data["topic_list"].get("topics", [])
                if not topic_list:
                    break

                topics.extend(topic_list)
                pbar.update(1)
                page += 1

                if max_pages and page >= max_pages:
                    break

        return topics

    def scrape_topic_details(self, topic_id: int) -> Optional[Dict]:
        return self._make_request(f"{self.BASE_URL}/t/{topic_id}.json")

    def scrape_all_topic_details(self, topic_ids: List[int], batch_size: int = 100) -> List[Dict]:
        all_details, batch_count = [], 0

        for i, topic_id in enumerate(tqdm(topic_ids, desc="Topics")):
            details = self.scrape_topic_details(topic_id)
            if details:
                all_details.append(details)

            if (i + 1) % batch_size == 0:
                batch_count += 1
                self._save_batch(all_details, batch_count)
                all_details = []

        if all_details:
            batch_count += 1
            self._save_batch(all_details, batch_count)

        return all_details

    def _save_batch(self, topics: List[Dict], batch_num: int) -> None:
        output_path = self.output_dir / f"topics_batch_{batch_num}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(topics, f, indent=2, ensure_ascii=False)

    def scrape_categories(self) -> List[Dict]:
        data = self._make_request(f"{self.BASE_URL}/categories.json")
        return data["category_list"].get("categories", []) if data and "category_list" in data else []

    def save_metadata(self, topics: List[Dict], filename: str = "topics_metadata.json") -> None:
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(topics, f, indent=2, ensure_ascii=False)


def main():
    scraper = PyTorchForumScraper()

    categories = scraper.scrape_categories()
    scraper.save_metadata(categories, "categories.json")

    topics = scraper.scrape_latest_topics()
    scraper.save_metadata(topics, "topics_metadata.json")

    topic_ids = [topic["id"] for topic in topics]
    scraper.scrape_all_topic_details(topic_ids, batch_size=100)


if __name__ == "__main__":
    main()
