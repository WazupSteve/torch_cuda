"""
Feature Engineering Module

Processes raw forum data and extracts features for analysis.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FeatureEngineer:
    CUDA_KEYWORDS = ["cuda", "gpu", "device", "cudnn", "nvidia", "RuntimeError: CUDA", "out of memory", "cublas", "cusparse", "nccl"]
    ERROR_KEYWORDS = ["Traceback", "Error:", "Exception", "error:", "exception:", "failed", "RuntimeError"]

    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self) -> List[Dict]:
        all_topics = []
        batch_files = sorted(self.input_dir.glob("topics_batch_*.json"))

        if not batch_files:
            logger.warning(f"No batch files found in {self.input_dir}")
            return []

        for batch_file in tqdm(batch_files, desc="Loading batches"):
            try:
                with open(batch_file, "r", encoding="utf-8") as f:
                    all_topics.extend(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading {batch_file}: {e}")

        return all_topics

    def extract_text_from_html(self, html: str) -> str:
        return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True) if html else ""

    def has_code_block(self, html: str) -> bool:
        return bool(BeautifulSoup(html, "html.parser").find_all(["pre", "code"])) if html else False

    def count_code_blocks(self, html: str) -> int:
        return len(BeautifulSoup(html, "html.parser").find_all("pre")) if html else 0

    def is_cuda_related(self, text: str) -> bool:
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.CUDA_KEYWORDS)

    def has_error_trace(self, text: str) -> bool:
        return any(keyword in text for keyword in self.ERROR_KEYWORDS)

    def extract_features(self, topic: Dict) -> Dict:
        posts = topic.get("post_stream", {}).get("posts", [])
        if not posts:
            return None

        first_post = posts[0]
        cooked = first_post.get("cooked", "")
        text = self.extract_text_from_html(cooked)
        title = topic.get("title", "")
        full_text = f"{title} {text}"

        created = datetime.fromisoformat(topic["created_at"].replace("Z", "+00:00"))
        features = {
            "topic_id": topic.get("id"),
            "title": title,
            "category_id": topic.get("category_id"),
            "created_at": topic.get("created_at"),
            "views": topic.get("views", 0),
            "reply_count": topic.get("posts_count", 1) - 1,
            "like_count": topic.get("like_count", 0),
            "is_cuda_related": self.is_cuda_related(full_text),
            "has_code_block": self.has_code_block(cooked),
            "code_block_count": self.count_code_blocks(cooked),
            "question_length": len(text),
            "has_error_trace": self.has_error_trace(text),
            "has_accepted_answer": topic.get("has_accepted_answer", False),
            "is_resolved": bool(topic.get("accepted_answer", {}).get("post_number")),
            "hour_of_day": created.hour,
            "day_of_week": created.strftime("%A"),
        }

        # Time to resolution
        if features["is_resolved"] and "accepted_answer" in topic:
            for post in posts:
                if post.get("post_number") == topic["accepted_answer"].get("post_number"):
                    resolved_at = datetime.fromisoformat(post["created_at"].replace("Z", "+00:00"))
                    features["time_to_resolution_hours"] = (resolved_at - created).total_seconds() / 3600
                    break
            else:
                features["time_to_resolution_hours"] = None
        else:
            features["time_to_resolution_hours"] = None

        # Time to first response
        if len(posts) > 1:
            replied_at = datetime.fromisoformat(posts[1]["created_at"].replace("Z", "+00:00"))
            features["time_to_first_response_hours"] = (replied_at - created).total_seconds() / 3600
        else:
            features["time_to_first_response_hours"] = None

        return features

    def process_all_topics(self, topics: List[Dict]) -> pd.DataFrame:
        features_list = []
        for topic in tqdm(topics, desc="Topics"):
            features = self.extract_features(topic)
            if features:
                features_list.append(features)
        return pd.DataFrame(features_list)

    def save_processed_data(self, df: pd.DataFrame, filename: str = "forum_data.csv") -> None:
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)


def main():
    engineer = FeatureEngineer()
    topics = engineer.load_raw_data()
    df = engineer.process_all_topics(topics)
    engineer.save_processed_data(df)


if __name__ == "__main__":
    main()
