"""
NatyaVeda — YouTube Data Collector
Uses yt-dlp to download Indian classical dance videos with quality filters.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Query Templates
# ─────────────────────────────────────────────────────────────────────────────

DANCE_QUERIES: dict[str, list[str]] = {
    "bharatanatyam": [
        "Bharatanatyam solo stage performance",
        "Bharatanatyam arangetram varnam full",
        "Bharatanatyam recital classical full",
    ],
    "kathak": [
        "Kathak solo performance classical stage",
        "Kathak thumri tarana recital full",
        "Kathak tatkar chakkar performance",
    ],
    "odissi": [
        "Odissi classical dance performance full",
        "Odissi Pallavi Mangalacharan recital",
        "Odissi solo tribhangi dance",
    ],
    "kuchipudi": [
        "Kuchipudi classical dance full performance",
        "Kuchipudi tarangam stage recital",
        "Kuchipudi solo dance performance",
    ],
    "manipuri": [
        "Manipuri Ras Lila classical dance",
        "Manipuri classical dance solo full",
        "Manipuri Sankirtana performance stage",
    ],
    "mohiniyattam": [
        "Mohiniyattam classical dance Kerala full",
        "Mohiniyattam solo recital performance",
        "Mohiniyattam stage performance complete",
    ],
    "sattriya": [
        "Sattriya dance Assam classical full",
        "Sattriya classical dance solo performance",
        "Sattriya recital stage performance",
    ],
    "kathakali": [
        "Kathakali full performance story complete",
        "Kathakali solo character performance full",
        "Kathakali Navarasas classical Kerala",
    ],
}

# Keywords indicating NON-dance content — used to filter search results
NEGATIVE_KEYWORDS = [
    "tutorial", "learn", "how to", "lesson", "class", "workshop",
    "beginners", "teaching", "demonstration", "basic steps",
    "reaction", "behind the scenes", "backstage", "interview",
    "compilation", "mashup", "remix", "fusion", "contemporary",
    "bollywood",
]

# Keywords that raise confidence this is a real performance
POSITIVE_KEYWORDS = [
    "performance", "recital", "stage", "arangetram", "concert",
    "festival", "full", "classical", "solo", "live",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_title(title: str) -> float:
    """Return 0-1 performance probability based on title keywords."""
    title_lower = title.lower()
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in title_lower)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in title_lower)
    score = min(1.0, pos / max(len(POSITIVE_KEYWORDS), 1))
    score -= neg * 0.15
    return max(0.0, score)


def _parse_duration(duration_str: str) -> int:
    """Parse ISO 8601 / colon-separated duration to total seconds."""
    if isinstance(duration_str, (int, float)):
        return int(duration_str)
    # HH:MM:SS or MM:SS
    parts = list(map(int, str(duration_str).split(":")))
    multipliers = [1, 60, 3600]
    return sum(p * m for p, m in zip(reversed(parts), multipliers))


# ─────────────────────────────────────────────────────────────────────────────
# Downloader
# ─────────────────────────────────────────────────────────────────────────────

class DanceVideoDownloader:
    """
    Downloads Indian classical dance videos from YouTube using yt-dlp.

    Usage
    -----
    >>> dl = DanceVideoDownloader(output_dir="data/raw", config_path="config/config.yaml")
    >>> dl.download_all(max_per_dance=50)
    """

    def __init__(
        self,
        output_dir: str | Path,
        config_path: str | Path = "config/config.yaml",
        dances: Optional[list[str]] = None,
        min_duration_sec: int = 60,
        max_duration_sec: int = 1200,
        min_views: int = 500,
        max_per_query: int = 25,
        preferred_format: str = "bestvideo[height>=720][ext=mp4]+bestaudio[ext=m4a]/best[height>=480]",
        cookies_file: Optional[str] = None,
        proxy: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.min_views = min_views
        self.max_per_query = max_per_query
        self.preferred_format = preferred_format
        self.cookies_file = cookies_file
        self.proxy = proxy
        self.dry_run = dry_run
        self.dances = dances or list(DANCE_QUERIES.keys())

        # Load config overrides
        if Path(config_path).exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            dc = cfg.get("data_collection", {})
            yt = dc.get("youtube", {})
            filt = dc.get("filters", {})
            self.min_duration_sec = filt.get("min_duration_sec", self.min_duration_sec)
            self.max_duration_sec = filt.get("max_duration_sec", self.max_duration_sec)
            self.min_views = filt.get("min_views", self.min_views)
            self.max_per_query = filt.get("max_videos_per_query", self.max_per_query)
            if yt.get("preferred_format"):
                self.preferred_format = yt["preferred_format"]
            self.cookies_file = yt.get("cookies_file") or self.cookies_file
            self.proxy = yt.get("proxy") or self.proxy

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────

    def download_all(self, max_per_dance: int = 100) -> dict[str, list[Path]]:
        """Download videos for all configured dance forms."""
        results: dict[str, list[Path]] = {}
        for dance in self.dances:
            logger.info(f"[{dance.upper()}] Starting download …")
            paths = self.download_dance(dance, max_videos=max_per_dance)
            results[dance] = paths
            logger.info(f"[{dance.upper()}] Downloaded {len(paths)} videos.")
        return results

    def download_dance(self, dance_form: str, max_videos: int = 100) -> list[Path]:
        """Download videos for a single dance form."""
        queries = DANCE_QUERIES.get(dance_form, [])
        dance_dir = self.output_dir / dance_form
        dance_dir.mkdir(parents=True, exist_ok=True)

        downloaded: list[Path] = []
        seen_ids: set[str] = set()

        for query in queries:
            if len(downloaded) >= max_videos:
                break

            logger.info(f"  Searching: '{query}'")
            candidates = self._search_youtube(query, limit=self.max_per_query)

            for meta in candidates:
                if len(downloaded) >= max_videos:
                    break
                vid_id = meta.get("id", "")
                if vid_id in seen_ids:
                    continue
                seen_ids.add(vid_id)

                if not self._passes_filters(meta, dance_form):
                    continue

                path = self._download_video(meta["id"], dance_dir, dance_form)
                if path:
                    downloaded.append(path)

            time.sleep(2)  # polite delay between queries

        return downloaded

    # ──────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────

    def _build_ytdlp_base_cmd(self) -> list[str]:
        cmd = ["yt-dlp", "--no-warnings", "--quiet"]
        if self.cookies_file:
            cmd += ["--cookies", self.cookies_file]
        if self.proxy:
            cmd += ["--proxy", self.proxy]
        return cmd

    def _search_youtube(self, query: str, limit: int = 25) -> list[dict]:
        """Use yt-dlp to search YouTube and return metadata list."""
        search_url = f"ytsearch{limit}:{query}"
        cmd = self._build_ytdlp_base_cmd() + [
            "--dump-json",
            "--flat-playlist",
            "--no-download",
            search_url,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            metas = []
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    metas.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return metas
        except subprocess.TimeoutExpired:
            logger.warning(f"Search timeout for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def _passes_filters(self, meta: dict, dance_form: str) -> bool:
        """Return True if video metadata passes quality/relevance filters."""
        title = meta.get("title", "")
        duration = meta.get("duration", 0) or 0
        view_count = meta.get("view_count", 0) or 0

        # Duration check
        dur_sec = _parse_duration(duration)
        if dur_sec < self.min_duration_sec or dur_sec > self.max_duration_sec:
            logger.debug(f"Skipping (duration {dur_sec}s): {title[:60]}")
            return False

        # View count check
        if view_count < self.min_views:
            logger.debug(f"Skipping (views {view_count}): {title[:60]}")
            return False

        # Title keyword negative filter
        title_lower = title.lower()
        for neg_kw in NEGATIVE_KEYWORDS:
            if neg_kw in title_lower:
                logger.debug(f"Skipping (negative keyword '{neg_kw}'): {title[:60]}")
                return False

        # Dance form name should be in title (loose check)
        dance_display = dance_form.replace("_", " ").lower()
        title_score = _score_title(title)
        has_dance_name = (
            dance_display in title_lower
            or dance_form[:4] in title_lower  # partial e.g. "kath" matches kathak/kathakali
        )

        if not has_dance_name and title_score < 0.1:
            logger.debug(f"Skipping (dance name not in title): {title[:60]}")
            return False

        return True

    def _download_video(
        self, video_id: str, out_dir: Path, dance_form: str
    ) -> Optional[Path]:
        """Download a single video by ID. Returns path or None on failure."""
        out_template = str(out_dir / f"{dance_form}_%(id)s.%(ext)s")
        cmd = self._build_ytdlp_base_cmd() + [
            "-f", self.preferred_format,
            "-o", out_template,
            "--merge-output-format", "mp4",
            "--write-info-json",
            "--no-playlist",
            "--retries", "3",
            f"https://www.youtube.com/watch?v={video_id}",
        ]

        if self.dry_run:
            logger.info(f"  [DRY RUN] Would download: {video_id}")
            return None

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                # Find the downloaded file
                for f in out_dir.glob(f"{dance_form}_{video_id}.*"):
                    if f.suffix == ".mp4":
                        logger.info(f"  ✓ Downloaded: {f.name}")
                        return f
        except subprocess.TimeoutExpired:
            logger.warning(f"Download timeout: {video_id}")
        except Exception as e:
            logger.error(f"Download failed {video_id}: {e}")

        return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="NatyaVeda YouTube Downloader")
    parser.add_argument("--dances", nargs="+", default=list(DANCE_QUERIES.keys()),
                        help="Dance forms to download")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--max-per-dance", type=int, default=100)
    parser.add_argument("--min-duration", type=int, default=60)
    parser.add_argument("--max-duration", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    dl = DanceVideoDownloader(
        output_dir=args.output,
        config_path=args.config,
        dances=args.dances,
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration,
        dry_run=args.dry_run,
    )
    results = dl.download_all(max_per_dance=args.max_per_dance)
    total = sum(len(v) for v in results.values())
    print(f"\n✅ Total downloaded: {total} videos")
    for dance, paths in results.items():
        print(f"   {dance:20s}: {len(paths)} videos")


if __name__ == "__main__":
    main()
