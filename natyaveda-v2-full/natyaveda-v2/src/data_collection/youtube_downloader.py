r"""
NatyaVeda — YouTube Data Collector (Updated for Shorts & Volume)
Uses yt-dlp to download Indian classical dance videos with quality filters.
Now supports Shorts and includes bot-detection bypass.
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
# Query Templates (Expanded for High Volume & Shorts)
# ─────────────────────────────────────────────────────────────────────────────

DANCE_QUERIES: dict[str, list[str]] = {
    "bharatanatyam": [
        "Bharatanatyam solo stage performance",
        "Bharatanatyam Margam Varnam Tillana full",
        "Bharatanatyam recital classical stage recital",
        "Bharatanatyam arangetram performance full",
        "Bharatanatyam Shorts dance reel",
        "Bharatanatyam Music Academy Madras recital",
    ],
    "kathak": [
        "Kathak solo performance classical stage",
        "Kathak tatkar chakkar recital full",
        "Kathak solo Lucknow Jaipur gharana stage",
        "Kathak thumri tarana performance full",
        "Kathak dance Shorts viral reel",
        "Kathak Raigarh Festival performance",
    ],
    "odissi": [
        "Odissi classical dance performance full",
        "Odissi Pallavi Mangalacharan recital stage",
        "Odissi solo tribhangi dance performance",
        "Odissi Konark Festival dance recital",
        "Odissi dance Shorts stage reel",
        "Odissi Mukteswar Festival performance",
    ],
    "kuchipudi": [
        "Kuchipudi classical dance full performance",
        "Kuchipudi tarangam plate dance stage",
        "Kuchipudi solo dance recital stage",
        "Kuchipudi Siddhendra Yogi Mahotsav full",
        "Kuchipudi dance Shorts solo reel",
    ],
    "manipuri": [
        "Manipuri Ras Lila classical dance stage",
        "Manipuri Pung Cholom performance solo",
        "Manipuri Sankirtana performance stage full",
        "Manipuri solo classical dance recital",
        "Manipuri dance Shorts stage reel",
    ],
    "mohiniyattam": [
        "Mohiniyattam classical dance Kerala stage",
        "Mohiniyattam solo recital performance full",
        "Mohiniyattam Soorya Festival classical dance",
        "Mohiniyattam Kerala Kalamandalam recital",
        "Mohiniyattam dance Shorts Kerala reel",
    ],
    "sattriya": [
        "Sattriya dance Assam classical solo",
        "Sattriya Gayan Bayan performance stage",
        "Sattriya recital Sangeet Natak Akademi",
        "Sattriya classical dance solo performance",
        "Sattriya dance Shorts Assam reel",
    ],
    "kathakali": [
        "Kathakali full performance story play",
        "Kathakali Navarasas classical solo Kerala",
        "Kathakali Vesham stage performance full",
        "Kathakali dance Shorts Kerala reel",
        "Kathakali Kalamandalam story complete",
    ],
}

# Adjusted to allow Shorts/Reels while filtering tutorials
NEGATIVE_KEYWORDS = [
    "tutorial", "learn", "how to", "lesson", "class", "workshop",
    "beginners", "teaching", "demonstration", "basic steps",
    "behind the scenes", "backstage", "interview",
    "compilation", "mashup", "remix", "fusion", "contemporary",
    "bollywood", "vlog",
]

# Added "shorts" and "reel" to boost performance detection
POSITIVE_KEYWORDS = [
    "performance", "recital", "stage", "arangetram", "concert",
    "festival", "full", "classical", "solo", "live", "shorts", 
    "reel", "varnam", "tatkar", "pallavi", "doordarshan",
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
    parts = list(map(int, str(duration_str).split(":")))
    multipliers = [1, 60, 3600]
    return sum(p * m for p, m in zip(reversed(parts), multipliers))


# ─────────────────────────────────────────────────────────────────────────────
# Downloader
# ─────────────────────────────────────────────────────────────────────────────

class DanceVideoDownloader:
    """
    Downloads Indian classical dance videos from YouTube using yt-dlp.
    """

    def __init__(
        self,
        output_dir: str | Path,
        config_path: str | Path = "config/config.yaml",
        dances: Optional[list[str]] = None,
        min_duration_sec: int = 15,   # Lowered to 15s to include Shorts
        max_duration_sec: int = 1800, # Raised to 30m for full Margams
        min_views: int = 100,         # Lowered to include high-quality academy videos
        max_per_query: int = 50,      # Increased for more search depth
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

    def download_all(self, max_per_dance: int = 200) -> dict[str, list[Path]]:
        """Download videos for all configured dance forms."""
        results: dict[str, list[Path]] = {}
        for dance in self.dances:
            logger.info(f"[{dance.upper()}] Starting download …")
            paths = self.download_dance(dance, max_videos=max_per_dance)
            results[dance] = paths
            logger.info(f"[{dance.upper()}] Downloaded {len(paths)} videos.")
        return results

    def download_dance(self, dance_form: str, max_videos: int = 200) -> list[Path]:
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

            time.sleep(1) 
        return downloaded

    def _build_ytdlp_base_cmd(self) -> list[str]:
        # Added --cookies-from-browser to bypass bot detection
        cmd = ["yt-dlp", "--no-warnings", "--quiet", "--cookies-from-browser", "chrome"]
        if self.cookies_file:
            cmd += ["--cookies", self.cookies_file]
        if self.proxy:
            cmd += ["--proxy", self.proxy]
        return cmd

    def _search_youtube(self, query: str, limit: int = 50) -> list[dict]:
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
                if not line.strip(): continue
                try:
                    metas.append(json.loads(line))
                except json.JSONDecodeError: continue
            return metas
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def _passes_filters(self, meta: dict, dance_form: str) -> bool:
        title = meta.get("title", "")
        duration = meta.get("duration", 0) or 0
        view_count = meta.get("view_count", 0) or 0

        dur_sec = _parse_duration(duration)
        if dur_sec < self.min_duration_sec or dur_sec > self.max_duration_sec:
            return False

        if view_count < self.min_views:
            return False

        title_lower = title.lower()
        for neg_kw in NEGATIVE_KEYWORDS:
            if neg_kw in title_lower:
                return False

        dance_display = dance_form.replace("_", " ").lower()
        title_score = _score_title(title)
        has_dance_name = (dance_display in title_lower or dance_form[:4] in title_lower)

        if not has_dance_name and title_score < 0.1:
            return False

        return True

    def _download_video(self, video_id: str, out_dir: Path, dance_form: str) -> Optional[Path]:
        out_template = str(out_dir / f"{dance_form}_%(id)s.%(ext)s")
        # Ensure cookies-from-browser is used in the download command as well
        cmd = self._build_ytdlp_base_cmd() + [
            "-f", self.preferred_format,
            "-o", out_template,
            "--merge-output-format", "mp4",
            "--no-playlist",
            f"https://www.youtube.com/watch?v={video_id}",
        ]

        if self.dry_run: return None

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                for f in out_dir.glob(f"{dance_form}_{video_id}.mp4"):
                    return f
        except Exception as e:
            logger.error(f"Download failed {video_id}: {e}")
        return None


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="NatyaVeda YouTube Downloader")
    parser.add_argument("--dances", nargs="+", default=list(DANCE_QUERIES.keys()))
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--max-per-dance", type=int, default=200)
    parser.add_argument("--min-duration", type=int, default=15) # Matches Shorts
    parser.add_argument("--max-duration", type=int, default=1800)
    args = parser.parse_args()

    dl = DanceVideoDownloader(
        output_dir=args.output,
        dances=args.dances,
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration,
    )
    results = dl.download_all(max_per_dance=args.max_per_dance)
    total = sum(len(v) for v in results.values())
    print(f"\n✅ Total downloaded: {total} videos")


if __name__ == "__main__":
    main()