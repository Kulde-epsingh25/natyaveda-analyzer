#!/usr/bin/env python3
"""
NatyaVeda — Universal Downloader
Supports single videos and full playlists with automatic skip-logic.
"""

import logging
import subprocess
import time
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent.absolute()
OUTPUT_BASE = BASE_DIR / "data" / "raw"
ARCHIVE_FILE = BASE_DIR / "data" / "downloaded_history.txt"
BROWSER = "chrome"
USE_BROWSER_COOKIES = False
VIDEO_FORMAT = "bestvideo[height>=720]/bestvideo/best"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Links (Kathak only for the next run)
DANCE_VIDEOS = {
    "kathak": [
        "https://youtu.be/UBYqv21c0Yk?si=HHeOscCU88GALoC2",
        "https://youtu.be/BIaaiuuQFWU?si=RMkr8dZ3eL5ni_d7",
        "https://youtube.com/shorts/1odxGD7Unig?si=V5GXhfy_PQhl0jB1",
        "https://youtu.be/Tv7Ft6vQE3o?si=e7e-Apalpl-EgXf7",
        "https://youtu.be/V68xvvzkU1c?si=8pZj0ccEhONXzRxG",
        "https://youtu.be/Ef-eGHmZuVI?si=Ue774DPe708i5TGQ",
        "https://youtu.be/QnSPX9ZTSCk?si=9gAcuVn_t0orNG0i",
        "https://youtu.be/kRLNadwcOzo?si=daY09ZJpA55W2HA3",
        "https://youtu.be/-F6IoIk1v5E?si=WacXV3xUDwccMaL8",
        "https://youtu.be/nMu2ipOVEtQ?si=_GVhRW-ayGzZXa5-",
        "https://youtu.be/fFPMDDbGOW0?si=G9uYWUDUfQk_kS8u",
        "https://youtu.be/1SFyvFP5GSU?si=HiHV6YIkFfifMvFJ",
        "https://youtube.com/playlist?list=PL6tjUn8LhxF5w8sPtlK_uByFD0dPzQM2D&si=V5UhGJTzExa0bFVN",
        "https://youtu.be/yq4dOpYZABA?si=fqC1grbj2vvenL-a",
        "https://youtu.be/6Ap74ucreRA?si=Oder9g8WnC6RdHnT",
        "https://youtu.be/2SOErT9v4rg?si=ApnchO6u509xwqLt",
        "https://youtu.be/PadVHZKfkfE?si=KLV3m-Yi_d4r2Tf5",
        "https://youtu.be/0mwmb1HTapM?si=mXgf7bwcCqvuw0LW",
        "https://youtu.be/A2alNCOJ184?si=YZLwPGhP70z8PNB6",
        "https://youtu.be/cBjWNh_6jCg?si=BQE9ayBtIB7mvjKd",
        "https://youtu.be/6r3jTGfSQe0?si=0oWKJpQYzHQIrHD9",
    ],
}

def download_content(url: str, output_dir: Path, dance_form: str):
    """Download single video or playlist into the dance folder."""
    # Template keeps stable file naming and downloads only video (no audio).
    out_template = str(output_dir / f"{dance_form}_%(id)s.%(ext)s")

    base_cmd = [
        "yt-dlp",
        "--no-warnings",
        "--ignore-errors",
        "--concurrent-fragments", "8",
        "-f", VIDEO_FORMAT,
        "-o", out_template,
        "--download-archive", str(ARCHIVE_FILE),
        url
    ]

    cmd_with_cookies = [*base_cmd[:2], "--cookies-from-browser", BROWSER, *base_cmd[2:]]

    try:
        if USE_BROWSER_COOKIES:
            result = subprocess.run(cmd_with_cookies, check=False)
            if result.returncode == 0:
                return
            logger.warning("Cookie-based download failed, retrying without browser cookies.")

        # We don't use --quiet here so you can see playlist progress.
        subprocess.run(base_cmd, check=True)
    except Exception as e:
        logger.error(f"❌ Error processing {url}: {e}")

def main():
    print(f"\n--- NATYAVEDA UNIVERSAL DOWNLOADER ---")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    for dance_form, urls in DANCE_VIDEOS.items():
        dance_dir = OUTPUT_BASE / dance_form
        dance_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n🚀 Starting Category: [{dance_form.upper()}]")
        
        for url in urls:
            download_content(url, dance_dir, dance_form)
            time.sleep(2) # Prevent bot detection

    logger.info(f"\n✅ All links and playlists processed!")
  
if __name__ == "__main__":
    main()