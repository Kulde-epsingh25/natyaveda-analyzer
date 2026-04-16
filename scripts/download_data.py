"""
NatyaVeda — Download Data Script (CLI wrapper)
Delegates to src/data_collection/youtube_downloader.py

Usage:
  python scripts/download_data.py --dances bharatanatyam kathak --max-per-dance 50
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_collection.youtube_downloader import main
if __name__ == "__main__":
    main()
