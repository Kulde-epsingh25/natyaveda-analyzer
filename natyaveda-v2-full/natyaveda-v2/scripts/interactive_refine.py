import argparse
import subprocess
import sys
import time
from pathlib import Path

# Setup absolute paths to handle spaces and parentheses in Windows
BASE_DIR = Path(__file__).parent.parent.absolute()
RAW_DIR = BASE_DIR / "data" / "raw"
REFINED_DIR = BASE_DIR / "data" / "refined"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EXTRACT_SCRIPT = BASE_DIR / "scripts" / "extract_features.py"


def parse_time_to_seconds(raw: str) -> float:
    """Accept HH:MM:SS(.ms), MM:SS(.ms), or seconds as float."""
    value = raw.strip()
    if not value:
        raise ValueError("Empty time value")

    if ":" not in value:
        return float(value)

    parts = value.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

    raise ValueError(f"Invalid time format: {raw}")


def seconds_to_hhmmss(seconds: float) -> str:
    whole = int(max(0, seconds))
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def ask_time(prompt: str, default_seconds: float | None = None) -> float:
    suffix = ""
    if default_seconds is not None:
        suffix = f" [default {seconds_to_hhmmss(default_seconds)}]"

    while True:
        raw = input(f"  {prompt}{suffix}: ").strip()
        if not raw and default_seconds is not None:
            return default_seconds
        try:
            return parse_time_to_seconds(raw)
        except ValueError as exc:
            print(f"  Invalid time: {exc}. Try HH:MM:SS, MM:SS, or seconds.")


def collect_videos(raw_dir: Path, glob_patterns: list[str]) -> list[Path]:
    videos: list[Path] = []
    for dance_dir in sorted(raw_dir.iterdir()):
        if not dance_dir.is_dir():
            continue

        found: list[Path] = []
        for pattern in glob_patterns:
            found.extend(sorted(dance_dir.glob(pattern)))

        if found:
            print(f"  [+] Found {len(found)} videos in {dance_dir.name}")
            videos.extend(found)

    return videos


def make_player_command(video_path: Path, width: int, height: int) -> list[str]:
    return [
        "ffplay",
        "-x", str(width),
        "-y", str(height),
        "-alwaysontop",
        "-window_title", f"REFINE: {video_path.name}",
        "-i", str(video_path),
    ]


def save_clip(source_video: Path, out_path: Path, start_sec: float, end_sec: float) -> bool:
    if end_sec <= start_sec:
        print("  End time must be greater than start time.")
        return False

    cut_cmd = [
        "ffmpeg",
        "-y",
        "-ss", seconds_to_hhmmss(start_sec),
        "-to", seconds_to_hhmmss(end_sec),
        "-i", str(source_video),
        "-c", "copy",
        "-map", "0:v:0",
        str(out_path),
    ]

    result = subprocess.run(cut_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return result.returncode == 0


def auto_clip_video(source_video: Path, out_dir: Path, clip_sec: int) -> bool:
    """Split one video into fixed-length video-only clips."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"auto_{source_video.stem}_%03d.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(source_video),
        "-map", "0:v:0",
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(clip_sec),
        "-reset_timestamps", "1",
        str(pattern),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return result.returncode == 0


def run_auto_clip_all(clip_sec: int):
    print("--- NATYAVEDA AUTO CLIPPER ---")
    print(f"Clip length: {clip_sec}s")
    print(f"Checking: {RAW_DIR}")

    if not RAW_DIR.exists():
        print(f"ERROR: Folder '{RAW_DIR}' not found!")
        return

    videos = collect_videos(RAW_DIR, ["*.mp4", "*.webm", "*.mkv"])
    if not videos:
        print("ERROR: No videos found. Check if data/raw has dance subfolders!")
        return

    REFINED_DIR.mkdir(parents=True, exist_ok=True)
    success = 0

    for index, vid_path in enumerate(videos, start=1):
        dance_form = vid_path.parent.name
        out_dir = REFINED_DIR / dance_form
        print(f"[{index}/{len(videos)}] Clipping {vid_path.name} ...")
        if auto_clip_video(vid_path, out_dir, clip_sec):
            success += 1
        else:
            print(f"  Failed: {vid_path.name}")

    print(f"Done. Clipped {success}/{len(videos)} videos into {clip_sec}s segments.")


def run_interactive(default_clip_sec: int, width: int, height: int):
    print(f"--- NATYAVEDA INTERACTIVE REFINER ---")
    print(f"📂 Checking: {RAW_DIR}")
    
    if not RAW_DIR.exists():
        print(f"❌ ERROR: Folder '{RAW_DIR}' not found!")
        return

    videos = collect_videos(RAW_DIR, ["*.mp4", "*.webm", "*.mkv"])

    if not videos:
        print("❌ ERROR: No videos found. Check if data/raw has dance subfolders!")
        return

    REFINED_DIR.mkdir(parents=True, exist_ok=True)

    last_start = 0.0
    last_end = float(default_clip_sec)
    last_duration = float(default_clip_sec)

    for index, vid_path in enumerate(videos, start=1):
        dance_form = vid_path.parent.name
        print(f"\n{'='*60}\n[{index}/{len(videos)}] OPENING: {vid_path.name}\n{'='*60}")

        play_cmd = make_player_command(vid_path, width, height)
        
        try:
            player = subprocess.Popen(play_cmd)
        except Exception as e:
            print(f"❌ Could not start ffplay: {e}")
            continue

        clips_count = 0
        while True:
            print(f"\nCLIP CONTROL for {vid_path.name}:")
            print("  [y] Start+End | [c] Quick clip from start (+default duration)")
            print("  [r] Repeat last duration from a new start")
            print("  [k] Keep last clip times | [s] Skip video | [q] Quit")
            cmd = input("  Command: ").strip().lower()
            
            if cmd == 'q':
                player.terminate()
                return
            if cmd == 's':
                player.terminate()
                break
            if cmd == 'y':
                start_sec = ask_time("Start (HH:MM:SS / MM:SS / sec)", last_start)
                end_sec = ask_time("End   (HH:MM:SS / MM:SS / sec)", last_end)
            elif cmd == 'c':
                start_sec = ask_time("Start (HH:MM:SS / MM:SS / sec)", last_end)
                end_sec = start_sec + float(default_clip_sec)
                print(f"  Using quick duration: +{default_clip_sec}s -> {seconds_to_hhmmss(end_sec)}")
            elif cmd == 'r':
                start_sec = ask_time("Start (HH:MM:SS / MM:SS / sec)", last_end)
                end_sec = start_sec + last_duration
                print(f"  Reusing last duration: +{int(last_duration)}s -> {seconds_to_hhmmss(end_sec)}")
            elif cmd == 'k':
                start_sec = last_start
                end_sec = last_end
                print(f"  Reusing last exact range: {seconds_to_hhmmss(start_sec)} to {seconds_to_hhmmss(end_sec)}")
            else:
                print("  Unknown command. Use y/c/r/k/s/q.")
                continue
                
            out_dir = REFINED_DIR / dance_form
            out_dir.mkdir(parents=True, exist_ok=True)
            clip_name = f"manual_{int(time.time())}_{vid_path.stem}.mp4"
            out_path = out_dir / clip_name

            print("  Cutting clip...")
            ok = save_clip(vid_path, out_path, start_sec, end_sec)
            if not ok:
                print("  Failed to save clip. Check ffmpeg installation and timestamps.")
                continue

            last_start = start_sec
            last_end = end_sec
            last_duration = end_sec - start_sec
            print(f"  Saved: {clip_name}")
            print(f"  Range: {seconds_to_hhmmss(start_sec)} -> {seconds_to_hhmmss(end_sec)}")
            clips_count += 1

            if input("  Add another clip from this video? (y/n): ").strip().lower() != 'y':
                player.terminate()
                break

        # Parallel Extraction for the dance form you just finished
        if clips_count > 0:
            print(f"🚀 Background extraction starting for {dance_form}...")
            # Run extraction only for the dance form you just refined to save VRAM
            extract_cmd = [
                sys.executable, str(EXTRACT_SCRIPT),
                "--input", str(REFINED_DIR),
                "--output", str(PROCESSED_DIR),
                "--dances", dance_form,
                "--device", "cuda"
            ]
            subprocess.Popen(extract_cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NatyaVeda interactive clip refiner")
    parser.add_argument("--default-clip-sec", type=int, default=20,
                        help="Default seconds for quick clip mode (command c)")
    parser.add_argument("--window-width", type=int, default=640,
                        help="ffplay window width")
    parser.add_argument("--window-height", type=int, default=360,
                        help="ffplay window height")
    parser.add_argument("--auto-clip-all", action="store_true",
                        help="Split all videos in data/raw into fixed-length clips and exit")
    parser.add_argument("--clip-sec", type=int, default=15,
                        help="Segment length in seconds for --auto-clip-all")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.auto_clip_all:
        run_auto_clip_all(args.clip_sec)
    else:
        run_interactive(args.default_clip_sec, args.window_width, args.window_height)