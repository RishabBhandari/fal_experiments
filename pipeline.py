"""Video + photo helpers for the Recast pipeline.

Videos: ingest (upload or link), find shot cuts, pick/validate a window that fits Recast's limits, trim.
Photos: upscale tiny images, AI-clean a person's photos into one portrait, describe people with a vision model.
Prompt: names what each original performer has that must go (hair, eyewear), since Recast otherwise tends to keep
distinctive features like dreadlocks and sunglasses.
"""
import json
import re
import subprocess
from pathlib import Path

import cv2
import fal_client
import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ENHANCE = "fal-ai/nano-banana-pro/edit"
VISION = "openrouter/router/vision"
VISION_MODEL = "anthropic/claude-sonnet-5"

MIN_LEN, MAX_LEN, MAX_SHOT = 5.0, 30.0, 14.8  # Recast: 5-30 s total, no shot over 15 s (small safety margin)
MAX_SOURCE = 600  # seconds; longer uploads/links are rejected
MIN_SIDE = 768  # Recast rejects photos under 256x256; small pastes/screenshots get upscaled well past that


def _run(args: list, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run([FFMPEG, "-hide_banner", *args], capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


# ---------- video ----------

def download(url: str, dst: Path) -> Path:
    """Fetch a video from a direct link or a social post (X, YouTube, TikTok, Instagram...) with yt-dlp."""
    import yt_dlp
    opts = {
        "outtmpl": str(dst.with_suffix(".%(ext)s")), "noplaylist": True, "quiet": True, "no_warnings": True,
        "format": "bv*[vcodec^=avc1][height<=1080]+ba/b[ext=mp4]/bv*[height<=1080]+ba/b",
        "merge_output_format": "mp4", "ffmpeg_location": FFMPEG, "max_filesize": 500 * 1024 * 1024,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if (info.get("duration") or 0) > MAX_SOURCE:
            raise ValueError("That video is over 10 minutes. Use a shorter one.")
    found = sorted(dst.parent.glob(dst.stem + ".*"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise ValueError("Couldn't download a video from that link.")
    return found[-1]


def normalize(src: Path, dst: Path) -> Path:
    """Re-encode any upload to browser/Recast-friendly H.264 + AAC MP4, max 1080p tall."""
    r = _run(["-y", "-i", str(src), "-t", str(MAX_SOURCE), "-vf", "scale=-2:'min(1080,ih)'", "-c:v", "libx264",
              "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
              "-movflags", "+faststart", str(dst)])
    if r.returncode != 0 or not dst.exists():
        raise ValueError("Couldn't read that video file. Try an MP4 or MOV.")
    return dst


def probe(path: Path) -> dict:
    """Duration and shot-cut times (seconds)."""
    r = _run(["-i", str(path), "-vf", "scale=320:-2,select='gt(scene,0.3)',showinfo", "-an", "-f", "null", "-"])
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    if not m:
        raise ValueError("Couldn't read that video.")
    duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    cuts = [round(float(t), 2) for t in re.findall(r"pts_time:([\d.]+)", r.stderr)]
    return {"duration": round(duration, 2), "cuts": cuts}


def check_window(meta: dict, start: float, end: float) -> str | None:
    """Return a problem description, or None if [start, end] is a valid Recast input."""
    length = end - start
    if start < 0 or end > meta["duration"] + 0.05:
        return "The selection is outside the video."
    if length < MIN_LEN:
        return f"The selection must be at least {MIN_LEN:g} seconds."
    if length > MAX_LEN + 0.05:
        return f"The selection must be at most {MAX_LEN:g} seconds."
    edges = [start] + [c for c in meta["cuts"] if start < c < end] + [end]
    longest = max(b - a for a, b in zip(edges, edges[1:]))
    if longest > MAX_SHOT:
        return f"One shot in the selection is {longest:.1f} s long; Recast allows at most 15 s per shot."
    return None


def auto_window(meta: dict) -> tuple[float, float]:
    """Earliest, longest window (up to 30 s) that satisfies Recast's limits."""
    dur = meta["duration"]
    length = min(dur, MAX_LEN)
    while length >= MIN_LEN:
        s = 0.0
        while s + length <= dur + 1e-6:
            if check_window(meta, s, s + length) is None:
                return round(s, 2), round(s + length, 2)
            s += 0.1
        length -= 1.0
    raise ValueError("This video has a single shot longer than 15 seconds everywhere, which Recast can't take. "
                     "Pick a video with more cuts, or trim it first.")


def trim(src: Path, start: float, end: float, dst: Path) -> Path:
    if not dst.exists():
        r = _run(["-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end - start:.3f}", "-c:v", "libx264",
                  "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                  "-movflags", "+faststart", str(dst)])
        if r.returncode != 0:
            raise ValueError("Couldn't trim the video.")
    return dst


def frame_grid(src: Path, meta: dict, start: float, end: float, dst: Path) -> Path:
    """One frame from the middle of each shot in the window (up to 6), tiled, for the vision model."""
    edges = [start] + [c for c in meta["cuts"] if start < c < end] + [end]
    mids = [(a + b) / 2 for a, b in zip(edges, edges[1:])][:6]
    cap = cv2.VideoCapture(str(src))
    frames = []
    for t in mids:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1]))))
    cap.release()
    if not frames:
        raise ValueError("Couldn't read frames from the video.")
    cols = min(3, len(frames))
    while len(frames) % cols:
        frames.append(np.zeros_like(frames[0]))
    rows = [np.hstack(frames[i:i + cols]) for i in range(0, len(frames), cols)]
    cv2.imwrite(str(dst), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return dst


# ---------- photos ----------

def prepare_photo(data: bytes, dst: Path) -> Path:
    """Decode any image, upscale so the short side is at least MIN_SIDE, save as high-quality JPEG."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Couldn't read one of the photos. Try a JPG or PNG.")
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        s = MIN_SIDE / min(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dst


# A conservative edit of the first photo (not a from-scratch headshot): regenerating a studio portrait drifted the
# face (slimmer, older). Hats come off so Recast sees the real hair; glasses come off so the video has none.
ENHANCE_PROMPT = (
    "All images show the SAME real person. Edit image 1 into a clean, sharp, high-resolution, evenly lit photo of "
    "this exact person from the chest up, facing the camera, on a plain light-gray background. Remove any glasses or "
    "sunglasses and any hat or cap: show their natural open eyes and their real hair, using the other images to see "
    "the hair, hairline and face from other angles. Keep the face identical to the photos: same face shape and "
    "width, weight, cheeks, jaw, nose, eyes, eyebrows, mouth, teeth, ears, skin tone and texture, facial hair and "
    "apparent age. Do not slim, beautify, smooth the skin, restyle the hair, or make them look older or younger. "
    "Photorealistic, one person, no text."
)


def enhance(urls: list[str]) -> str:
    r = fal_client.subscribe(ENHANCE, arguments={"prompt": ENHANCE_PROMPT, "image_urls": urls, "resolution": "2K",
                                                 "aspect_ratio": "3:4", "output_format": "jpeg", "num_images": 1})
    return r["images"][0]["url"]


def _vision_json(prompt: str, urls: list[str]) -> dict:
    try:
        r = fal_client.subscribe(VISION, arguments={"model": VISION_MODEL, "prompt": prompt, "image_urls": urls,
                                                    "temperature": 0.1})
        m = re.search(r"\{.*\}", r.get("output", ""), re.S)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}  # descriptions only sharpen the prompt; the job still works without them


SCENE_PROMPT = (
    "These frames come from one video (one frame per shot). Identify the TWO main people, LEFT and RIGHT as they "
    "stand in the wide shots. Reply with JSON only: {\"left\": {...}, \"right\": {...}} where each has: \"who\" (a "
    "short visual identifier, e.g. 'man in striped shirt at the microphone'), \"hair\" (length, color, texture, "
    "style), \"eyewear\" (exactly \"none\" or what they wear), \"headwear\" (\"none\" or what), \"facial_hair\". "
    "Under 15 words per value. Describe appearance only; do not guess who they are."
)

PERSON_PROMPT = (
    "Describe this person's head for a video artist who must recreate them. Reply with JSON only: \"hair\" (length, "
    "color, texture, style), \"hairline\", \"facial_hair\" (\"none\" or description), \"face\" (shape, build, "
    "notable features, apparent age range), \"skin_tone\". Ignore any glasses or hats. Under 20 words per value. "
    "Describe appearance only; do not guess who they are."
)


def describe_scene(grid_url: str) -> dict:
    return _vision_json(SCENE_PROMPT, [grid_url])


def describe_person(url: str) -> dict:
    return _vision_json(PERSON_PROMPT, [url])


# ---------- prompt ----------

def _side(label: str, n: int, orig: dict, new: dict) -> str:
    who = orig.get("who") or f"main person on the {label}"
    if not re.match(r"(the|a|an)\s", who, re.I):
        who = "the " + who
    lines = [f"{label}: the person in reference photo {n} replaces {who}."]
    if new:
        lines.append(f"New look: hair {new.get('hair', '')}; hairline {new.get('hairline', '')}; facial hair "
                     f"{new.get('facial_hair', 'none')}; face {new.get('face', '')}; skin {new.get('skin_tone', '')}.")
    none = ("none", "")
    gone = []
    if orig.get("hair"):
        gone.append(f"the original hair ({orig['hair']})")
    if orig.get("eyewear", "none").lower() not in none:
        gone.append(f"the {orig['eyewear']}")
    if orig.get("facial_hair", "none").lower() not in none and new.get("facial_hair", "").lower().startswith("none"):
        gone.append(f"the original facial hair ({orig['facial_hair']})")
    if gone:
        lines.append("Completely remove " + ", ".join(gone) + "; none of it remains in any frame.")
    return " ".join(lines)


# Structure follows the Replace / Match / Preserve pattern from fal's and MiniMax's character-replacement guides.
def build_prompt(scene: dict, left: dict, right: dict, extra: str = "") -> str:
    parts = [
        "Replace the two main performers with the people in the reference photos.",
        _side("LEFT", 1, scene.get("left", {}), left),
        _side("RIGHT", 2, scene.get("right", {}), right),
        "Match each new person's face, hair, hairline, skin tone and identity exactly in every shot, including "
        "close-ups and profile views. Neither person wears glasses or sunglasses; their own natural eyes are open "
        "and visible.",
        "Preserve the original movement, pose, expression, mouth movements, timing, clothing, jewelry, background, "
        "camera path, framing, lighting, cuts and audio.",
    ]
    if extra:
        parts.append(extra)
    return " ".join(p for p in parts if p)[:2000]
