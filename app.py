"""Hotel Lobby x fal Recast: give a few photos of two people (and optionally any video), get the clip with them in it.

Pipeline (see pipeline.py):
1. Video: upload a file or paste a link; it's normalized, shot cuts are found, and a window that fits Recast's
   limits (5-30 s, no shot over 15 s) is picked. The bundled clip is the default.
2. Enhance: each person's photos go to Nano Banana Pro edit -> one sharp, glasses-free, hat-free photo.
3. Describe: a vision model notes the original performers' hair/eyewear and the new people's hair/face, so the
   prompt can say exactly what must change.
4. Recast: minimax/h3-max/recast (H3 Max) swaps the two people in, keeping motion, camera, cuts and audio.

The fal key stays in .env on the server and never reaches the browser.
"""
import hashlib
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fal_client
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request, send_from_directory

import pipeline as pl

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# Private fal deployment of H3 Max Recast (the public "minimax/h3-max/recast" routes to the same thing).
RECAST = os.environ.get("RECAST_ENDPOINT", "fal-ai/minimax-h3-max-recast-kv")
DEFAULT_VIDEO = ROOT / "assets" / "default_source.mp4"
UPLOADS = ROOT / "uploads"
VIDEOS = UPLOADS / "videos"
VIDEOS.mkdir(parents=True, exist_ok=True)
MAX_PHOTOS = 4  # per person

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
_cache = {}  # (kind, key) -> value: fal URLs for trimmed clips, scene descriptions, person descriptions


def video_path(vid: str) -> Path:
    if vid == "default":
        return DEFAULT_VIDEO
    if not re.fullmatch(r"[0-9a-f]{12}", vid or ""):
        abort(404)
    p = VIDEOS / f"{vid}.mp4"
    if not p.exists():
        abort(404)
    return p


def video_meta(vid: str) -> dict:
    p = video_path(vid)
    meta_file = p.with_suffix(".json")
    if meta_file.exists() and meta_file.stat().st_mtime >= p.stat().st_mtime:
        return json.loads(meta_file.read_text())
    meta = pl.probe(p)
    meta["start"], meta["end"] = pl.auto_window(meta)
    meta_file.write_text(json.dumps(meta))
    return meta


def cached(kind: str, key: str, fn):
    if (kind, key) not in _cache:
        _cache[(kind, key)] = fn()
    return _cache[(kind, key)]


def upload_photos(side: str) -> list[str]:
    files = request.files.getlist(side)[:MAX_PHOTOS]
    if not files:
        raise ValueError("Add at least one photo for each person.")
    tag = uuid.uuid4().hex[:10]
    return [fal_client.upload_file(str(pl.prepare_photo(f.read(), UPLOADS / f"{tag}_{side}{i}.jpg")))
            for i, f in enumerate(files)]


def friendly(e: Exception) -> str:
    # fal validation errors come back as a list of dicts; show just their messages
    msgs = re.findall(r"'msg': '([^']+)'", str(e))
    return " ".join(msgs) or str(e)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/video/<vid>.mp4")
def serve_video(vid):
    p = video_path(vid)
    return send_from_directory(p.parent, p.name, max_age=0)


@app.get("/api/video/default")
def default_video():
    return jsonify(id="default", src="/video/default.mp4", name="Hotel lobby", **video_meta("default"))


@app.post("/api/video")
def add_video():
    """A video file or a link -> normalized copy + duration, cuts and a suggested window."""
    vid = uuid.uuid4().hex[:12]
    raw = VIDEOS / f"{vid}_raw"
    try:
        f = request.files.get("video")
        url = (request.form.get("url") or "").strip()
        if f:
            raw = raw.with_suffix(Path(f.filename or "").suffix or ".mp4")
            f.save(raw)
            name = f.filename or "Uploaded video"
        elif re.match(r"https?://", url):
            raw = pl.download(url, raw)
            name = url
        else:
            return jsonify(error="Drop a video file or paste a video link."), 400
        pl.normalize(raw, VIDEOS / f"{vid}.mp4")
        meta = video_meta(vid)
    except Exception as e:
        return jsonify(error=friendly(e)), 400
    finally:
        for p in VIDEOS.glob(f"{vid}_raw*"):
            p.unlink(missing_ok=True)
    return jsonify(id=vid, src=f"/video/{vid}.mp4", name=name, **meta)


@app.post("/api/enhance")
def enhance_route():
    """Photos for both people -> one AI-cleaned photo each."""
    try:
        sets = [upload_photos("left"), upload_photos("right")]
        with ThreadPoolExecutor(2) as pool:
            left, right = pool.map(pl.enhance, sets)
    except Exception as e:
        return jsonify(error=f"Couldn't enhance the photos: {friendly(e)}"), 502
    return jsonify(left=left, right=right)


@app.post("/api/generate")
def generate():
    """Two photos (fal URLs from /api/enhance, or raw files) + video window -> queued Recast job."""
    if not os.environ.get("FAL_KEY"):
        return jsonify(error="FAL_KEY is missing from .env"), 500
    form = request.form
    resolution = "1080P" if form.get("resolution") == "1080P" else "768P"
    extra = (form.get("prompt") or "").strip()[:400]
    vid = form.get("video_id") or "default"
    try:
        src, meta = video_path(vid), video_meta(vid)
        start, end = round(float(form.get("start", meta["start"])), 2), round(float(form.get("end", meta["end"])), 2)
        problem = pl.check_window(meta, start, end)
        if problem:
            return jsonify(error=problem), 400

        refs = []
        for side in ("left", "right"):
            url = form.get(f"{side}_url", "")
            refs.append(url if url.startswith("https://") else upload_photos(side)[0])

        key = hashlib.sha1(f"{vid}:{src.stat().st_mtime}:{start}:{end}".encode()).hexdigest()[:12]
        clip = pl.trim(src, start, end, VIDEOS / f"clip_{key}.mp4")

        def clip_url():
            return cached("clip", key, lambda: fal_client.upload_file(str(clip)))

        def scene():
            grid = pl.frame_grid(src, meta, start, end, VIDEOS / f"grid_{key}.jpg")
            return cached("scene", key, lambda: pl.describe_scene(fal_client.upload_file(str(grid))))

        def person(url):
            return cached("person", url, lambda: pl.describe_person(url))

        with ThreadPoolExecutor(4) as pool:  # upload + three vision calls at once (~5 s total)
            jobs = [pool.submit(clip_url), pool.submit(scene), pool.submit(person, refs[0]),
                    pool.submit(person, refs[1])]
            video_url, scene_desc, left_desc, right_desc = [j.result() for j in jobs]

        prompt = pl.build_prompt(scene_desc, left_desc, right_desc, extra)
        handle = fal_client.submit(RECAST, arguments={"video_url": video_url, "reference_image_urls": refs,
                                                      "prompt": prompt, "resolution": resolution})
    except Exception as e:
        return jsonify(error=f"Couldn't start the job: {friendly(e)}"), 502
    return jsonify(id=handle.request_id, prompt=prompt)


@app.get("/api/status/<request_id>")
def status(request_id):
    try:
        s = fal_client.status(RECAST, request_id, with_logs=False)
        if isinstance(s, fal_client.Completed):
            result = fal_client.result(RECAST, request_id)
            return jsonify(state="done", video=result["video"]["url"], seed=result.get("seed"))
        if isinstance(s, fal_client.Queued):
            return jsonify(state="queued", position=s.position)
        return jsonify(state="running")
    except Exception as e:
        return jsonify(state="error", error=friendly(e)), 200


if __name__ == "__main__":
    app.run(port=5056, debug=False, threaded=True)
