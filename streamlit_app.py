"""Streamlit version of the Recast site, for Streamlit Community Cloud. Same pipeline as app.py (see pipeline.py).

Secrets (Streamlit Cloud: App settings > Secrets; locally: .streamlit/secrets.toml or .env):
    FAL_KEY = "..."          # required
    APP_PASSWORD = "..."     # recommended: anyone with the link could otherwise spend your fal credits
"""
import hashlib
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fal_client
import streamlit as st
from dotenv import load_dotenv

import pipeline as pl

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
RECAST = os.environ.get("RECAST_ENDPOINT", "fal-ai/minimax-h3-max-recast-kv")
DEFAULT_VIDEO = ROOT / "assets" / "default_source.mp4"
WORK = ROOT / "uploads" / "st"
WORK.mkdir(parents=True, exist_ok=True)
MAX_PHOTOS = 4
PRICE = {"768P": 0.30, "1080P": 0.45}

st.set_page_config(page_title="H3 Max Recast Testing", page_icon="🎬", layout="centered")


def secret(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file (local run): fall back to .env / environment
        pass
    return os.environ.get(name, "")


os.environ["FAL_KEY"] = secret("FAL_KEY")
PASSWORD = secret("APP_PASSWORD")
ss = st.session_state

st.markdown("# :orange[fal AI] MiniMax H3 Max Recast Testing")

# ---------- password gate ----------
if PASSWORD and not ss.get("authed"):
    pw = st.text_input("Password", type="password")
    if pw:
        if pw == PASSWORD:
            ss.authed = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()
if not os.environ["FAL_KEY"]:
    st.error("FAL_KEY isn't set. Add it under App settings → Secrets.")
    st.stop()

st.caption("Pick a video, add a few photos of two people. AI cleans them up, then fal Recast puts them in.")


# ---------- video ----------
@st.cache_data(show_spinner=False)
def video_meta(path: str, mtime: float) -> dict:
    meta = pl.probe(Path(path))
    meta["start"], meta["end"] = pl.auto_window(meta)
    return meta


def set_video(path: Path, name: str):
    meta = video_meta(str(path), path.stat().st_mtime)
    ss.video = {"path": str(path), "name": name, **meta}
    ss.window = (meta["start"], meta["end"])


def ingest(fn, name: str):
    vid = uuid.uuid4().hex[:12]
    raw = WORK / f"{vid}_raw"
    try:
        with st.spinner("Preparing the video…"):
            raw = fn(raw)
            out = pl.normalize(raw, WORK / f"{vid}.mp4")
            set_video(out, name)
    except Exception as e:
        st.error(f"Couldn't load that video: {e}")
    finally:
        for p in WORK.glob(f"{vid}_raw*"):
            p.unlink(missing_ok=True)


if "video" not in ss:
    set_video(DEFAULT_VIDEO, "Hotel lobby")

st.subheader("1 · Video")
src = st.radio("Source", ["Default video", "Upload a file", "Paste a link"], horizontal=True,
               label_visibility="collapsed")
if src == "Default video" and ss.video["path"] != str(DEFAULT_VIDEO):
    set_video(DEFAULT_VIDEO, "Hotel lobby")
elif src == "Upload a file":
    f = st.file_uploader("Video file", type=["mp4", "mov", "webm", "m4v", "mkv"])
    if f and ss.get("video_upload") != f.file_id:
        ss.video_upload = f.file_id

        def save(raw, f=f):
            dst = raw.with_suffix(Path(f.name).suffix or ".mp4")
            dst.write_bytes(f.getvalue())
            return dst
        ingest(save, f.name)
elif src == "Paste a link":
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    url = c1.text_input("Video link", placeholder="X, YouTube, TikTok, Instagram, .mp4…")
    if c2.button("Load", use_container_width=True) and url.strip():
        ingest(lambda raw: pl.download(url.strip(), raw), url.strip())

v = ss.video
start, end = st.slider("Part of the video to use (seconds)", 0.0, float(v["duration"]), ss.window, step=0.1)
ss.window = (start, end)
st.video(v["path"], start_time=start, end_time=end, loop=True, muted=True)
problem = pl.check_window(v, start, end)
shots = len(v["cuts"]) + 1
st.caption(f"{v['name']} · {v['duration']:.1f} s total · {shots} shot{'s' if shots > 1 else ''} · "
           "Recast needs 5–30 s with no shot over 15 s.")
if problem:
    st.error(problem)

# ---------- photos ----------
st.subheader("2 · Photos")
st.caption("Drag photos in or browse. Put each person's clearest, front-facing photo first; extra angles (no hat) "
           "help with hair and profile shots.")
cols = st.columns(2)
files = {}
for col, side in zip(cols, ("left", "right")):
    with col:
        up = st.file_uploader(f"{side.title()} person (up to {MAX_PHOTOS})", type=["jpg", "jpeg", "png", "webp"],
                              accept_multiple_files=True, key=f"photos_{side}")
        files[side] = (up or [])[:MAX_PHOTOS]
        if files[side]:
            st.image([f.getvalue() for f in files[side]], width=110)
ready = bool(files["left"] and files["right"])
photo_key = hashlib.sha1("".join(f.file_id for s in ("left", "right") for f in files[s]).encode()).hexdigest()
if ss.get("portraits_key") != photo_key:
    ss.portraits = None  # photos changed: old enhanced versions no longer apply

# ---------- options ----------
st.subheader("Options")
enh = st.checkbox("AI-enhance photos (sharpen, remove glasses & hats)", value=True)
res = st.selectbox("Quality", ["768P", "1080P"], format_func=lambda r: f"{r[:-1]}p · ${PRICE[r]:.2f}/s"
                   + (" (sharper)" if r == "1080P" else ""))
extra = st.text_area("Extra instructions for the video (optional)", placeholder="e.g. Keep their hats on.",
                     max_chars=400)


def upload_side(side: str) -> list[str]:
    tag = uuid.uuid4().hex[:10]
    return [fal_client.upload_file(str(pl.prepare_photo(f.getvalue(), WORK / f"{tag}_{side}{i}.jpg")))
            for i, f in enumerate(files[side])]


def run_enhance():
    with st.spinner("Enhancing photos (sharpening, removing glasses and hats)…"):
        with ThreadPoolExecutor(2) as pool:
            left, right = pool.map(lambda s: pl.enhance(upload_side(s)), ("left", "right"))
    ss.portraits, ss.portraits_key = {"left": left, "right": right}, photo_key


length = end - start
cost = length * PRICE[res] + (0.30 if enh and not ss.get("portraits") else 0)
b1, b2 = st.columns(2)
if b1.button("Enhance photos (~$0.30)", disabled=not (ready and enh), use_container_width=True):
    try:
        run_enhance()
    except Exception as e:
        st.error(f"Couldn't enhance the photos: {e}")
go = b2.button(f"Make the video · {length:.1f} s · ~${cost:.2f}", type="primary",
               disabled=not ready or bool(problem), use_container_width=True)

if enh and ss.get("portraits"):
    st.subheader("3 · Enhanced photos")
    p1, p2 = st.columns(2)
    p1.image(ss.portraits["left"], caption="Left person")
    p2.image(ss.portraits["right"], caption="Right person")

# ---------- recast ----------
if go:
    try:
        if enh and not ss.get("portraits"):
            run_enhance()  # portraits appear above on the next rerun
        with st.status("Making the video…", expanded=True) as status:
            st.write("Trimming the video and analyzing the people in it…")
            refs = ([ss.portraits["left"], ss.portraits["right"]] if enh
                    else [upload_side("left")[0], upload_side("right")[0]])
            key = hashlib.sha1(f"{v['path']}:{start}:{end}".encode()).hexdigest()[:12]
            clip = pl.trim(Path(v["path"]), start, end, WORK / f"clip_{key}.mp4")
            grid = pl.frame_grid(Path(v["path"]), v, start, end, WORK / f"grid_{key}.jpg")
            with ThreadPoolExecutor(4) as pool:
                jobs = [pool.submit(fal_client.upload_file, str(clip)),
                        pool.submit(lambda: pl.describe_scene(fal_client.upload_file(str(grid)))),
                        pool.submit(pl.describe_person, refs[0]), pool.submit(pl.describe_person, refs[1])]
                video_url, scene, left_d, right_d = [j.result() for j in jobs]
            prompt = pl.build_prompt(scene, left_d, right_d, extra.strip())
            handle = fal_client.submit(RECAST, arguments={"video_url": video_url, "reference_image_urls": refs,
                                                          "prompt": prompt, "resolution": res})
            line, t0 = st.empty(), time.time()
            while True:
                s = fal_client.status(RECAST, handle.request_id)
                if isinstance(s, fal_client.Completed):
                    break
                waiting = isinstance(s, fal_client.Queued)
                line.write(f"{'Waiting in line' if waiting else 'Recasting'}… {int(time.time() - t0)} s "
                           "(usually a few minutes)")
                time.sleep(4)
            result = fal_client.result(RECAST, handle.request_id)
            status.update(label="Done! 🎉", state="complete", expanded=False)
        ss.result = {"video": result["video"]["url"], "prompt": prompt}
    except Exception as e:
        st.error(f"Something went wrong: {e}")

if ss.get("result"):
    st.video(ss.result["video"], loop=True)
    st.markdown(f"[Download MP4]({ss.result['video']})")
    with st.expander("Prompt sent to Recast"):
        st.write(ss.result["prompt"])
