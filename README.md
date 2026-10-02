# fal AI MiniMax H3 Max Recast Testing

Paste two photos and get the "Hotel Lobby" meme clip with those two people in it, made with fal's
**H3 Max Recast** model.

## Run

Needs Python 3.11+.

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   (then put your fal key in .env)
.venv\Scripts\python app.py
```

Then open http://localhost:5056 (or double-click `start.bat` after the setup above).

## Use

Click a person's card and press Ctrl+V to paste a photo (or drop / choose files). Add up to 4 photos per person;
different angles help. **Enhance photos** shows the AI-cleaned portraits first; **Make the video** runs the clip.

## Pipeline

1. **Enhance** (`fal-ai/nano-banana-pro/edit`, $0.15/person): all of a person's photos become one sharp,
   front-facing, glasses-free 2K portrait. Untick "AI-enhance" to send the first raw photo instead.
2. **Recast** (`fal-ai/minimax-h3-max-recast-kv` (H3 Max Recast; override with `RECAST_ENDPOINT` in `.env`)): swaps the two portraits into the clip left to right, keeping
   motion, camera, cuts and audio. The prompt forbids glasses and sunglasses in every frame.

**Video.** `assets/default_source.mp4` (the 10 s hotel lobby clip) is the default; `../Ref. Video.mp4` is the older 31.8 s clip if you want to upload it. On the page you can paste a link (X, YouTube,
TikTok, Instagram, direct .mp4 via yt-dlp), drop or upload a file, or paste a video file. Shot cuts are detected and the
longest valid window is pre-selected (Recast needs 5-30 s with no shot over 15 s); adjust Start/End to shorten it and
lower the cost.

**Quality.** A vision model (Claude Sonnet 5 via fal) describes the original performers' hair and eyewear and the new
people's hair and face, and the prompt says exactly what to remove. Without that, Recast tended to keep the
performers' dreadlocks and sunglasses. The prompt used is shown under the button after each run.
