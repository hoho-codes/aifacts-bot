"""
facts_pipeline.py

Standalone helpers for a "random facts" video pipeline: fetch a raw fact,
polish it with Groq into a punchy short-form script, generate a matching
background image via Hugging Face's FLUX, synthesize narration via
edge-tts (free, no API key), and burn in centered captions via ffmpeg's
drawtext filter. Kept separate from coffee.py since this is a distinct
content pipeline that happens to share infrastructure patterns (ffmpeg,
retries, fallbacks) rather than code.

Requires: pip install edge-tts requests huggingface_hub
ffmpeg must be available on PATH (already true on ubuntu-latest runners).
"""

import asyncio
import os
import random
import subprocess
import textwrap
import requests

FACTS_API_URL = "https://uselessfacts.jsph.pl/api/v2/facts/random?language=en"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

FALLBACK_FACTS = [
    "Honey never spoils. Archaeologists have found 3,000-year-old honey in Egyptian tombs that's still edible.",
    "Octopuses have three hearts, and two of them stop beating when they swim.",
    "Bananas are berries, but strawberries aren't.",
    "A day on Venus is longer than a year on Venus.",
    "Wombat poop is cube-shaped.",
]

IMAGE_STYLE_MODIFIERS = [
    "warm cinematic photography style",
    "soft editorial illustration style",
    "clean minimalist digital art style",
    "vintage textbook illustration style",
    "moody atmospheric photography",
    "bright flat-design vector illustration",
]


# ---------------------------------------------------------------------------
# 1. Fetch a raw fact from the free uselessfacts API
# ---------------------------------------------------------------------------

def fetch_random_fact() -> str:
    last_err = None
    for attempt in range(3):
        try:
            res = requests.get(FACTS_API_URL, timeout=15)
            res.raise_for_status()
            data = res.json()
            fact = data.get("text", "").strip()
            if not fact:
                raise ValueError("Empty fact text in response")
            print(f"Fetched fact: {fact}")
            return fact
        except Exception as e:
            last_err = e
            print(f"fetch_random_fact attempt {attempt + 1} failed ({e}); retrying...")

    print(f"uselessfacts API failed after retries ({last_err}); using fallback fact.")
    return random.choice(FALLBACK_FACTS)


# ---------------------------------------------------------------------------
# 2. Polish the raw fact into a punchy short-form script via Groq
# ---------------------------------------------------------------------------

def polish_fact_with_groq(raw_fact: str) -> str:
    if not GROQ_API_KEY:
        print("No GROQ_API_KEY set; using raw fact unmodified.")
        return raw_fact

    system_instruction = (
        "You rewrite trivia facts into punchy, short-form video scripts. "
        "Keep it under 30 words, one or two short sentences, spoken-language "
        "tone, no hashtags, no emojis, no quotation marks. Open with a hook "
        "if the original fact doesn't already have one. Return ONLY the "
        "rewritten fact, nothing else."
    )

    last_err = None
    for attempt in range(3):
        try:
            res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": raw_fact},
                    ],
                    "max_tokens": 200,
                    "temperature": 0.9,
                    "reasoning_effort": "low",
                },
                timeout=30,
            )
            res.raise_for_status()
            polished = res.json()["choices"][0]["message"]["content"].strip().strip('"')
            if not polished:
                raise ValueError("Groq returned an empty script")
            print(f"Polished fact: {polished}")
            return polished
        except Exception as e:
            last_err = e
            print(f"polish_fact_with_groq attempt {attempt + 1} failed ({e}); retrying...")

    print(f"Groq polishing failed after retries ({last_err}); using raw fact unmodified.")
    return raw_fact


def get_fact_script() -> str:
    """Convenience wrapper: fetch a raw fact, then polish it."""
    raw_fact = fetch_random_fact()
    return polish_fact_with_groq(raw_fact)


# ---------------------------------------------------------------------------
# 3. Generate a background image matching the fact via Groq + FLUX
# ---------------------------------------------------------------------------

def build_image_prompt_with_groq(fact_text: str) -> str:
    """
    Turns the fact into a short visual scene description suitable for an
    image generator -- the raw fact text itself is usually not a usable
    image prompt (too abstract, too sentence-shaped), so this asks Groq
    to imagine a single concrete visual that represents the fact.
    Falls back to a generic prompt built from the fact text if Groq fails.
    """
    style = random.choice(IMAGE_STYLE_MODIFIERS)

    if not GROQ_API_KEY:
        print("No GROQ_API_KEY set; using a basic fallback image prompt.")
        return f"A simple illustration representing: {fact_text}. {style}."

    system_instruction = (
        "You turn trivia facts into short, concrete prompts for an AI image "
        "generator. Describe a single clear visual scene that represents the "
        "fact -- no text, no words, no diagrams, just a real scene or object. "
        f"Render it in this style: {style}. "
        "Compose the scene vertically, with the subject centered and filling "
        "the frame. The final image must be a full-bleed photo or illustration "
        "with no borders, frames, white margins, or visible photo-paper edges. "
        "Under 25 words total. Return ONLY the prompt text, nothing else."
    )

    last_err = None
    for attempt in range(3):
        try:
            res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": fact_text},
                    ],
                    "max_tokens": 150,
                    "temperature": 1.0,
                    "reasoning_effort": "low",
                },
                timeout=30,
            )
            res.raise_for_status()
            image_prompt = res.json()["choices"][0]["message"]["content"].strip().strip('"')
            if not image_prompt:
                raise ValueError("Groq returned an empty image prompt")
            print(f"Image prompt: {image_prompt}")
            return image_prompt
        except Exception as e:
            last_err = e
            print(f"build_image_prompt_with_groq attempt {attempt + 1} failed ({e}); retrying...")

    print(f"Groq image-prompt generation failed after retries ({last_err}); using fallback prompt.")
    return f"A simple illustration representing: {fact_text}. {style}."


def generate_background_image(
    fact_text: str,
    out_path: str,
    width: int = 1024,
    height: int = 1280,
) -> str:
    """
    Generates a background image matching the fact, via FLUX.1-schnell on
    Hugging Face's Inference API -- same model/provider pattern as
    coffee.py, but the prompt is derived from the fact's content rather
    than a fixed subject pool. width/height default to a 4:5 ratio
    (wider than pure 9:16) to leave headroom for any pan/zoom motion
    applied downstream, same reasoning as coffee.py's aspect adjustment.
    """
    from huggingface_hub import InferenceClient

    image_prompt = build_image_prompt_with_groq(fact_text)

    client = InferenceClient(token=HF_TOKEN)
    client.headers["x-use-cache"] = "0"
    model_id = "black-forest-labs/FLUX.1-schnell"

    last_err = None
    for attempt in range(3):
        try:
            image = client.text_to_image(
                prompt=image_prompt,
                model=model_id,
                width=width,
                height=height,
            )
            image.save(out_path)
            print(f"Background image saved to {out_path}")
            return out_path
        except Exception as e:
            last_err = e
            print(f"generate_background_image attempt {attempt + 1} failed ({e}); retrying...")

    raise RuntimeError(f"Background image generation failed after retries: {last_err}")


# ---------------------------------------------------------------------------
# 4. Text-to-speech narration via edge-tts
# ---------------------------------------------------------------------------

TTS_VOICES = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
]


async def _generate_tts_async(text: str, voice: str, out_path: str) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def generate_narration(text: str, out_path: str, voice: str = None) -> str:
    voice = voice or random.choice(TTS_VOICES)
    print(f"Generating narration with voice: {voice}")

    last_err = None
    for attempt in range(3):
        try:
            asyncio.run(_generate_tts_async(text, voice, out_path))
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"Narration saved to {out_path}")
                return out_path
            raise RuntimeError("edge-tts produced an empty file")
        except Exception as e:
            last_err = e
            print(f"generate_narration attempt {attempt + 1} failed ({e}); retrying...")

    raise RuntimeError(f"edge-tts failed after retries: {last_err}")


def get_audio_duration(audio_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


# ---------------------------------------------------------------------------
# 5. Burned-in captions via ffmpeg drawtext -- centered, large, short-form
# ---------------------------------------------------------------------------

def _escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", "\\\\\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
    )


def _wrap_text(text: str, width_chars: int = 18) -> str:
    wrapped = textwrap.fill(text, width=width_chars)
    return wrapped.replace("\n", "\\n")


def build_caption_filter(
    text: str,
    font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    font_size: int = 72,
    box_color: str = "black@0.55",
) -> str:
    safe_text = _escape_drawtext(_wrap_text(text))
    return (
        f"drawtext=fontfile={font_path}:text='{safe_text}':"
        f"fontsize={font_size}:fontcolor=white:"
        f"box=1:boxcolor={box_color}:boxborderw=30:"
        f"x=(w-text_w)/2:y=(h-text_h)/2:line_spacing=16"
    )


def render_caption_video(
    background_path: str,
    caption_text: str,
    output_path: str,
    duration: float,
    fps: int = 30,
    out_w: int = 1080,
    out_h: int = 1920,
) -> str:
    caption_filter = build_caption_filter(caption_text)

    vf = (
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h},"
        f"boxblur=8:4,"
        f"{caption_filter},"
        f"fade=t=in:st=0:d=0.4,fade=t=out:st={max(duration - 0.4, 0)}:d=0.4"
    )

    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", background_path,
        "-vf", vf,
        "-t", str(duration),
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg caption render failed:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg exited with code {result.returncode}")

    print(f"Caption video rendered at {output_path}")
    return output_path


def mux_narration_with_video(
    video_path: str,
    narration_path: str,
    music_path: str,
    output_path: str,
    duration: float,
    music_volume: float = 0.15,
    narration_volume: float = 1.0,
) -> str:
    fade_start = max(duration - 1, 0)
    filter_complex = (
        f"[1:a]volume={narration_volume}[narr];"
        f"[2:a]atrim=0:{duration},afade=t=out:st={fade_start}:d=1,volume={music_volume}[music];"
        f"[narr][music]amix=inputs=2:duration=first:dropout_transition=1[aout]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", narration_path,
        "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg narration mux failed:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg exited with code {result.returncode}")

    print(f"Final narrated video created at {output_path}")
    return output_path
