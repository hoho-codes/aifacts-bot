"""
facts.py

A "random facts" video pipeline: fetch a raw fact, polish it with Groq 
into a punchy short-form script, generate a matching background image 
via Hugging Face's FLUX, synthesize narration via edge-tts (free, no 
API key), and burn in captions via ffmpeg's drawtext filter.

Requires: pip install edge-tts requests huggingface_hub pillow
ffmpeg must be available on PATH (already true on ubuntu-latest runners).
"""

import asyncio
import os
import random
import subprocess
import textwrap
import requests
import glob

FACTS_API_URL = "https://uselessfacts.jsph.pl/api/v2/facts/random?language=en"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

YT_CLIENT_ID = os.environ["YT_CLIENT_ID"]
YT_CLIENT_SECRET = os.environ["YT_CLIENT_SECRET"]
YT_REFRESH_TOKEN = os.environ["YT_REFRESH_TOKEN"]
YT_PRIVACY_STATUS = os.environ.get("YT_PRIVACY_STATUS", "unlisted")

IMAGE_FILENAME = "assets/generated_image.png"
AUDIO_FILENAME = "assets/generated_audio.mp3"
CAP_VIDEO_FILENAME = "assets/captioned_video.mp4"
VIDEO_FILENAME = "assets/generated_video.mp4"

YT_TITLE_MAX_LEN = 100  # YouTube's hard limit on video titles
SHORTS_TAG = " #Shorts"

FALLBACK_FACTS = [
    "Honey never spoils. Archaeologists have found 3,000-year-old honey in Egyptian tombs that's still edible.",
    "Octopuses have three hearts, and two of them stop beating when they swim.",
    "Bananas are berries, but strawberries aren't.",
    "A day on Venus is longer than a year on Venus.",
    "Wombat poop is cube-shaped.",
]

# IMAGE_STYLE_MODIFIERS = [
#     "warm cinematic photography style",
#     "soft editorial illustration style",
#     "clean minimalist digital art style",
#     "vintage textbook illustration style",
#     "moody atmospheric photography",
#     "bright flat-design vector illustration",
# ]

IMAGE_STYLE_MODIFIERS = [
    "hyper-realistic photography, 8k, sharp focus, natural lighting",
    "cinematic realistic photography, shallow depth of field, dramatic lighting",
    "photorealistic macro photography, extreme detail, professional lighting",
    "realistic documentary-style photography, natural color grading",
    "hyper-detailed realistic photo, studio lighting, high dynamic range",
    "moody atmospheric realistic photography, volumetric light",
]


EFFECTS_WEIGHTED = [
    ("zoompan_in", 31.25),
    ("zoompan_out", 25),
    ("pan_horizontal", 25),
    ("color_drift", 18.75),
]

CAPTION_COLOR_PALETTES = [
    {"fontcolor": "0xFFEE00", "bordercolor": "0x000000@0.9"},   # bright yellow / black
    {"fontcolor": "0x00F0FF", "bordercolor": "0x001A2E@0.9"},   # electric cyan / deep navy
    {"fontcolor": "0xFF2E63", "bordercolor": "0x1A0010@0.9"},   # hot pink / near-black
    {"fontcolor": "0x39FF14", "bordercolor": "0x0A1A00@0.9"},   # neon green / dark
    {"fontcolor": "0xFFFFFF", "bordercolor": "0xFF6B00@0.9"},   # white text / bold orange border
    {"fontcolor": "0xFF9F1C", "bordercolor": "0x1A0F00@0.9"},   # vivid orange / near-black
    {"fontcolor": "0xB026FF", "bordercolor": "0x0F001A@0.9"},   # electric purple / near-black
    {"fontcolor": "0x00FFC2", "bordercolor": "0x001A15@0.9"},   # aqua/teal / dark teal
    {"fontcolor": "0xFF3131", "bordercolor": "0x1A0000@0.9"},   # bold red / near-black
    {"fontcolor": "0xFFFFFF", "bordercolor": "0x2E1A47@0.9"},   # white text / deep violet border
    {"fontcolor": "0xFDF200", "bordercolor": "0xFF00A0@0.9"},   # lemon yellow / hot magenta border
    {"fontcolor": "0x7CFC00", "bordercolor": "0x0A1A00@0.9"},   # lawn green / dark
]

def weighted_choice(pairs):
    items, weights = zip(*pairs)
    return random.choices(items, weights=weights, k=1)[0]

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


def fetch_random_facts(n: int = 5) -> list[str]:
    """
    Fetches n raw facts from the uselessfacts API (one request per fact --
    the API has no bulk endpoint). Each slot gets its own 3-attempt retry
    loop, same pattern as fetch_random_fact(), and skips duplicates so the
    batch given to Groq is actually n distinct options. If too few facts
    come back, tops up with FALLBACK_FACTS so selection still has enough
    to choose from.
    """
    facts = []
    seen = set()

    for i in range(n):
        last_err = None
        for attempt in range(3):
            try:
                res = requests.get(FACTS_API_URL, timeout=15)
                res.raise_for_status()
                data = res.json()
                fact = data.get("text", "").strip()
                if not fact:
                    raise ValueError("Empty fact text in response")
                if fact in seen:
                    raise ValueError("Duplicate fact, retrying for a fresh one")
                seen.add(fact)
                facts.append(fact)
                print(f"Fetched candidate {i + 1}/{n}: {fact}")
                break
            except Exception as e:
                last_err = e
                print(f"fetch_random_facts candidate {i + 1} attempt {attempt + 1} failed ({e}); retrying...")
        else:
            print(f"Failed to fetch candidate {i + 1}/{n} after retries ({last_err}); skipping.")

    if len(facts) < 2:
        print("Too few candidates fetched; topping up with fallback facts.")
        for f in FALLBACK_FACTS:
            if f not in seen:
                facts.append(f)
                seen.add(f)
            if len(facts) >= max(n, 2):
                break

    return facts


def pick_best_fact_with_groq(facts: list[str]) -> str:
    """
    Given a batch of raw candidate facts, asks Groq to pick the single
    most surprising / disbelief-inducing one to build this video around --
    same "surprising over dry trivia" framing as polish_fact_with_groq's
    system instruction, but used here as a selection step over several
    options rather than a rewrite of one. Falls back to random.choice
    if Groq is unavailable or its reply can't be parsed into a valid index.
    """
    if len(facts) <= 1:
        return facts[0] if facts else random.choice(FALLBACK_FACTS)

    if not GROQ_API_KEY:
        print("No GROQ_API_KEY set; picking a random candidate instead of ranking.")
        return random.choice(facts)

    numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))

    system_instruction = (
        "You are choosing which trivia fact to turn into a short-form video "
        "for a channel built on surprising, disbelief-inducing facts -- not "
        "dry trivia. You will be given a numbered list of candidate facts. "
        "Pick the ONE most surprising, counterintuitive, or shocking fact -- "
        "the one that would make someone stop scrolling and think 'wait, "
        "really?'. Favor facts with an angle involving danger, money, the "
        "human body, crime, or a common misconception over facts that are "
        "merely mildly interesting. "
        "Return ONLY the number of your chosen fact, nothing else."
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
                        {"role": "user", "content": numbered},
                    ],
                    "max_tokens": 300,  # generous margin -- reasoning overhead eats the budget before the number comes out, same lesson as generate_youtube_title
                    "temperature": 0.7,
                    "reasoning_effort": "low",
                },
                timeout=30,
            )
            res.raise_for_status()
            choice = res.json()["choices"][0]
            raw = choice["message"]["content"].strip()
            if choice.get("finish_reason") == "length":
                print(f"Warning: fact-selection hit the token limit (finish_reason=length): {raw!r}")

            digits = "".join(ch for ch in raw if ch.isdigit())
            if not digits:
                raise ValueError(f"No number found in Groq response: {raw!r}")
            choice_idx = int(digits) - 1
            if not (0 <= choice_idx < len(facts)):
                raise ValueError(f"Groq picked an out-of-range index: {raw!r}")

            chosen = facts[choice_idx]
            print(f"Groq picked candidate #{choice_idx + 1}: {chosen}")
            return chosen
        except Exception as e:
            last_err = e
            print(f"pick_best_fact_with_groq attempt {attempt + 1} failed ({e}); retrying...")

    print(f"Groq fact selection failed after retries ({last_err}); picking a random candidate instead.")
    return random.choice(facts)

# ---------------------------------------------------------------------------
# 2. Polish the raw fact into a punchy short-form script via Groq
# ---------------------------------------------------------------------------

def polish_fact_with_groq(raw_fact: str) -> str:
    if not GROQ_API_KEY:
        print("No GROQ_API_KEY set; using raw fact unmodified.")
        return raw_fact

    system_instruction = (
        "You rewrite trivia facts into short-form video narration scripts "
        "for a channel built on surprising, disbelief-inducing facts -- not "
        "dry trivia. Reframe the fact to emphasize what's counterintuitive, "
        "shocking, or contrary to common assumption, rather than stating it "
        "neutrally. If the fact has an angle involving danger, money, the "
        "human body, crime, or a common misconception, lead with that angle. "
        "Open the script with a hook that creates genuine doubt or surprise "
        "('You'd never guess...', 'Most people have no idea...', 'This "
        "sounds made up, but...') rather than a plain statement of the fact. "
        "Aim for 30 to 40 words, two to three sentences, spoken-language "
        "tone, no hashtags, no emojis, no quotation marks. "
        "Return ONLY the rewritten script, nothing else."
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
                    "max_tokens": 300,
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


# def get_fact_script() -> str:
#     """Convenience wrapper: fetch a raw fact, then polish it."""
#     raw_fact = fetch_random_fact()
#     return polish_fact_with_groq(raw_fact)

def get_fact_script(n_candidates: int = 5) -> str:
    """
    Convenience wrapper: fetch n candidate facts, let Groq pick the
    most surprising one, then polish it into narration.
    """
    candidates = fetch_random_facts(n_candidates)
    best_fact = pick_best_fact_with_groq(candidates)
    return polish_fact_with_groq(best_fact)

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
        "generator. Describe a single clear, photorealistic scene that represents "
        "the fact -- as if shot by a professional photographer, not illustrated "
        "or stylized. No text, no words, no diagrams, just a real scene or object "
        "with realistic lighting, texture, and detail. "
        f"Render it in this style: {style}. "
        "Compose the scene vertically, with the subject centered and filling "
        "the frame. The final image must be a full-bleed photo with no borders, "
        "frames, white margins, or visible photo-paper edges. "
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


def generate_caption_with_groq(fact_text: str) -> tuple[str, str]:
    """
    Generates a short two-part on-screen caption from the fact: a hook line
    (phrased as a direct, provocative question rather than a generic
    "Did You Know?!") and a punchy answer line. Used only for the
    burned-in caption -- narration still uses the full polished
    fact_text from get_fact_script().
    """
    fallback_hook = random.choice([
        "Wait, WHAT?!",
        "You've Been Wrong",
        "Nobody Knows This",
        "This Sounds Fake",
    ])
    fallback_answer = textwrap.shorten(fact_text.split(".")[0], width=60, placeholder="...")

    if not GROQ_API_KEY:
        print("No GROQ_API_KEY set; using fallback caption.")
        return fallback_hook, fallback_answer

    system_instruction = (
        "You write short on-screen captions for a viral trivia Shorts video. "
        "Given a fact script, return TWO lines separated by '|||':\n"
        "The first is a short, provocative HOOK (4-7 words) that creates "
        "genuine disbelief or curiosity -- phrase it as a direct question "
        "the viewer needs answered, or a bold claim that sounds almost "
        "unbelievable. Avoid generic openers like 'Did You Know' -- instead "
        "use patterns like 'Which everyday thing actually...', 'You've been "
        "wrong about...', 'This is banned in...', 'Nobody tells you this "
        "about...'. Make it specific to THIS fact, not generic.\n"
        "The second is the punchiest single detail from the fact, rewritten "
        "as a short, catchy phrase (under 8 words, no full sentence needed) "
        "that delivers the payoff.\n"
        "No hashtags, no emojis, no quotation marks. "
        "Return ONLY 'hook|||answer', nothing else."
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
            raw = res.json()["choices"][0]["message"]["content"].strip().strip('"')
            if "|||" not in raw:
                raise ValueError(f"Missing '|||' separator in caption response: {raw!r}")
            hook, answer = raw.split("|||", 1)
            hook, answer = hook.strip(), answer.strip()
            if not hook or not answer:
                raise ValueError("Groq returned an empty hook or answer")
            print(f"Caption -> hook: {hook!r} | answer: {answer!r}")
            return hook, answer
        except Exception as e:
            last_err = e
            print(f"generate_caption_with_groq attempt {attempt + 1} failed ({e}); retrying...")

    print(f"Groq caption generation failed after retries ({last_err}); using fallback caption.")
    return fallback_hook, fallback_answer
    

def generate_background_image(
    fact_text: str,
    out_path: str,
    width: int = 1152,
    height: int = 1440,
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
    "en-GB-RyanNeural",
]


async def _generate_tts_async(text: str, voice: str, out_path: str) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate="-10%")
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
    """
    Escapes characters that break ffmpeg's drawtext filter syntax.
    Applied BEFORE line-wrapping's \\n markers are inserted, so the
    backslash-doubling step here never touches the newline escapes.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\u2019")
    text = text.replace(",", "\\,")
    text = text.replace("%", "\\%")
    return text


# def _wrap_text(text: str, width_chars: int = 18) -> str:
#     wrapped = textwrap.fill(text, width=width_chars)
#     return wrapped  # keep literal \n newlines here -- don't convert yet


# def build_caption_filter(
#     text: str,
#     caption_file_path: str,
#     font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
#     out_w: int = 1080,
# ) -> str:
#     escaped = _escape_drawtext(text)

#     word_count = len(text.split())
#     if word_count <= 15:
#         font_size = 88
#     elif word_count <= 30:
#         font_size = 68
#     elif word_count <= 50:
#         font_size = 54
#     else:
#         font_size = 44

#     avg_char_width_px = font_size * 0.58
#     usable_width_px = out_w - 80
#     wrap_width_chars = max(int(usable_width_px / avg_char_width_px), 8)

#     wrapped = textwrap.fill(escaped, width=wrap_width_chars)
#     num_lines = wrapped.count("\n") + 1

#     with open(caption_file_path, "w", encoding="utf-8") as f:
#         f.write(wrapped)

#     line_spacing = 16
#     bottom_padding = 60

#     palette = random.choice(CAPTION_COLOR_PALETTES)

#     return (
#         f"drawtext=fontfile={font_path}:textfile={caption_file_path}:"
#         f"fontsize={font_size}:fontcolor={palette['fontcolor']}:"
#         f"borderw=3:bordercolor={palette['bordercolor']}:"
#         f"shadowcolor=black@0.9:shadowx=3:shadowy=3:"
#         f"x=(w-text_w)/2:y=h-text_h-{bottom_padding}:line_spacing={line_spacing}"
#     )

def _build_single_caption_filter(
    text: str,
    caption_file_path: str,
    font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    out_w: int = 1080,
    enable_expr: str = None,
    position: str = "bottom",
    top_padding: int = 100,
    bottom_padding: int = 60,
    palette: dict = None,
) -> str:
    escaped = _escape_drawtext(text)

    word_count = len(text.split())
    if word_count <= 5:
        font_size = 100
    elif word_count <= 10:
        font_size = 80
    else:
        font_size = 64

    avg_char_width_px = font_size * 0.58
    usable_width_px = out_w - 80
    wrap_width_chars = max(int(usable_width_px / avg_char_width_px), 8)

    wrapped = textwrap.fill(escaped, width=wrap_width_chars)
    with open(caption_file_path, "w", encoding="utf-8") as f:
        f.write(wrapped)

    palette = palette or random.choice(CAPTION_COLOR_PALETTES)
    enable_part = f":enable='{enable_expr}'" if enable_expr else ""
    y_expr = f"{top_padding}" if position == "top" else f"h-text_h-{bottom_padding}"

    return (
        f"drawtext=fontfile={font_path}:textfile={caption_file_path}:"
        f"fontsize={font_size}:fontcolor={palette['fontcolor']}:"
        f"borderw=3:bordercolor={palette['bordercolor']}:"
        f"shadowcolor=black@0.9:shadowx=3:shadowy=3:"
        f"text_align=C:"
        f"x=(w-text_w)/2:y={y_expr}:line_spacing=16{enable_part}"
    )


def build_two_part_caption_filter(
    hook_text: str,
    answer_text: str,
    duration: float,
    out_w: int = 1080,
    answer_delay_fraction: float = 0.15,
    min_answer_delay: float = 1.5,
    max_answer_delay: float = 4.0,
) -> tuple[str, dict]:
    answer_delay = duration * answer_delay_fraction
    answer_delay = max(min_answer_delay, min(answer_delay, max_answer_delay))
    answer_delay = min(answer_delay, max(duration - 0.3, 0))

    palette = random.choice(CAPTION_COLOR_PALETTES)

    hook_filter = _build_single_caption_filter(
        hook_text, "assets/caption_hook.txt", out_w=out_w,
        enable_expr=None,
        position="top", top_padding=280,
        palette=palette,
    )
    answer_filter = _build_single_caption_filter(
        answer_text, "assets/caption_answer.txt", out_w=out_w,
        enable_expr=f"gte(t,{answer_delay})",
        position="bottom", bottom_padding=220,
        palette=palette,
    )
    return f"{hook_filter},{answer_filter}", palette
    

def build_motion_filter(effect_name: str, duration: float, fps: int = 30) -> str:
    total_frames = max(int(duration * fps), 1)
    baseline_frames = 5 * fps

    zoom_increment = 0.0007 * (baseline_frames / total_frames)
    pan_speed = 40  # px/sec
    breathing_cycle_frames = 10
    color_drift_increment = 0.0006 * (baseline_frames / total_frames)

    filters = {
        "zoompan_in": f"zoompan=z='min(zoom+{zoom_increment},1.3)':d={total_frames}:s=1080x1920:fps={fps}",
        "zoompan_out": f"zoompan=z='if(eq(on,1),1.3,max(1.001,zoom-{zoom_increment}))':d={total_frames}:s=1080x1920:fps={fps}",
        "pan_horizontal": f"crop=1080:1920:x='min(t*{pan_speed},iw-1080)':y=0",
        "breathing_zoom": f"zoompan=z='1.1+0.05*sin(on/{breathing_cycle_frames})':d={total_frames}:s=1080x1920:fps={fps}",
        "color_drift": f"eq=saturation=1.1,zoompan=z='min(zoom+{color_drift_increment},1.25)':d={total_frames}:s=1080x1920:fps={fps}",
    }
    return filters.get(effect_name, filters["zoompan_in"])


def render_caption_video(
    background_path: str,
    hook_text: str,
    answer_text: str,
    output_path: str,
    duration: float,
    fps: int = 30,
    out_w: int = 1080,
    out_h: int = 1920,
) -> tuple[str, dict]:
    caption_filter, palette = build_two_part_caption_filter(hook_text, answer_text, duration, out_w=out_w)

    effect = weighted_choice(EFFECTS_WEIGHTED)
    print(f"Selected motion effect: {effect}")
    motion_filter = build_motion_filter(effect, duration, fps)

    if effect == "pan_horizontal":
        scale_crop = f"scale=1600:1920:force_original_aspect_ratio=increase"
    else:
        scale_crop = f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h}"

    vf = (
        f"{scale_crop},"
        f"{motion_filter},"
        f"boxblur=3:2,"
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

    print(f"Caption video rendered at {output_path} with effect '{effect}'")
    return output_path, palette


def build_outro_filter(
    text: str = "Subscribe for More",
    font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    out_w: int = 1080,
    palette: dict = None,
) -> str:
    """
    Simple centered outro text filter -- large, bold, high-contrast,
    no fancy styling since it only needs to read clearly for ~1.5s.
    Uses the same color palette as the captions if provided, for
    visual consistency across the whole video.
    """
    palette = palette or random.choice(CAPTION_COLOR_PALETTES)
    escaped = _escape_drawtext(text)
    return (
        f"drawtext=fontfile={font_path}:text='{escaped}':"
        f"fontsize=76:fontcolor={palette['fontcolor']}:"
        f"borderw=4:bordercolor={palette['bordercolor']}:"
        f"text_align=C:"
        f"x=(w-text_w)/2:y=(h-text_h)/2"
    )


def render_outro_clip(
    background_path: str,
    output_path: str,
    duration: float = 1.5,
    fps: int = 30,
    out_w: int = 1080,
    out_h: int = 1920,
    palette: dict = None,
) -> str:
    outro_filter = build_outro_filter(out_w=out_w, palette=palette)

    vf = (
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h},"
        f"boxblur=8:4,"
        f"{outro_filter},"
        f"fade=t=in:st=0:d=0.3"
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
        print(f"ffmpeg outro render failed:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg exited with code {result.returncode}")

    print(f"Outro clip rendered at {output_path}")
    return output_path
    

def concat_video_with_outro(main_video_path: str, outro_path: str, output_path: str) -> str:
    """
    Concatenates the main captioned video with the outro clip using
    ffmpeg's concat demuxer. Both inputs must already share the same
    codec/resolution/fps, which they do here (both produced by the same
    libx264/1080x1920/30fps pipeline).
    """
    concat_list_path = "assets/concat_list.txt"
    with open(concat_list_path, "w") as f:
        f.write(f"file '{os.path.abspath(main_video_path)}'\n")
        f.write(f"file '{os.path.abspath(outro_path)}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg concat failed:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg exited with code {result.returncode}")

    print(f"Video with outro created at {output_path}")
    return output_path


def mux_narration_with_video(
    video_path: str,
    narration_path: str,
    output_path: str,
    duration: float,
    music_volume: float = 0.15,
    narration_volume: float = 1.0,
):
    music_files = glob.glob("assets/music/*.mp3")
    print(f"Found {len(music_files)} music file(s) in assets/music/")
    random.shuffle(music_files)

    fade_start = max(duration - 1, 0)

    for music_path in music_files:
        print(f"Trying background music: {music_path}")
        filter_complex = (
            f"[1:a]apad,volume={narration_volume}[narr];"
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
            "-t", str(duration),
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Final narrated video created at {output_path} with music: {music_path}")
            return output_path
        else:
            print(f"Track failed ({music_path}), trying next if available:\n{result.stderr[-500:]}")

    print("No usable music track found — muxing narration only, no music.")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", narration_path,
        "-filter_complex", "[1:a]apad[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-t", str(duration),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg narration-only mux failed:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg exited with code {result.returncode}")

    print(f"Final narrated video created at {output_path} (no music)")
    return output_path
    

def yt_refresh_access_token() -> str:
    res = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": YT_CLIENT_ID,
            "client_secret": YT_CLIENT_SECRET,
            "refresh_token": YT_REFRESH_TOKEN,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    if not res.ok:
        print(f"YouTube token refresh error body: {res.text}")
    res.raise_for_status()
    return res.json()["access_token"]
 
 
def publish_to_youtube(video_path: str, title: str, description: str, tags=None):
    try:
        access_token = yt_refresh_access_token()

        metadata = {
            "snippet": {
                "title": title[:100],
                "description": description,
                "tags": tags or ["facts", "shorts", "didyouknow"],
                "categoryId": "27",
            },
            "status": {
                "privacyStatus": YT_PRIVACY_STATUS,
                "selfDeclaredMadeForKids": False,
                "containsSyntheticMedia": true,
            },
        }

        init_res = requests.post(
            "https://www.googleapis.com/upload/youtube/v3/videos"
            "?uploadType=resumable&part=snippet,status",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Type": "video/mp4",
            },
            json=metadata,
            timeout=30,
        )
        if not init_res.ok:
            print(f"YouTube init error body: {init_res.text}")  # <-- add this
        init_res.raise_for_status()
        upload_url = init_res.headers["Location"]

        with open(video_path, "rb") as f:
            video_bytes = f.read()

        upload_res = requests.put(
            upload_url,
            headers={"Content-Type": "video/mp4"},
            data=video_bytes,
            timeout=180,
        )
        if not upload_res.ok:
            print(f"YouTube upload error body: {upload_res.text}")  # <-- and this
        upload_res.raise_for_status()
        return upload_res
    except Exception as e:
        print(f"YouTube error: {e}")
        return None


# ---------------------------------------------------------------------------
# YouTube title & description generation via Groq
# ---------------------------------------------------------------------------

def generate_youtube_title(fact_text: str) -> str:
    """
    Generates a punchy, clickable YouTube title from the fact script,
    then appends ' #Shorts'. The Groq instruction reserves room for the
    tag up front so the combined result respects YouTube's 100-char
    title limit; a hard truncation afterward is a final safety net in
    case Groq ignores the length instruction.
    """
    reserved = len(SHORTS_TAG)
    max_title_len = YT_TITLE_MAX_LEN - reserved

    if not GROQ_API_KEY:
        print("No GROQ_API_KEY set; using a truncated fact as title.")
        base_title = fact_text[:max_title_len].rstrip()
        return base_title + SHORTS_TAG

    system_instruction = (
        "You write short, clickable YouTube titles for a trivia facts "
        "channel built on disbelief and surprise. Given a fact script, write "
        "ONE punchy title using patterns like 'You've Been Wrong About...', "
        "'Nobody Tells You This About...', 'The Reason [X] Will Shock You', "
        "or a direct provocative question. Avoid neutral, generic phrasing -- "
        "make the viewer need to know the answer. "
        f"HARD LIMIT: {max_title_len} characters, no exceptions. "
        "No hashtags, no emojis, no quotation marks. "
        "Return ONLY the title text, nothing else."
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
                    "max_tokens": 400,   # was 150 -- still not enough margin over reasoning overhead
                    "temperature": 0.9,
                    "reasoning_effort": "low",
                },
                timeout=30,
            )
            res.raise_for_status()
            choice = res.json()["choices"][0]
            title = choice["message"]["content"].strip().strip('"')
            if choice.get("finish_reason") == "length":
                print(f"Warning: title generation hit the token limit (finish_reason=length): {title!r}")
            if not title:
                raise ValueError("Groq returned an empty title")

            # Hard safety net regardless of what Groq actually returned
            title = title[:max_title_len].rstrip()
            final_title = title + SHORTS_TAG
            print(f"Generated title: {final_title}")
            return final_title
        except Exception as e:
            last_err = e
            print(f"generate_youtube_title attempt {attempt + 1} failed ({e}); retrying...")

    print(f"Groq title generation failed after retries ({last_err}); using fallback title.")
    base_title = fact_text[:max_title_len].rstrip()
    return base_title + SHORTS_TAG


def generate_youtube_description(fact_text: str) -> str:
    fallback_description = f"{fact_text}\n\n#facts #shorts #didyouknow"

    if not GROQ_API_KEY:
        print("No GROQ_API_KEY set; using fallback description.")
        return fallback_description

    system_instruction = (
        "You write YouTube Shorts descriptions for a trivia facts channel. "
        "Given a fact script, write a short, well-formatted description: "
        "one or two sentences expanding slightly on the fact, then a blank "
        "line, then 4-6 relevant hashtags on their own line. "
        "No emojis in the hashtags. Keep the whole thing under 400 characters. "
        "Return ONLY the description text, nothing else."
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
                    "max_tokens": 500,   # was 200 -- reasoning overhead was eating most of the budget
                    "temperature": 0.9,
                    "reasoning_effort": "low",
                },
                timeout=30,
            )
            res.raise_for_status()
            choice = res.json()["choices"][0]
            description = choice["message"]["content"].strip().strip('"')
            if choice.get("finish_reason") == "length":
                print(f"Warning: description generation hit the token limit (finish_reason=length): {description!r}")
            if not description or len(description) < 30:
                raise ValueError(f"Groq returned an empty or suspiciously short description: {description!r}")
            print(f"Generated description: {description}")
            return description
        except Exception as e:
            last_err = e
            print(f"generate_youtube_description attempt {attempt + 1} failed ({e}); retrying...")

    print(f"Groq description generation failed after retries ({last_err}); using fallback description.")
    return fallback_description


def commit_video():
    """
    Commits and pushes the final video to the repo, same pattern as
    coffee.py's commit_image(). GitHub Actions runners are ephemeral --
    without this, the generated file disappears the moment the job ends.
    Requires `permissions: contents: write` in the workflow YAML.
    """
    print("Committing video to repo...")
    subprocess.run(["git", "config", "user.name", "facts-bot"])
    subprocess.run(["git", "config", "user.email", "facts-bot@users.noreply.github.com"])
    subprocess.run(["git", "add", VIDEO_FILENAME])
    commit_result = subprocess.run(["git", "commit", "-m", "Daily fact video"], capture_output=True, text=True)
    if commit_result.returncode != 0:
        print(f"Nothing to commit or commit failed:\n{commit_result.stderr}")
        return
    push_result = subprocess.run(["git", "push"], capture_output=True, text=True)
    if push_result.returncode != 0:
        print(f"Video push failed:\n{push_result.stderr}")
    else:
        print("Video committed and pushed successfully.")


def main():
    fact = get_fact_script()
    hook, answer = generate_caption_with_groq(fact)
    generate_background_image(fact, IMAGE_FILENAME)
    narration_path = generate_narration(fact, AUDIO_FILENAME)
    duration = get_audio_duration(narration_path)
    _, palette = render_caption_video(IMAGE_FILENAME, hook, answer, CAP_VIDEO_FILENAME, duration)

    outro_path = "assets/outro_clip.mp4"
    render_outro_clip(IMAGE_FILENAME, outro_path, palette=palette)

    combined_path = "assets/combined_with_outro.mp4"
    concat_video_with_outro(CAP_VIDEO_FILENAME, outro_path, combined_path)
    os.replace(combined_path, CAP_VIDEO_FILENAME)

    mux_narration_with_video(CAP_VIDEO_FILENAME, narration_path, VIDEO_FILENAME, duration + 1.5)

    commit_video()

    title = generate_youtube_title(fact)
    description = generate_youtube_description(fact)

    res = publish_to_youtube(VIDEO_FILENAME, title, description, tags=["facts", "shorts", "didyouknow"])

    if res is not None and res.ok:
        print(f"Uploaded Short: {res.json().get('id')}")
    else:
        print("YouTube upload failed; see error above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
