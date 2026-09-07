# Facts Bot

Daily automated pipeline: fetches a random fact → rewrites it with Groq →
generates a matching image (FLUX) → narrates it (edge-tts, free) → adds
motion + burned-in captions + music (ffmpeg) → generates a title/description
(Groq) → uploads to YouTube Shorts. Runs via GitHub Actions, ~$0/month.

## Requirements
pip install requests huggingface_hub edge-tts

`ffmpeg`/`ffprobe` installed explicitly in the workflow.

## Secrets
| Secret | Purpose |
|---|---|
| `GROQ_API_KEY` | Script, prompts, title, description |
| `HF_TOKEN` | FLUX image generation |
| `YT_CLIENT_ID` / `YT_CLIENT_SECRET` | Google OAuth app |
| `YT_REFRESH_TOKEN` | Per-channel — generate via `get_youtube_refresh_token.py` |
| `YT_PRIVACY_STATUS` | `public` / `unlisted` / `private` |

## Local assets
`assets/music/*.mp3` — random background track each run; falls back to no music if missing.

## Watch out for
- Fact API and edge-tts are free but unofficial (no SLA).
- Hugging Face free tier caps ~$0.10/month — shared across bots if using the same account.
- Repo grows over time from daily video commits.

## Manual run
Actions tab → workflow → **Run workflow**.
