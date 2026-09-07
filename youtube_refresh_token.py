"""
One-off script to (re)generate a YouTube OAuth refresh token.
Run this locally whenever your existing refresh token expires
(every ~7 days while the OAuth consent screen is unverified/in Testing).

Requires: pip install requests
"""

import requests
import webbrowser
import urllib.parse

# --- Fill these in with your values from Google Cloud Console ---
CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"
REDIRECT_URI = "http://localhost:8080"  # must be added as an Authorized redirect URI in Cloud Console

SCOPES = "https://www.googleapis.com/auth/youtube.upload"


def build_authorize_url():
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "offline",   # required to get a refresh_token
        "prompt": "consent",        # forces a new refresh_token every time
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)


def exchange_code_for_tokens(code):
    res = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        },
    )
    res.raise_for_status()
    return res.json()


if __name__ == "__main__":
    url = build_authorize_url()
    print("Opening browser for Google authorization...")
    print(f"If it doesn't open automatically, visit:\n{url}\n")
    webbrowser.open(url)

    print("After approving, you'll be redirected to a localhost URL that won't load —")
    print("that's expected. Copy the 'code' parameter from that browser address bar.\n")

    code = input("Paste the code here: ").strip()

    tokens = exchange_code_for_tokens(code)
    print("\nSuccess! Update your GitHub secret with this value:\n")
    print(f"YT_REFRESH_TOKEN = {tokens['refresh_token']}")
    print(f"\n(access_token also issued, expires in {tokens.get('expires_in')} seconds — not needed, the script refreshes this automatically)")
