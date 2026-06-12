"""
gemini_client.py
----------------
PURPOSE:
  Shared Gemini client factory used by all modules that call the Gemini API.
  Centralises authentication so credentials are configured in one place only.

AUTH METHOD:
  Uses a GCP service account (same approach as ClaimSense's claude_client.py).
  The full service account JSON is stored as a single-line string in the
  GCP_SERVICE_ACCOUNT_JSON environment variable — not as a file path.

  To flatten your JSON file into a single line for .env, see the setup guide.
  PowerShell command: read the key file, replace newlines, copy to clipboard.
  Then paste: GCP_SERVICE_ACCOUNT_JSON={"type":"service_account",...}

ENVIRONMENT VARIABLES:
  GCP_SERVICE_ACCOUNT_JSON  full service account JSON string (required)
  GOOGLE_CLOUD_LOCATION     Vertex AI region (optional, defaults to us-central1)

USAGE:
  from gemini_client import get_client, GEMINI_MODEL

  client   = get_client()
  response = client.models.generate_content(
      model=GEMINI_MODEL, contents="your prompt"
  )

  # For embeddings:
  result = client.models.embed_content(
      model=EMBEDDING_MODEL,
      contents="your text",
      config={"task_type": "RETRIEVAL_DOCUMENT"},
  )

WHY A SHARED MODULE:
  metadata_generator.py, embedder.py, router.py, and engine.py all need
  a Gemini client. Without this module each would duplicate the auth logic
  and each would need updating if credentials change.
"""

import json
import os

from google import genai
from google.genai import types
from google.oauth2 import service_account

# Model names used across the project.
GEMINI_MODEL    = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = "models/text-embedding-004"


def get_client() -> genai.Client:
    """
    Build and return an authenticated Gemini client.

    Reads GCP_SERVICE_ACCOUNT_JSON from environment, parses it,
    creates OAuth2 credentials, and returns a Vertex AI Gemini client.

    Called fresh on each request — no module-level caching so that
    environment variable changes (e.g. in tests) are always picked up.

    Raises RuntimeError if GCP_SERVICE_ACCOUNT_JSON is not set.
    """
    raw_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not raw_json:
        raise RuntimeError(
            "GCP_SERVICE_ACCOUNT_JSON is not set. "
            "Add the full service account JSON as a single line to your .env file."
        )

    try:
        service_account_info = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"GCP_SERVICE_ACCOUNT_JSON is not valid JSON: {e}. "
            "Make sure the JSON was flattened to a single line with no newlines."
        ) from e

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    return genai.Client(
        vertexai=True,
        project=service_account_info["project_id"],
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        credentials=credentials,
    )
