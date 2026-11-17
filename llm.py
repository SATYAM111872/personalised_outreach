import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import config as C

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")

# "-latest" model aliases are served on v1beta only. Set it explicitly: newer
# google-genai releases may default to v1, which returns 404 NOT_FOUND for the alias.
API_VERSION = "v1beta"


def _chat(model):
    # max_retries=1 turns off the SDK's own hidden retries (default 6, with
    # backoff): agent.call_llm does the retrying, within its own deadline.
    # Without this, one call could retry inside the SDK past that deadline and
    # keep a worker thread busy after the run had already given up on it.
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=API_KEY,
        temperature=0,
        api_version=API_VERSION,
        max_retries=1,
        timeout=C.LLM_REQUEST_TIMEOUT_S,
    )


llm = _chat("gemini-flash-lite-latest")
llm_fallback = _chat(C.LLM_FALLBACK_MODEL) if C.LLM_FALLBACK_MODEL else llm

# Stage 3 ranking. If embedding calls fail, ranking falls back to keyword overlap.
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY,
    api_version=API_VERSION,
)
