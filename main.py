import json
import os
import re
import time
import logging
from pathlib import Path
from difflib import SequenceMatcher

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, field_validator

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Load ENV
# ─────────────────────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────────
# Groq Client
# ─────────────────────────────────────────────────────────────
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL_NAME = "llama-3.1-8b-instant"

# ─────────────────────────────────────────────────────────────
# Load Catalog
# ─────────────────────────────────────────────────────────────
CATALOG_PATH = Path(__file__).parent / "catalog.json"

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    CATALOG = json.load(f)

CATALOG_BY_URL = {
    p["url"]: p for p in CATALOG
}

VALID_URLS = set(CATALOG_BY_URL.keys())

# Maximum catalog items to include in prompt
MAX_CATALOG_IN_PROMPT = 20

# Maximum conversation turns to include
MAX_HISTORY_TURNS = 4

# ─────────────────────────────────────────────────────────────
# Keyword-based catalog pre-filter
# ─────────────────────────────────────────────────────────────
def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase keywords from text."""
    return set(re.findall(r'[a-z0-9#+.]+', text.lower()))


def _score_product(product: dict, keywords: set[str]) -> float:
    """Score a product's relevance to the query keywords."""
    searchable = " ".join([
        product.get("name", ""),
        product.get("description", ""),
        product.get("keys", ""),
        product.get("test_type", ""),
    ]).lower()

    product_keywords = _extract_keywords(searchable)
    score = 0.0

    for kw in keywords:
        # Exact match
        if kw in product_keywords:
            score += 2.0
        # Substring match (e.g. "java" in "javascript")
        elif any(kw in pk for pk in product_keywords):
            score += 1.0
        # Fuzzy match
        else:
            best = max(
                (SequenceMatcher(None, kw, pk).ratio()
                 for pk in product_keywords),
                default=0.0
            )
            if best > 0.7:
                score += best * 0.5

    return score


def filter_catalog(query: str) -> list[dict]:
    """Return the top-N most relevant catalog items for query."""
    keywords = _extract_keywords(query)

    if not keywords:
        return CATALOG[:MAX_CATALOG_IN_PROMPT]

    scored = [
        (p, _score_product(p, keywords))
        for p in CATALOG
    ]

    # Keep items with score > 0, sorted by relevance
    relevant = sorted(
        [(p, s) for p, s in scored if s > 0],
        key=lambda x: x[1],
        reverse=True
    )

    results = [p for p, _ in relevant[:MAX_CATALOG_IN_PROMPT]]

    # If very few matches, add some general items
    if len(results) < 5:
        for p in CATALOG:
            if p not in results:
                results.append(p)
            if len(results) >= 10:
                break

    return results


# ─────────────────────────────────────────────────────────────
# Convert catalog subset to compact prompt text
# ─────────────────────────────────────────────────────────────
def build_catalog_text(products: list[dict]) -> str:
    """Build a compact text representation of catalog items."""
    lines = []

    for p in products:
        # Compact format: name | type | URL (no descriptions)
        line = (
            f'- {p["name"]} '
            f'| {p.get("test_type", "")} '
            f'| {p["url"]}'
        )
        lines.append(line)

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """
You are an SHL assessment consultant.
Recommend 1-10 assessments from the catalog below.
Rules: Only use catalog URLs. Be concise. Return ONLY valid JSON.
Format:
{{"reply":"text","recommendations":[{{"name":"...","url":"...","test_type":"..."}}],"end_of_conversation":false}}

CATALOG:
{catalog}
"""

# ─────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):

        if v not in ["user", "assistant"]:
            raise ValueError(
                "role must be user or assistant"
            )

        return v


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):

        if not v:
            raise ValueError(
                "messages cannot be empty"
            )

        if v[-1].role != "user":
            raise ValueError(
                "last message must be user"
            )

        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ─────────────────────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────────────────────
def build_prompt(messages, catalog_text: str):
    """Build prompt with filtered catalog and limited history."""
    system = SYSTEM_PROMPT_TEMPLATE.format(
        catalog=catalog_text
    )

    # Limit conversation history to last N turns
    recent = messages[-(MAX_HISTORY_TURNS * 2):]

    parts = [
        system,
        "\nConversation:\n"
    ]

    for m in recent:
        role = (
            "User"
            if m.role == "user"
            else "Assistant"
        )
        parts.append(f"{role}: {m.content}")

    parts.append("\nAssistant:")

    return "\n".join(parts)

# ─────────────────────────────────────────────────────────────
# Validate Recommendations
# ─────────────────────────────────────────────────────────────
def validate_recommendations(recs):

    validated = []

    seen = set()

    for r in recs:

        url = r.get("url", "").strip()

        if url in VALID_URLS and url not in seen:

            seen.add(url)

            product = CATALOG_BY_URL[url]

            validated.append({
                "name": product["name"],
                "url": product["url"],
                "test_type": product.get(
                    "test_type",
                    ""
                )
            })

    return validated[:10]

# ─────────────────────────────────────────────────────────────
# Groq Response
# ─────────────────────────────────────────────────────────────
def generate_response(prompt, retries=3):
    """Call Groq with retry + exponential backoff for rate limits."""
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return ONLY valid JSON. "
                            "No markdown."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.2,
                max_tokens=512,
            )

            return completion.choices[0].message.content

        except Exception as e:
            error_str = str(e)
            if "rate_limit" in error_str or "413" in error_str:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    f"Rate limited, retrying in {wait}s "
                    f"(attempt {attempt + 1}/{retries})"
                )
                time.sleep(wait)
            else:
                raise

    raise Exception("Rate limit exceeded after retries")

# ─────────────────────────────────────────────────────────────
# Main Agent Logic
# ─────────────────────────────────────────────────────────────
def call_agent(messages):

    # Extract user query for catalog filtering
    user_query = messages[-1].content

    # Pre-filter catalog to relevant items
    filtered = filter_catalog(user_query)
    catalog_text = build_catalog_text(filtered)

    logger.info(
        f"Filtered catalog: {len(filtered)}/{len(CATALOG)} items"
    )

    prompt = build_prompt(messages, catalog_text)

    raw = generate_response(prompt)

    logger.info(f"Raw LLM response: {raw[:500]}")

    # Try extracting JSON
    json_match = re.search(
        r'\{.*\}',
        raw,
        re.DOTALL
    )

    json_str = (
        json_match.group(0)
        if json_match
        else raw
    )

    try:

        data = json.loads(json_str)

    except Exception as e:

        logger.error(
            f"JSON parse failed: {e}"
        )

        logger.error(raw)

        return ChatResponse(
            reply=(
                "Sorry, I had trouble "
                "processing that request."
            ),
            recommendations=[],
            end_of_conversation=False
        )

    reply = str(
        data.get("reply", "")
    ).strip()

    raw_recs = data.get(
        "recommendations",
        []
    )

    end = bool(
        data.get(
            "end_of_conversation",
            False
        )
    )

    validated = validate_recommendations(
        raw_recs
    )

    return ChatResponse(
        reply=reply,

        recommendations=[
            Recommendation(**r)
            for r in validated
        ],

        end_of_conversation=end,
    )

# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Health Endpoint
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():

    return {
        "status": "ok",
        "model": MODEL_NAME,
        "catalog_size": len(CATALOG),
    }

# ─────────────────────────────────────────────────────────────
# Chat Endpoint
# ─────────────────────────────────────────────────────────────
@app.post(
    "/chat",
    response_model=ChatResponse
)
async def chat(request: ChatRequest):

    try:

        start = time.time()

        result = call_agent(
            request.messages
        )

        elapsed = time.time() - start

        logger.info(
            f"Done in {elapsed:.2f}s"
        )

        return result

    except Exception as e:

        logger.error(
            f"Server error: {e}",
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ─────────────────────────────────────────────────────────────
# Run Local
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )