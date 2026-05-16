# SHL Assessment Recommender — Approach Document

## Problem Decomposition

The task requires a conversational agent that takes hiring managers from vague intent to a grounded shortlist of SHL assessments. I decomposed it into four sub-problems:

1. **Catalog acquisition**: Scrape and structure the full SHL Individual Test Solutions catalog
2. **Retrieval strategy**: Decide how to make catalog knowledge available to the LLM
3. **Agent design**: Define when to clarify, recommend, refine, compare, or refuse
4. **API contract**: Expose a stateless FastAPI service with the exact required schema

---

## Design Choices

### Retrieval: Full catalog in system prompt (no vector DB)

I evaluated two approaches:

| Approach | Pros | Cons |
|---|---|---|
| Vector search (FAISS/Chroma) | Scales to 10k+ products | Extra infra, retrieval errors, latency |
| Full catalog in system prompt | Zero retrieval errors, simpler | Token cost, context window limit |

With ~120 products and Claude claude-sonnet-4-20250514's 200k token context, the entire catalog fits comfortably. Each product entry is ~3-4 lines (name, test type, description, URL) totaling ~32k tokens in the system prompt. This eliminates false negatives from retrieval — every product is always visible to the model.

**The critical advantage**: retrieval errors would cause the agent to miss relevant products, hurting Recall@10. With full context, the LLM can reason over the complete catalog on every turn.

### Agent Design: Structured JSON output

The LLM is instructed to always output a JSON object with three fields: `reply`, `recommendations`, and `end_of_conversation`. This schema-first approach means:

- No post-processing needed to extract structured data
- JSON parsing failures fall back gracefully (raw text becomes the reply)
- URL validation runs *after* LLM output to catch any hallucinated URLs

### Behavioral Rules (in priority order)

1. **Don't recommend turn 1 on vague queries** — the behavior probe explicitly tests this
2. **Recommend 1–10 items once context is sufficient** — role + purpose = enough
3. **Refine, don't restart** — when user says "add X" or "remove Y", update the current shortlist
4. **Compare from catalog data** — when asked to compare two products, use description fields only
5. **Refuse off-scope** — legal questions, general hiring advice, prompt injection

### Scope Enforcement

The system prompt explicitly:
- Lists what the agent must NOT do (legal advice, general hiring advice)
- Specifies that every URL must come from the catalog
- Validates post-hoc: all recommendation URLs are checked against the catalog; invalid ones are silently dropped with a warning log

### Stateless design

The API stores no per-conversation state. Full history is sent on every request, which:
- Simplifies deployment (no session store needed)
- Makes the service trivially horizontally scalable
- Matches exactly what the evaluator sends

---

## Prompt Engineering

Key prompt decisions:

**Role framing**: "expert SHL assessment consultant" not "helpful assistant" — creates appropriate expertise stance and natural refusal behavior for off-scope requests.

**Explicit clarification rule**: "Ask ONE focused question" — prevents the agent from asking multiple questions at once, which the sample conversations show SHL expects.

**Refinement instruction**: Explicitly stating "Refine mid-conversation... do NOT start over" maps directly to the `C8/C9` behavior pattern of add/remove from shortlist.

**JSON-only output**: Removes the need for regex extraction from prose. Fallback handles parse failures gracefully.

---

## Evaluation Approach

I ran the 10 public sample conversations as a replay harness:

- **Schema compliance**: Every response checked for `reply`, `recommendations`, `end_of_conversation`
- **Catalog-only check**: All recommendation URLs validated against scraped catalog
- **Turn cap**: Conversations capped at 8 turns (user + assistant combined)
- **Recall@10**: Final shortlist compared against expected URLs from sample conversations
- **Behavior probes manually verified**:
  - Vague query → no turn-1 recommendations ✓
  - Off-scope question → polite refusal ✓
  - Mid-conversation refinement → shortlist updated ✓
  - Comparison question → grounded from catalog data ✓

**What didn't work initially**:
- First attempt used a vector DB (FAISS) — retrieval missed edge cases like SVAR accent variants
- Switched to full-context after observing the catalog fits within token limits
- JSON extraction initially failed when LLM wrapped output in markdown code fences — fixed with regex stripping

---

## Stack

- **LLM**: Claude claude-sonnet-4-20250514 (via Anthropic SDK) — chosen for instruction-following reliability and 200k context window
- **Framework**: FastAPI + Pydantic v2 — type-safe, fast, excellent schema validation
- **Deployment**: Render (free tier, Docker) — zero cold start on paid tier, simple env var management
- **No vector DB**: deliberate; catalog size makes it unnecessary

**AI tools used**: Claude (Anthropic) assisted with code generation throughout development. All design decisions, prompt engineering, and evaluation interpretation were done by the author.
