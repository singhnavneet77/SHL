# SHL Assessment Recommender

Conversational agent for recommending SHL Individual Test Solutions via dialogue.

## Setup

```bash
pip install -r requirements.txt
export API_KEY=your_key_here
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

### GET /health
```json
{"status": "ok"}
```

### POST /chat

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer, mid-level"},
    {"role": "assistant", "content": "What is the seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are assessments that fit...",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

## Architecture

- **LLM**: GROQ_API_KEY
- **Catalog**: 119 SHL Individual Test Solutions embedded in system prompt
- **Stateless**: Full conversation history sent on each request
- **Schema validation**: Post-LLM URL validation against catalog

## Deployment (Render)

1. Push to GitHub
2. Create Render Web Service, point to repo
3. Set `GROQ_API_KEY` env var
4. Deploy
