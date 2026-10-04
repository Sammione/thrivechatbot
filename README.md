# Customer Support Agent (mediated architecture)

An AI support assistant for an e-commerce store. It answers buyers, vendors and admins using **your APIs
as its knowledge base**: orders, products, stock, delivery, recommendations, payouts, plus help-centre
articles for policies.

```
 User ──► Your app/backend ──► Support Agent (/chat) ──► LLM (Claude)
          (logs user in,          │                        │ wants to call a tool
           sends identity)        ▼                        ▼
                               MEDIATOR  ◄─────────────────┘
                 ┌───────────────┼──────────────────┬───────────────┐
           role check     force user's own IDs   confirm writes   redact + audit
                 └───────────────┼──────────────────┴───────────────┘
                                 ▼
        Your APIs (OpenAPI / Postman)      Help articles      Human handoff
```

## What the mediator does

The LLM never calls your services directly. Every call goes through the mediator, which:

1. **Reads all your APIs automatically** from OpenAPI specs or Postman collections (`config/services.yaml`)
   and turns each endpoint into a tool. Add an endpoint, call `POST /admin/reload`, and the agent can use it.
2. **Checks the user's role** (buyer / vendor / admin) against `config/policy.yaml`.
3. **Forces identity parameters** from the logged-in session (`bind: {customer_id: user_id}`). The parameter
   is hidden from the AI and overwritten on every call, so a prompt like "show me customer X's orders"
   can't reach another user's data.
4. **Holds write actions** (cancel, return, refund...) until the user presses Confirm.
5. **Adds credentials**, or forwards the user's own token so your backend checks permissions too.
6. **Redacts** sensitive fields (card numbers, tokens, bank accounts) and **audits** every call
   (`data/agent/audit.jsonl`).

The agent can also search **help articles** (`knowledge/*.md`) and **escalate to a human**
(tickets in `data/agent/tickets.jsonl`; swap for Zendesk/Freshdesk in `support_agent/tickets.py`).

With many endpoints (>40) it switches to *catalog mode*: it searches the operation list first, then calls
the one it needs, instead of loading every tool at once. This keeps it accurate and cheaper.

## Try it in 2 minutes (with the mock backend)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...                           # Free key from console.groq.com (or set ANTHROPIC_API_KEY)
export SHOP_NAME="My Store"

uvicorn examples.mock_backend:app --port 8000           # terminal 1: fake store APIs
uvicorn support_agent.server:app --port 8001            # terminal 2: the chatbot
```
Open http://localhost:8001 and set the user at the top. Mock users: buyers `c_1`, `c_2`; vendors `v_1`, `v_2`.

Things to try:
- buyer `c_1`: "Where is my order?", "Cancel my pending order", "Show me bags under 20,000",
  "What shoes go with the Ankara dress p_6?", "Can I return sneakers?", "Show me order o_102" (belongs to c_2: refused)
- vendor `v_1`: "When is my next payout?", "Which of my products are out of stock?"

## Connect your real APIs

1. In `config/services.yaml`, replace the two examples with your services, each with an `openapi:` URL/file
   or an exported `postman:` collection. FastAPI, NestJS (Swagger), Django (drf-spectacular), Laravel
   (Scribe) and Express (swagger-jsdoc) can all produce OpenAPI.
2. In `config/policy.yaml`, add rules. Check the exact operation names with `GET /tools`.
   **Bind every customer/vendor ID parameter to `user_id`** for non-admin roles.
3. Write clear endpoint summaries: they are what the AI reads to choose a tool.
4. Replace `knowledge/help_centre.md` with your real policies (the included text is a placeholder).
5. Endpoints that need file uploads are skipped automatically (listed under `skipped` in `GET /tools`).

## Calling it from your backend

```bash
curl -X POST http://localhost:8001/chat \
  -H "X-User-Id: c_1" -H "X-User-Role: buyer" -H "X-Agent-Key: $AGENT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is my order?", "session_id": null}'
```
Response: `{"session_id", "reply", "pending_action", "tools_used"}`. Send the `session_id` back on the next
message. If `pending_action` is set, show Confirm/Cancel buttons and call
`POST /chat/confirm {"session_id", "action_id", "approve": true}`.

Optional `X-User-Token`: the user's token, forwarded to services that have `forward_user_token: true`.

**Security:** only your backend should call the agent. Set `AGENT_API_KEY`, and never let the browser set
`X-User-Id` / `X-User-Role` itself, otherwise anyone could claim to be an admin.
The test page at `/` is for development only.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /chat` | Send a message, get a reply |
| `POST /chat/confirm` | Confirm or cancel a pending write action |
| `GET /tools` | Operations available to this role, skipped endpoints, load errors |
| `POST /admin/reload` | Re-read API specs, policy and help articles (admin) |
| `GET /` | Test chat page |

## Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `GROQ_API_KEY` | | Free tier key from [console.groq.com](https://console.groq.com) |
| `ANTHROPIC_API_KEY` | | Alternative: Claude API key |
| `AGENT_PROVIDER` | `auto` | `groq`, `anthropic`, `deepseek`, or `auto` |
| `AGENT_MODEL` | `llama-3.3-70b-versatile` (Groq) | Model ID (or `claude-3-5-sonnet-latest`) |
| `SHOP_NAME` | `our store` | used in the assistant's instructions |
| `AGENT_API_KEY` | (none) | shared secret your backend sends as `X-Agent-Key` |
| `AGENT_TOOL_MODE` | `auto` | `direct`, `catalog` or `auto` |
| `AGENT_MAX_STEPS` | `8` | tool-call rounds per message |
| `AGENT_SERVICES` / `AGENT_POLICY` | `config/...yaml` | config file paths |
| `AGENT_ROLES` | `buyer,vendor,admin` | |

Sessions are kept in memory; use Redis (`support_agent/agent.py` `SessionStore`) if you run several instances.

Docker: `docker build -t support-agent . && docker run -p 8001:8001 -e GROQ_API_KEY=... support-agent`

## Project layout
```
support_agent/
  server.py        FastAPI app: /chat, /chat/confirm, /tools, test page
  agent.py         LLM tool-use loop and sessions
  mediator.py      access control, ID binding, confirmations, redaction, audit
  spec_loader.py   OpenAPI + Postman -> operations
  policy.py        rules from policy.yaml
  knowledge.py     help-article search
  tickets.py       human handoff
config/            services.yaml, policy.yaml
knowledge/         help-centre articles (markdown)
collections/       example OpenAPI spec and Postman collection
examples/          mock backend for local testing
tests/             python -m pytest -q   (no API key needed)
```
