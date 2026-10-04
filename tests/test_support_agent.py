"""Support agent tests: spec parsing, mediator policy, and the agent loop with a fake LLM."""
import json
from types import SimpleNamespace

import httpx
import pytest

from support_agent.agent import SessionStore, SupportAgent
from support_agent.knowledge import KnowledgeBase
from support_agent.mediator import Mediator, UserContext
from support_agent.policy import Policy
from support_agent.settings import AgentSettings
from support_agent.spec_loader import load_source, parse_openapi, parse_postman
from support_agent.tickets import TicketStore

POSTMAN = "collections/example_shop.postman_collection.json"
OPENAPI = "collections/example_catalog.openapi.yaml"


def test_openapi_parsing():
    ops, skipped = parse_openapi(load_source(OPENAPI), "catalog")
    names = {o.name for o in ops}
    assert "catalog__get_recommendations" in names and "catalog__list_vendor_payouts" in names
    assert any("upload_product_image" in s for s in skipped)       # file upload skipped
    rec = next(o for o in ops if o.op_id == "get_recommendations")
    assert rec.path_params == ["product_id"] and "complete_the_look" in rec.query_params


def test_postman_parsing():
    ops, skipped = parse_postman(load_source(POSTMAN), "shop")
    by = {o.op_id: o for o in ops}
    assert by["orders_get_order"].path == "/orders/{order_id}"
    assert by["orders_cancel_order"].body_mode == "json"
    assert "customer_id" in by["orders_list_my_orders"].query_params
    assert by["shipping_track_shipment"].path_params == ["order_id"]
    assert any("Upload invoice" in s for s in skipped)


@pytest.fixture
def mediator(tmp_path):
    calls = []

    def handler(req: httpx.Request):
        calls.append(req)
        return httpx.Response(200, json={"path": req.url.path, "query": dict(req.url.params),
                                         "card_number": "4111", "bank_account": "0123456789",
                                         "body": json.loads(req.content) if req.content else None})

    services = [{"name": "catalog", "base_url": "http://catalog", "openapi": OPENAPI},
                {"name": "shop", "base_url": "http://shop", "postman": POSTMAN}]
    m = Mediator(services, Policy.load("config/policy.yaml"), http=httpx.Client(transport=httpx.MockTransport(handler)),
                 audit_path=str(tmp_path / "audit.jsonl"))
    m.load()
    m.calls = calls
    return m


BUYER = UserContext("c_42", "buyer")
VENDOR = UserContext("v_7", "vendor")
ADMIN = UserContext("admin_1", "admin")


def test_roles_and_hidden(mediator):
    buyer_tools = {t["name"] for t in mediator.tools(BUYER)}
    assert "catalog__get_recommendations" in buyer_tools
    assert "catalog__admin_stats" not in buyer_tools and "catalog__health" not in buyer_tools
    assert "catalog__list_vendor_payouts" not in buyer_tools
    assert not any(n.startswith("shop__admin_") for n in buyer_tools)
    assert "error" in mediator.call("catalog__admin_stats", {}, BUYER)


def test_identity_binding_cannot_be_overridden(mediator):
    tool = next(t for t in mediator.tools(VENDOR) if t["name"] == "catalog__list_vendor_payouts")
    assert "vendor_id" not in tool["input_schema"]["properties"]      # AI can't even see it
    out = mediator.call("catalog__list_vendor_payouts", {"vendor_id": "v_victim"}, VENDOR)
    assert out["data"]["path"] == "/vendors/v_7/payouts"
    assert out["data"]["bank_account"] == "[redacted]" and out["data"]["card_number"] == "[redacted]"
    out = mediator.call("shop__orders_list_my_orders", {"customer_id": "someone_else"}, BUYER)
    assert out["data"]["query"]["customer_id"] == "c_42"
    # admin is not bound
    out = mediator.call("catalog__list_vendor_payouts", {"vendor_id": "v_9"}, ADMIN)
    assert out["data"]["path"] == "/vendors/v_9/payouts"


def test_write_needs_confirmation(mediator):
    out = mediator.call("shop__orders_cancel_order", {"order_id": "o_5", "body": {"reason": "late"}}, BUYER)
    assert out["status"] == "pending_confirmation" and not mediator.calls
    assert "error" in mediator.confirm(out["action_id"], UserContext("intruder", "buyer"), True)
    res = mediator.confirm(out["action_id"], BUYER, True)
    assert res["ok"] and res["data"]["path"] == "/orders/o_5/cancel"
    assert res["data"]["body"] == {"reason": "late", "customer_id": "c_42"}
    assert "error" in mediator.confirm(out["action_id"], BUYER, True)   # single use


def test_knowledge_search():
    kb = KnowledgeBase("knowledge")
    assert "7 days" in kb.search("can I return shoes")[0]["text"]
    assert "Friday" in kb.search("when do vendors get paid")[0]["text"]


class FakeClient:
    """Scripted stand-in for anthropic.Anthropic()."""
    def __init__(self, script):
        self.script, self.seen = list(script), []
        self.messages = self

    def create(self, **kw):
        self.seen.append(kw)
        return self.script.pop(0)


def tool_use(name, inp, id_="t1"):
    return SimpleNamespace(stop_reason="tool_use",
                           content=[SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)])


def text(t):
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=t)])


def make_agent(mediator, tmp_path, script, **kw):
    s = AgentSettings(**kw)
    client = FakeClient(script)
    return SupportAgent(s, mediator, KnowledgeBase("knowledge"), TicketStore(tmp_path / "t.jsonl"), client), client


def test_agent_loop_uses_api_then_answers(mediator, tmp_path):
    agent, client = make_agent(mediator, tmp_path, [
        tool_use("catalog__get_recommendations", {"product_id": "p_1", "max_price": 20000}),
        text("Here are two bags under 20,000."),
    ])
    sessions = SessionStore()
    s = sessions.get_or_create(None, BUYER)
    s.messages.append({"role": "user", "content": "cheaper bags like product 3?"})
    out = agent.run(s, BUYER)
    assert out["reply"].startswith("Here are") and out["tools_used"] == ["catalog__get_recommendations"]
    assert mediator.calls[-1].url.path == "/products/p_1/recommendations"
    tool_names = {t["name"] for t in client.seen[0]["tools"]}
    assert "search_help_articles" in tool_names and "catalog__admin_stats" not in tool_names


def test_agent_reports_pending_action_and_catalog_mode(mediator, tmp_path):
    agent, client = make_agent(mediator, tmp_path, [
        tool_use("find_api_operations", {"query": "cancel order"}),
        tool_use("call_api_operation", {"name": "shop__orders_cancel_order",
                                        "arguments": {"order_id": "o_9", "body": {"reason": "x"}}}, "t2"),
        text("Please confirm you want to cancel order o_9."),
    ], tool_mode="catalog")
    s = SessionStore().get_or_create(None, BUYER)
    s.messages.append({"role": "user", "content": "cancel order o_9"})
    out = agent.run(s, BUYER)
    assert out["pending_action"] and "o_9" in out["pending_action"]["summary"]
    assert not mediator.calls                                    # nothing executed yet
    assert {t["name"] for t in client.seen[0]["tools"]} >= {"find_api_operations", "call_api_operation"}


def test_session_isolation():
    store = SessionStore()
    s = store.get_or_create(None, BUYER)
    with pytest.raises(PermissionError):
        store.get_or_create(s.id, UserContext("c_99", "buyer"))


def test_full_stack_against_mock_backend(tmp_path):
    """Real HTTP layer: mediator -> mock backend app, including the backend's own ownership check."""
    from fastapi.testclient import TestClient
    from examples.mock_backend import app as backend
    http = TestClient(backend)
    services = [{"name": "catalog", "base_url": "http://testserver", "openapi": OPENAPI},
                {"name": "shop", "base_url": "http://testserver", "postman": POSTMAN}]
    m = Mediator(services, Policy.load("config/policy.yaml"), http=http)
    m.load()
    c1 = UserContext("c_1", "buyer")
    orders = m.call("shop__orders_list_my_orders", {}, c1)["data"]
    assert {o["id"] for o in orders} == {"o_100", "o_101"}
    assert orders[0]["payment"].get("card_number", "[redacted]") == "[redacted]"
    assert m.call("shop__orders_get_order", {"order_id": "o_102"}, c1)["status_code"] == 404   # not c_1's order
    pend = m.call("shop__orders_cancel_order", {"order_id": "o_101", "body": {"reason": "late"}}, c1)
    assert m.confirm(pend["action_id"], c1, True)["data"]["status"] == "cancelled"
    recs = m.call("catalog__get_recommendations", {"product_id": "p_6", "complete_the_look": True}, c1)["data"]
    assert all(r["category"] != "clothing" for r in recs["recommendations"])
