"""
Fake store backend matching the example API specs, so you can try the chatbot without your real APIs.
Run:  uvicorn examples.mock_backend:app --port 8000

Test users: buyers c_1, c_2   vendors v_1, v_2
"""
from typing import Optional

from fastapi import Body, FastAPI, HTTPException

app = FastAPI(title="Mock store backend")

PRODUCTS = {
    "p_1": {"id": "p_1", "title": "Black leather tote bag", "category": "bag", "price": 25000, "stock": 4, "vendor_id": "v_1"},
    "p_2": {"id": "p_2", "title": "Brown leather shoulder bag", "category": "bag", "price": 18000, "stock": 9, "vendor_id": "v_2"},
    "p_3": {"id": "p_3", "title": "Mini crossbody bag, tan", "category": "bag", "price": 12500, "stock": 0, "vendor_id": "v_1"},
    "p_4": {"id": "p_4", "title": "White leather sneakers", "category": "shoe", "price": 30000, "stock": 6,
            "sizes": [39, 40, 41, 42, 43], "vendor_id": "v_2"},
    "p_5": {"id": "p_5", "title": "Block heel sandals, nude", "category": "shoe", "price": 22000, "stock": 3,
            "sizes": [37, 38, 39, 40], "vendor_id": "v_1"},
    "p_6": {"id": "p_6", "title": "Ankara midi wrap dress", "category": "clothing", "price": 27000, "stock": 5,
            "sizes": ["S", "M", "L"], "vendor_id": "v_2"},
}
ORDERS = {
    "o_100": {"id": "o_100", "customer_id": "c_1", "status": "shipped", "items": [{"item_id": "i_1", "product_id": "p_4", "size": 42}],
              "total": 30000, "payment": {"method": "card", "card_number": "**** 4417", "status": "paid"}},
    "o_101": {"id": "o_101", "customer_id": "c_1", "status": "pending", "items": [{"item_id": "i_2", "product_id": "p_1"}],
              "total": 25000, "payment": {"method": "transfer", "status": "paid"}},
    "o_102": {"id": "o_102", "customer_id": "c_2", "status": "delivered", "items": [{"item_id": "i_3", "product_id": "p_6", "size": "M"}],
              "total": 27000, "payment": {"method": "card", "status": "paid"}},
}
TRACKING = {"o_100": {"courier": "GIG Logistics", "status": "In transit - Ibadan hub", "estimated_delivery": "2 working days"},
            "o_102": {"courier": "Kwik", "status": "Delivered", "delivered_on": "last Tuesday"}}
PAYOUTS = {"v_1": [{"amount": 47500, "status": "paid", "date": "last Friday", "bank_account": "0123456789"}],
           "v_2": [{"amount": 81000, "status": "processing", "date": "this Friday", "bank_account": "9876543210"}]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/products")
def search_products(q: str = "", category: Optional[str] = None, max_price: Optional[float] = None, limit: int = 10):
    words = q.lower().split()
    out = [p for p in PRODUCTS.values()
           if (not category or p["category"] == category) and (max_price is None or p["price"] <= max_price)
           and all(w in p["title"].lower() for w in words)]
    return out[:limit]


@app.get("/products/{product_id}")
def get_product(product_id: str):
    if product_id not in PRODUCTS:
        raise HTTPException(404, "Product not found")
    return PRODUCTS[product_id]


@app.get("/products/{product_id}/recommendations")
def get_recommendations(product_id: str, max_price: Optional[float] = None, complete_the_look: bool = False):
    p = get_product(product_id)
    recs = [x for x in PRODUCTS.values() if x["id"] != product_id and x["stock"] > 0
            and ((x["category"] != p["category"]) if complete_the_look else (x["category"] == p["category"]))
            and (max_price is None or x["price"] <= max_price)]
    return {"product_id": product_id, "recommendations": recs[:5]}


@app.get("/vendors/{vendor_id}/products")
def list_vendor_products(vendor_id: str):
    return [p for p in PRODUCTS.values() if p["vendor_id"] == vendor_id]


@app.get("/vendors/{vendor_id}/payouts")
def list_vendor_payouts(vendor_id: str):
    return PAYOUTS.get(vendor_id, [])


@app.get("/admin/stats")
def admin_stats():
    return {"orders": len(ORDERS), "revenue": sum(o["total"] for o in ORDERS.values()), "products": len(PRODUCTS)}


def _own_order(order_id, customer_id):
    o = ORDERS.get(order_id)
    if not o or (customer_id and o["customer_id"] != customer_id):
        raise HTTPException(404, "Order not found")
    return o


@app.get("/orders")
def list_my_orders(customer_id: str, status: Optional[str] = None):
    return [o for o in ORDERS.values() if o["customer_id"] == customer_id and (not status or o["status"] == status)]


@app.get("/orders/{order_id}")
def get_order(order_id: str, customer_id: Optional[str] = None):
    return _own_order(order_id, customer_id)


@app.post("/orders/{order_id}/cancel")
def cancel_order(order_id: str, body: dict = Body(...)):
    o = _own_order(order_id, body.get("customer_id"))
    if o["status"] != "pending":
        raise HTTPException(409, f"Order is already {o['status']} and can't be cancelled; request a return instead")
    o["status"] = "cancelled"
    return {"order_id": order_id, "status": "cancelled", "refund": "5-10 working days to original payment method"}


@app.get("/shipments/{order_id}/tracking")
def track(order_id: str):
    if order_id not in TRACKING:
        raise HTTPException(404, "No shipment yet for this order")
    return TRACKING[order_id]


@app.post("/returns")
def create_return(body: dict = Body(...)):
    o = _own_order(body.get("order_id"), body.get("customer_id"))
    if o["status"] != "delivered":
        raise HTTPException(409, "Only delivered orders can be returned")
    return {"return_id": "r_" + o["id"][2:], "status": "requested", "next_step": "Courier pickup within 2 working days"}


@app.post("/admin/payments/{payment_id}/refund")
def refund(payment_id: str):
    return {"payment_id": payment_id, "status": "refunded"}
