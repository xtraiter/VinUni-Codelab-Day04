import json
import os
from typing import List, Dict, Any
from datetime import datetime

# Tránh Trap 1: Tính đường dẫn tương đối chính xác tới thư mục raw-data
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog (Milestone 2.1)
# ---------------------------------------------------------------------------

def search_product_catalog(
    category: str,
    max_price: int = 999999999999,
    min_price: int = 0,
    seats: int = None,
    keyword: str = None
) -> List[Dict[str, Any]]:
    """Tra cứu sản phẩm/dịch vụ Vingroup với nhiều tiêu chí lọc nâng cao."""
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]
    
    with open(catalog_file, "r", encoding="utf-8") as f:
        products = json.load(f)
    
    results = []
    for p in products:
        # 1. Lọc theo danh mục
        if p.get("category", "").lower() != category.lower():
            continue
        # 2. Lọc theo tầm giá (min - max)
        price = p.get("price_vnd", 0)
        if not (min_price <= price <= max_price):
            continue
        # 3. Lọc theo số chỗ ngồi (nếu là xe)
        if seats is not None and p.get("seats") != seats:
            continue
        # 4. Lọc theo từ khóa (tên mẫu xe, địa danh Vinpearl)
        if keyword and keyword.lower() not in p.get("name", "").lower():
            continue
            
        results.append(p)
        
    return results


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket (Milestone 2.2)
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")
    
    # Tránh Trap 2: Load existing tickets trước khi append
    existing_tickets = []
    if os.path.exists(tickets_file):
        with open(tickets_file, "r", encoding="utf-8") as f:
            try:
                existing_tickets = json.load(f)
            except json.JSONDecodeError:
                existing_tickets = []
    
    # Generate ticket ID theo format TK-YYYYMMDD-seq
    today = datetime.now().strftime("%Y%m%d")
    seq = len(existing_tickets) + 1
    ticket_id = f"TK-{today}-{seq:03d}"
    
    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority.lower(),
        "status": "open",
        "created_at": datetime.now().isoformat() + "+07:00",
        "category": "general"
    }
    existing_tickets.append(new_ticket)
    
    os.makedirs(os.path.dirname(tickets_file), exist_ok=True)
    with open(tickets_file, "w", encoding="utf-8") as f:
        json.dump(existing_tickets, f, indent=2, ensure_ascii=False)
    
    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "priority": priority.lower(),
        "status": "open",
        "message": f"Ticket {ticket_id} đã được tạo thành công."
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM (Milestone 2.3)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
   {
        "name": "search_product_catalog",
        "description": "Tra cứu danh mục sản phẩm xe điện VinFast hoặc gói dịch vụ Vinpearl theo nhiều tiêu chí lọc.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Loại sản phẩm: 'xe_dien' hoặc 'du_lich'.",
                    "enum": ["xe_dien", "du_lich"]
                },
                "max_price": {
                    "type": "integer",
                    "description": "Giá tối đa tính bằng VNĐ."
                },
                "min_price": {
                    "type": "integer",
                    "description": "Giá tối thiểu tính bằng VNĐ."
                },
                "seats": {
                    "type": "integer",
                    "description": "Số chỗ ngồi của xe (ví dụ: 4, 5, 7)."
                },
                "keyword": {
                    "type": "string",
                    "description": "Từ khóa tìm kiếm trong tên sản phẩm (ví dụ: 'VF8', 'Phú Quốc', 'Nha Trang')."
                }
            },
            "required": ["category"]
        }
    },
    {
        "name": "submit_support_ticket",
        "description": "Ghi nhận yêu cầu hỗ trợ hoặc sự cố kỹ thuật của khách hàng vào hệ thống ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Tên khách hàng yêu cầu hỗ trợ."
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả chi tiết vấn đề cần hỗ trợ."
                },
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên ('low', 'medium', 'high').",
                    "enum": ["low", "medium", "high"]
                }
            },
            "required": ["customer_name", "issue_description"]
        }
    }
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}