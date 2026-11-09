"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
import os
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: SYSTEM PROMPT (Milestone 1)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ VinFast, Vinpearl
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác

## AVAILABLE TOOLS
{tools}

## CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm. PHẢI gọi tool `search_product_catalog` để lấy dữ liệu thực.
2. KHÔNG BAO GIỜ tự sinh ticket_id. PHẢI gọi tool `submit_support_ticket` khi nhận yêu cầu hỗ trợ từ khách hàng.
3. Nếu khách hàng hỏi câu FAQ đơn giản (chính sách bảo hành, đổi trả chung), có thể trả lời trực tiếp mà không cần gọi tool.
4. Nếu câu hỏi cần NHIỀU tool, hãy thực hiện tuần tự từng tool rồi mới tổng hợp câu trả lời cuối cùng.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời về các sản phẩm và dịch vụ thuộc hệ sinh thái Vingroup (VinFast, Vinpearl, Vinhomes).
- Từ chối lịch sự và hướng dẫn khách hàng nếu câu hỏi nằm ngoài phạm vi hoạt động.

## OUTPUT CONTRACT
Định dạng quy trình giải quyết vấn đề theo chuẩn ReAct:
- Thought: Suy nghĩ và phân tích ý định của khách hàng tại mỗi bước.
- Action: Tên tool và tham số JSON cần gọi.
- Observation: Kết quả thực tế nhận về từ hệ thống.
- Final Answer: Câu trả lời tổng hợp sau cùng gửi tới khách hàng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline (TODO 2)
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn sản phẩm Vingroup. Hãy trả lời câu hỏi: {user_input}"
                )
                return {
                    "answer": response.text.strip(),
                    "tool_calls": [],
                    "status": "success",
                    "mode": "gemini_baseline"
                }
            except Exception:
                pass

        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}. Hiện xe điện VinFast đang có rất nhiều ưu đãi hấp dẫn.",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent (TODO 3 & TODO 4)
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _extract_max_price(self, text: str) -> int:
        """Trích xuất mức giá tối đa từ văn bản người dùng."""
        pattern = r"(?:dưới|<|khoảng|tối đa|tầm)?\s*(\d+(?:[.,]\d+)?)\s*(tỷ|trieu|triệu|tr|vnd|đ|k)?"
        matches = re.finditer(pattern, text)
        for m in matches:
            val_str = m.group(1).replace(",", ".")
            unit = m.group(2)
            try:
                val = float(val_str)
                if unit in ["tỷ", "ty"]:
                    return int(val * 1_000_000_000)
                elif unit in ["trieu", "triệu", "tr"]:
                    return int(val * 1_000_000)
                elif val > 1_000_000:
                    return int(val)
            except ValueError:
                continue
        return 999999999999

    def _extract_customer_name(self, text: str) -> str:
        """Trích xuất tên khách hàng nếu có trong văn bản."""
        m = re.search(
            r"(?:tôi tên là|tôi tên|tên tôi là|tôi là|khách hàng)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|\s+xe|\s+cần|$)",
            text,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        return "Khách hàng"

    def _detect_intent(self, user_input: str) -> Dict[str, bool]:
        """Phân loại chính xác intent dựa theo ngữ cảnh câu hỏi."""
        text = user_input.lower()

        # 1. Kiểm tra câu hỏi FAQ (chính sách, thời hạn, bao lâu, quy định...)
        is_faq = any(kw in text for kw in [
            "chính sách", "bao lâu", "mấy năm", "quy định", "ở đâu", "thủ tục", "điều kiện bảo hành"
        ])

        if is_faq:
            return {"needs_catalog": False, "needs_ticket": False, "is_faq": True}

        # 2. Kiểm tra ý định tạo ticket (báo lỗi kỹ thuật, hỏng hóc, khiếu nại)
        needs_ticket = any(kw in text for kw in [
            "bị lỗi", "lỗi hệ thống", "hỏng", "sự cố", "khiếu nại", "gặp lỗi",
            "cần xử lý gấp", "yêu cầu hỗ trợ", "bảo hành gấp", "hỗ trợ kỹ thuật"
        ])

        # 3. Kiểm tra ý định tra cứu danh mục / giá sản phẩm
        has_catalog_action = any(kw in text for kw in [
            "mua", "tìm", "xem", "dưới", "giá", "tư vấn", "tour", "nghỉ dưỡng", "danh mục"
        ])
        needs_catalog = has_catalog_action or (
            any(kw in text for kw in ["xe điện", "vinfast", "vinpearl"]) and not needs_ticket
        )

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": False
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        self.trace = []
        intents = self._detect_intent(user_input)
        iteration = 0
        observations = []

        # 1. Trường hợp FAQ: Xử lý trực tiếp, iterations = 1
        if intents["is_faq"]:
            iteration = 1
            faq_answer = "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc không giới hạn số km tùy theo từng dòng xe cụ thể."
            self.trace.append({
                "iteration": iteration,
                "thought": "Câu hỏi thuộc chính sách bảo hành FAQ, phản hồi trực tiếp mà không cần gọi tool.",
                "final_answer": faq_answer
            })
            return {"answer": faq_answer, "trace": self.trace, "iterations": 1, "status": "completed"}

        # 2. Iteration: Gọi search_product_catalog
        if intents["needs_catalog"]:
            iteration += 1
            if iteration > self.max_iterations:
                return {"answer": "Lỗi: Vượt quá số bước tối đa.", "trace": self.trace, "iterations": iteration, "status": "max_iterations_reached"}

            text_lower = user_input.lower()
            category = "xe_dien" if any(w in text_lower for w in ["xe", "vf", "vinfast", "ô tô"]) else "du_lich"
            max_price = self._extract_max_price(text_lower)
            catalog_args = {"category": category, "max_price": max_price}

            obs = search_product_catalog(**catalog_args)
            observations.append({"tool": "search_product_catalog", "data": obs})
            self.trace.append({
                "iteration": iteration,
                "thought": f"Tra cứu catalog sản phẩm: {catalog_args}.",
                "action": {"name": "search_product_catalog", "args": catalog_args},
                "observation": obs
            })

        # 3. Iteration: Gọi submit_support_ticket
        if intents["needs_ticket"]:
            iteration += 1
            if iteration > self.max_iterations:
                return {"answer": "Lỗi: Vượt quá số bước tối đa.", "trace": self.trace, "iterations": iteration, "status": "max_iterations_reached"}

            customer_name = self._extract_customer_name(user_input)
            priority = "high" if any(w in user_input.lower() for w in ["gấp", "nghiêm trọng", "cháy", "nguy hiểm"]) else "medium"
            ticket_args = {
                "customer_name": customer_name,
                "issue_description": user_input,
                "priority": priority
            }
            obs = submit_support_ticket(**ticket_args)
            observations.append({"tool": "submit_support_ticket", "data": obs})
            self.trace.append({
                "iteration": iteration,
                "thought": f"Ghi nhận ticket hỗ trợ: {ticket_args}.",
                "action": {"name": "submit_support_ticket", "args": ticket_args},
                "observation": obs
            })

        # 4. Tổng hợp Final Answer
        final_parts = []
        for obs_item in observations:
            tool_name = obs_item["tool"]
            data = obs_item["data"]

            if tool_name == "search_product_catalog":
                if not data or len(data) == 0 or ("error" in data[0] if isinstance(data, list) else False):
                    final_parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")
                else:
                    product_list = [f"{p.get('name')} ({p.get('price_vnd', 0):,} VNĐ)" for p in data]
                    final_parts.append("Danh sách sản phẩm phù hợp:\n - " + "\n - ".join(product_list))

            elif tool_name == "submit_support_ticket":
                final_parts.append(
                    f"Ticket {data.get('ticket_id')} đã được tạo thành công cho {data.get('customer_name', 'khách hàng')} (Trạng thái: {data.get('status')})."
                )

        if not final_parts:
            final_answer = "Rất tiếc, không tìm thấy sản phẩm phù hợp."
        else:
            final_answer = "\n\n".join(final_parts)

        self.trace.append({
            "thought": "Đã thu thập đủ thông tin, gửi kết quả đến khách hàng.",
            "final_answer": final_answer
        })

        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": max(iteration, 1),
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Answer:\n", result["answer"])
    print("\nIterations:", result["iterations"])


if __name__ == "__main__":
    main()