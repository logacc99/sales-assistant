"""Prompt builders, grounding guardrails, and context formatters for Bedrock generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.generation.models import GenerationRequest, ShopperContext
from src.retrieval.models import RetrievedChunk

SUPPORT_URL = "https://store.example.com/support"
NO_STACKING_DISCLAIMER_VI = "Lưu ý: Các mã giảm giá không thể kết hợp hoặc cộng dồn trong cùng một đơn hàng."
NO_STACKING_DISCLAIMER_EN = "Please note: Promotional codes cannot be combined or stacked at checkout."

OUT_OF_STOCK_NOTICE_VI = (
    "Sản phẩm hiện đang tạm hết hàng. Nhân viên cửa hàng sẽ sớm liên hệ lại với bạn để hỗ trợ thông tin nhập hàng/đặt trước."
)
OUT_OF_STOCK_NOTICE_EN = (
    "The requested product is currently out of stock. Our store staff will contact you shortly regarding restock timelines or pre-orders."
)

DEFAULT_SYSTEM_PROMPT = f"""Bạn là Trợ lý Bán hàng Trực tuyến (Sales Assistant) chuyên nghiệp, tận tâm và đáng tin cậy của cửa hàng thương mại điện tử.
Nhiệm vụ của bạn là tư vấn sản phẩm, giải thích chính sách và hướng dẫn chương trình khuyến mãi cho khách hàng dựa trên thông tin chính xác từ cửa hàng.

QUY TẮC CỐT LÕI VỀ CĂN CỨ THÔNG TIN (GROUNDING RULES):
1. CHỈ SỬ DỤNG THÔNG TIN TRONG NGỮ CẢNH: Bạn TUYỆT ĐỐI CHỈ trả lời dựa trên các dữ liệu thực tế được cung cấp trong thẻ <retrieved_context>. Không bao giờ bịa đặt giá tiền, thông số kỹ thuật, bảo hành hoặc chương trình giảm giá không có trong ngữ cảnh.
2. TRÍCH DẪN NGUỒN BẰNG NGOẶC VUÔNG: Trong ngữ cảnh, mỗi đoạn thông tin đều được đánh số thứ tự như [1], [2], v.v. Khi bạn đề cập đến sản phẩm, chính sách hoặc mã giảm giá cụ thể, hãy trích dẫn nguồn tương ứng bằng cách đặt số thứ tự trong ngoặc vuông ngay sau nội dung (ví dụ: "Bếp nướng gas Weber Spirit II [1] có giá 12.500.000 ₫").
3. THÔNG BÁO HẾT HÀNG (OUT OF STOCK): Nếu sản phẩm khách hàng quan tâm có trạng thái 'out_of_stock' hoặc 'backorder', bạn PHẢI thông báo rõ: "{OUT_OF_STOCK_NOTICE_VI}". Không cam kết ngày giao hàng nếu không có văn bản chính thức.
4. CHƯƠNG TRÌNH KHUYẾN MÃI (GIAI ĐOẠN 1):
   - Khi có chương trình khuyến mãi phù hợp, hãy liệt kê các mã hợp lệ, điều kiện áp dụng (mức chi tiêu tối thiểu, thời hạn, danh mục áp dụng).
   - KHÔNG tự ý tính toán giá thanh toán cuối cùng của đơn hàng trong giai đoạn này.
   - BẮT BUỘC luôn đính kèm câu lưu ý: "{NO_STACKING_DISCLAIMER_VI}"
5. KHI KHÔNG TÌM THẤY THÔNG TIN: Nếu ngữ cảnh không chứa thông tin cần thiết để giải đáp câu hỏi của khách hàng, hãy lịch sự giải thích rằng cửa hàng chưa có thông tin này và hướng dẫn khách liên hệ Trung tâm Hỗ trợ Khách hàng: {SUPPORT_URL}.
6. PHONG CÁCH & NGÔN NGỮ:
   - Giọng điệu lịch sự, nhã nhặn, thân thiện của nhân viên tư vấn bán hàng Việt Nam ("Dạ chào bạn...", "Dạ theo thông tin từ cửa hàng...").
   - Nếu khách hàng hỏi bằng tiếng Anh, hãy trả lời bằng tiếng Anh lưu loát với cùng các quy tắc nghiêm ngặt trên.
   - Trình bày câu trả lời rõ ràng bằng Markdown (danh sách đầu dòng, in đậm tên sản phẩm/mã giảm giá).
"""


class PromptBuilder:
    """Constructs grounded system prompts and messages for Bedrock Converse API."""

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def format_chunk_xml(self, chunk: RetrievedChunk, index: int) -> str:
        """Formats an individual chunk with metadata into an XML block."""
        category = chunk.category
        meta = chunk.metadata or {}
        attrs: List[str] = [f'index="[{index}]"', f'category="{category}"']

        if category == "product":
            product_name = meta.get("name") or meta.get("product_name") or ""
            sku = meta.get("sku") or ""
            price = meta.get("price")
            stock = meta.get("stock_status") or "in_stock"
            url = meta.get("product_url") or ""
            if product_name:
                attrs.append(f'name="{product_name}"')
            if sku:
                attrs.append(f'sku="{sku}"')
            if price is not None:
                attrs.append(f'price="{price}"')
            attrs.append(f'stock_status="{stock}"')
            if url:
                attrs.append(f'url="{url}"')

        elif category == "promotion":
            promo_code = meta.get("promo_code") or ""
            discount_type = meta.get("discount_type") or ""
            discount_val = meta.get("discount_value")
            min_spend = meta.get("min_spend", 0.0)
            if promo_code:
                attrs.append(f'promo_code="{promo_code}"')
            if discount_type:
                attrs.append(f'discount_type="{discount_type}"')
            if discount_val is not None:
                attrs.append(f'discount_value="{discount_val}"')
            attrs.append(f'min_spend="{min_spend}"')

        elif category == "policy":
            policy_type = meta.get("policy_type") or ""
            section_title = meta.get("section_title") or ""
            url = meta.get("policy_url") or ""
            if policy_type:
                attrs.append(f'policy_type="{policy_type}"')
            if section_title:
                attrs.append(f'section_title="{section_title}"')
            if url:
                attrs.append(f'policy_url="{url}"')

        attr_str = " ".join(attrs)
        content = (chunk.content or "").strip()
        return f"<chunk {attr_str}>\n{content}\n</chunk>"

    def format_context_xml(self, chunks: List[RetrievedChunk]) -> str:
        """Assembles all retrieved chunks into a parent <retrieved_context> XML block."""
        if not chunks:
            return "<retrieved_context>\n(Không tìm thấy dữ liệu liên quan trong danh mục cửa hàng)\n</retrieved_context>"

        chunk_blocks = [
            self.format_chunk_xml(chunk, idx) for idx, chunk in enumerate(chunks, start=1)
        ]
        return "<retrieved_context>\n" + "\n\n".join(chunk_blocks) + "\n</retrieved_context>"

    def format_shopper_context_xml(self, context: Optional[ShopperContext]) -> str:
        """Formats the active cart and shopper state into an XML block."""
        if not context:
            return ""

        lines = ["<shopper_context>"]
        lines.append(f"  <current_time_utc>{context.current_time.isoformat()}</current_time_utc>")
        lines.append(f"  <cart_subtotal currency=\"VND\">{context.cart_subtotal}</cart_subtotal>")
        if context.cart_items:
            lines.append("  <cart_items>")
            for item in context.cart_items:
                lines.append(
                    f'    <item sku="{item.sku}" name="{item.name}" price="{item.price}" qty="{item.quantity}" />'
                )
            lines.append("  </cart_items>")
        else:
            lines.append("  <cart_items empty=\"true\" />")
        lines.append("</shopper_context>")
        return "\n".join(lines)

    def build_messages(
        self,
        request: GenerationRequest,
        retained_chunks: List[RetrievedChunk],
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """
        Builds Bedrock Converse-compatible system prompts and user messages.
        
        Returns:
            Tuple of (system_prompts, messages)
        """
        system_text = (
            request.config.system_prompt_override
            if request.config and request.config.system_prompt_override
            else self.system_prompt
        )
        system_prompts = [{"text": system_text}]

        context_xml = self.format_context_xml(retained_chunks)
        shopper_xml = self.format_shopper_context_xml(request.shopper_context)

        user_content_parts = []
        if shopper_xml:
            user_content_parts.append(shopper_xml)
        user_content_parts.append(context_xml)
        user_content_parts.append(f"<customer_query>\n{request.query}\n</customer_query>")
        user_content_parts.append("Hãy trả lời câu hỏi của khách hàng theo đúng các quy tắc căn cứ thông tin trên.")

        full_user_text = "\n\n".join(user_content_parts)

        messages: List[Dict[str, Any]] = []

        # Incorporate prior conversation history if provided
        for turn in request.conversation_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": [{"text": content}]})

        # Add current user prompt
        messages.append({"role": "user", "content": [{"text": full_user_text}]})

        return system_prompts, messages
