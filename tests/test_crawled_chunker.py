"""Unit tests for DocumentCleaner and CrawledMarkdownChunker."""

from pathlib import Path
from src.ingestion.cleaner import DocumentCleaner
from src.ingestion.chunker import CrawledMarkdownChunker

SAMPLE_DIR = Path(__file__).resolve().parent / "fixtures"


def test_frontmatter_extraction():
    cleaner = DocumentCleaner()
    sample_file = SAMPLE_DIR / "crawled_danh-muc_bep-nuong-bbq_bep-nuong-dien_c9141638.md"
    assert sample_file.exists()

    metadata, body = cleaner.extract_frontmatter(sample_file.read_text(encoding="utf-8"))
    assert metadata["url"] == "https://bepbbq.com/danh-muc/bep-nuong-bbq/bep-nuong-dien/"
    assert "Bếp nướng điện ngoài trời" in metadata["title"]
    assert metadata["content_hash"] == "c9141638fb9db917f4050185f3c4f635cbe374d51ac3ebd1d94878457287ab2c"
    assert metadata["category"] == "crawled"


def test_boilerplate_stripping():
    cleaner = DocumentCleaner()
    sample_file = SAMPLE_DIR / "crawled_danh-muc_bep-nuong-bbq_bep-nuong-dien_c9141638.md"
    metadata, body = cleaner.extract_frontmatter(sample_file.read_text(encoding="utf-8"))

    cleaned = cleaner.strip_boilerplate(body)
    # Header navigation should be stripped
    assert "[Skip to content]" not in cleaned
    assert "THE VIETNAM'S LARGEST BARBEQUE RETAILER" not in cleaned
    # Footers should be stripped
    assert "#### VỀ CHÚNG TÔI" not in cleaned
    assert "Giấy CNĐKKD:" not in cleaned
    # Product items should remain
    assert "Pit Boss" in cleaned


def test_non_content_pages_skipped():
    chunker = CrawledMarkdownChunker()

    cart_file = SAMPLE_DIR / "crawled_gio-hang_8e004256.md"
    assert cart_file.exists()
    chunks = chunker.chunk(cart_file)
    assert len(chunks) == 0

    account_file = SAMPLE_DIR / "crawled_tai-khoan_dd7cd8c0.md"
    assert account_file.exists()
    chunks = chunker.chunk(account_file)
    assert len(chunks) == 0


def test_catalog_product_chunking():
    chunker = CrawledMarkdownChunker()
    sample_file = SAMPLE_DIR / "crawled_danh-muc_bep-nuong-bbq_bep-nuong-dien_c9141638.md"
    chunks = chunker.chunk(sample_file)

    assert len(chunks) >= 3
    # Check that chunks are typed and enriched
    for chunk in chunks:
        assert chunk.metadata.category == "product"
        assert chunk.metadata.product is not None
        assert chunk.metadata.product.price > 0
        assert chunk.metadata.product.currency == "VND"
        assert chunk.content_hash is not None
        assert chunk.id is not None
        assert "Tên sản phẩm:" in chunk.content
        assert "Giá:" in chunk.content
        assert "VND" in chunk.content

    # Verify specific out of stock product detection
    out_of_stock_chunks = [c for c in chunks if c.metadata.product.stock_status == "out_of_stock"]
    assert len(out_of_stock_chunks) >= 1
    assert "PRO V4P" in out_of_stock_chunks[0].metadata.product.name

    # Verify in-stock product with add-to-cart ID
    in_stock_chunks = [c for c in chunks if c.metadata.product.stock_status == "in_stock"]
    assert len(in_stock_chunks) >= 1
    navigator = [c for c in in_stock_chunks if "Navigator 1230" in c.metadata.product.name][0]
    assert navigator.metadata.product.product_id == "10523"
    assert navigator.metadata.product.price == 36850000.0


def test_to_langchain_document():
    chunker = CrawledMarkdownChunker()
    sample_file = SAMPLE_DIR / "crawled_danh-muc_bep-nuong-bbq_bep-nuong-dien_c9141638.md"
    chunks = chunker.chunk(sample_file)
    assert len(chunks) > 0

    doc = chunks[0].to_langchain_document()
    assert doc.page_content == chunks[0].content
    assert doc.metadata["id"] == chunks[0].id
    assert doc.metadata["content_hash"] == chunks[0].content_hash
    assert doc.metadata["category"] == "product"
    assert doc.metadata["product_name"] == chunks[0].metadata.product.name


def test_policy_chunking(tmp_path):
    policy_content = """---
url: "https://bepbbq.com/thong-tin/chinh-sach-doi-tra/"
title: "Chính sách đổi trả - BepBBQ"
crawl_timestamp: "2026-09-15T08:31:49.028518+00:00"
content_hash: "policy123hash"
category: "crawled"
---

# Điều kiện đổi trả hàng
Sản phẩm được đổi trả trong vòng 7 ngày kể từ khi nhận hàng.
Sản phẩm phải còn nguyên tem mác, hộp và phụ kiện đi kèm.

## Quy trình xử lý đổi trả
Khách hàng liên hệ hotline 0945 6845 19 để đăng ký đổi trả.
Nhân viên kỹ thuật sẽ kiểm tra tình trạng hàng hóa trước khi hoàn tiền.

#### VỀ CHÚNG TÔI
Giới thiệu BepBBQ
"""
    policy_file = tmp_path / "chinh-sach-doi-tra.md"
    policy_file.write_text(policy_content, encoding="utf-8")

    chunker = CrawledMarkdownChunker()
    chunks = chunker.chunk(policy_file)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.metadata.category == "policy"
        assert chunk.metadata.policy is not None
        assert chunk.metadata.policy.policy_type == "returns"
        assert chunk.metadata.policy.parent_section_content is not None
        assert "Chính sách: returns" in chunk.content


def test_product_detail_chunking(tmp_path):
    detail_content = """---
url: "https://bepbbq.com/san-pham/bep-nuong-blackstone-28/"
title: "Bếp nướng Blackstone Griddle 28 inch"
crawl_timestamp: "2026-09-15T08:31:49.028518+00:00"
content_hash: "detail_test_hash_123"
category: "crawled"
---

# Bếp nướng Blackstone Griddle 28 inch
Mã: BS-28-ORIGINAL
Danh mục: [Bếp nướng BBQ], [Bếp nướng gas]
Thẻ: [Blackstone], [Griddle]

Giá: 15.500.000 ₫

Mô tả sản phẩm:
Bếp nướng gas mặt phẳng cao cấp nhập khẩu từ Mỹ. Thiết kế 2 họng đốt độc lập, bề mặt thép cán nguội.
"""
    detail_file = tmp_path / "bep-nuong-blackstone-28.md"
    detail_file.write_text(detail_content, encoding="utf-8")

    chunker = CrawledMarkdownChunker()
    chunks = chunker.chunk(detail_file)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata.category == "product"
    assert chunk.metadata.product is not None
    assert chunk.metadata.product.name == "Bếp nướng Blackstone Griddle 28 inch"
    assert chunk.metadata.product.price == 15500000.0
    assert chunk.metadata.product.product_id == "BS-28-ORIGINAL"
    assert chunk.metadata.product.sku == "BS-28-ORIGINAL"
    assert "Bếp nướng BBQ" in chunk.metadata.product.categories
    assert "Blackstone" in chunk.metadata.product.attributes["tags"]

