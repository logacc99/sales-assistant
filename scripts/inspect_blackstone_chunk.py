from pathlib import Path
from src.ingestion.chunker import CrawledMarkdownChunker

file_path = Path("data/raw/website/bepbbq.com/crawled_san-pham_blackstone-griddle-original-28-2147_d6f02eb7.md")
chunker = CrawledMarkdownChunker()
chunks = chunker.chunk(file_path)

print(f"Total chunks produced: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"\n=== CHUNK {i+1} ===")
    print(f"ID: {c.id}")
    print(f"Content Hash: {c.content_hash}")
    print(f"Category: {c.metadata.category}")
    if c.metadata.product:
        print(f"Product Name: {c.metadata.product.name}")
        print(f"SKU / ID: {c.metadata.product.product_id}")
        print(f"Price: {c.metadata.product.price:,.0f} {c.metadata.product.currency}")
        print(f"Stock Status: {c.metadata.product.stock_status}")
        print(f"Categories: {c.metadata.product.categories}")
        print(f"Attributes: {c.metadata.product.attributes}")
        print(f"Product URL: {c.metadata.product.product_url}")
    print("\n--- FORMATTED CHUNK CONTENT FOR VECTOR EMBEDDING & BM25 ---")
    print(c.content)
