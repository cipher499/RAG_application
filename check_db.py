"""
check_db.py  —  cross-references every book in data/books against ChromaDB.
Run from your project root:  python check_db.py
"""

import os
import re
from collections import defaultdict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "chroma_db"
BOOKS_PATH  = "data/books"

# ── replicate the same filename parsing logic from ingest.py ──────────────────
def parse_filename(filename: str) -> dict:
    name, ext = os.path.splitext(filename)

    if " - " in name:
        author, title = name.split(" - ", 1)
    else:
        author, title = "unknown", name

    domain_match = re.search(r'\((.+?)\)', title)
    domain = domain_match.group(1).strip().lower() if domain_match else "unknown"
    title  = re.sub(r'\(.+?\)', '', title).strip()

    return {
        "file"   : filename,
        "ext"    : ext.lower(),
        "author" : author.strip().lower().replace(" ", "_"),
        "book"   : title.strip().lower().replace(" ", "_"),
        "domain" : domain,
    }
# ─────────────────────────────────────────────────────────────────────────────

def check_db():
    # ── 1. scan data/books ────────────────────────────────────────────────────
    if not os.path.isdir(BOOKS_PATH):
        print(f"\n  ❌  '{BOOKS_PATH}' folder not found. Run from your project root.\n")
        return

    book_files = [
        f for f in os.listdir(BOOKS_PATH)
        if os.path.isfile(os.path.join(BOOKS_PATH, f))
        and os.path.splitext(f)[1].lower() in (".pdf", ".epub")
    ]

    if not book_files:
        print(f"\n  No .pdf or .epub files found in '{BOOKS_PATH}'.\n")
        return

    books = [parse_filename(f) for f in sorted(book_files)]

    # ── 2. pull all chunks from ChromaDB ─────────────────────────────────────
    vectorstore = Chroma(
        collection_name="author_council",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=CHROMA_PATH
    )
    all_docs = vectorstore.get()
    total_chunks = len(all_docs["ids"])

    # count chunks per (author, book) pair
    chunk_counts = defaultdict(int)
    for meta in all_docs["metadatas"]:
        key = (meta.get("author", "unknown"), meta.get("book", "unknown"))
        chunk_counts[key] += 1

    # ── 3. print report ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Book ingestion status — '{BOOKS_PATH}'")
    print(f"  Total chunks in ChromaDB: {total_chunks}")
    print(f"{'='*65}\n")

    col_file   = 42
    col_type   = 6
    col_status = 12

    header = f"  {'File':<{col_file}} {'Type':<{col_type}} {'Status':<{col_status}} Chunks"
    print(header)
    print(f"  {'─'*(len(header)-2)}")

    for b in books:
        key    = (b["author"], b["book"])
        count  = chunk_counts.get(key, 0)
        ftype  = b["ext"].upper().lstrip(".")

        if count == 0:
            status = "❌  missing"
        elif count < 10:
            status = "⚠️  low"
        else:
            status = "✅  ok"

        # truncate long filenames for display
        display_name = b["file"]
        if len(display_name) > col_file:
            display_name = display_name[:col_file - 1] + "…"

        print(f"  {display_name:<{col_file}} {ftype:<{col_type}} {status:<{col_status}} {count if count else '—'}")

    # ── 4. summary ────────────────────────────────────────────────────────────
    missing = [b for b in books if chunk_counts.get((b["author"], b["book"]), 0) == 0]
    low     = [b for b in books if 0 < chunk_counts.get((b["author"], b["book"]), 0) < 10]

    print(f"\n  {'─'*63}")
    print(f"  {len(books)} file(s) found   "
          f"{len(books)-len(missing)-len(low)} ok   "
          f"{len(low)} low   "
          f"{len(missing)} not ingested")

    if missing:
        print(f"\n  Not ingested:")
        for b in missing:
            print(f"    • {b['file']}  ({b['ext'].upper()})")
            if b["ext"] == ".epub":
                print(f"      → check: pip install \"unstructured[epub]\"")
                print(f"      → check: the file has no DRM")
            elif b["ext"] == ".pdf":
                print(f"      → check: the PDF is not scanned/image-only")

    print(f"  {'─'*63}\n")

if __name__ == "__main__":
    check_db()