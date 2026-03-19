#!/usr/bin/env python3
"""
Stage 3a: Paper Embedding → Memory Layer (ChromaDB)
=====================================================
PDF를 텍스트로 변환 → chunk → Nemotron Embed로 벡터화 → ChromaDB에 저장.
이후 semantic search, 클러스터링, 유사 논문 탐색에 활용.

Usage:
    # Stage 2 출력(Zotero collection)의 PDF를 임베딩
    python stage3a_embedding.py --input outputs/extractions/stage2_*.json

    # 특정 PDF 디렉토리를 임베딩
    python stage3a_embedding.py --pdf-dir /tmp/research_pipeline_pdfs

    # 쿼리로 유사 논문 검색
    python stage3a_embedding.py --search "Bayesian observer model for set size effects"
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import yaml

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import chromadb
except ImportError:
    chromadb = None


# ─── Config / Logging ──────────────────────────────────────
def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict) -> logging.Logger:
    log_dir = Path(cfg.get("pipeline", {}).get("log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage3a")
    logger.setLevel(getattr(logging, cfg.get("pipeline", {}).get("log_level", "INFO")))
    fh = logging.FileHandler(log_dir / f"stage3a_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    return logger


# ─── PDF → Text ────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: Path) -> str:
    """PyMuPDF로 PDF에서 텍스트를 추출한다."""
    if fitz is None:
        raise ImportError("pymupdf not installed: pip install pymupdf")
    doc = fitz.open(str(pdf_path))
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    full_text = "\n".join(text_parts)
    # Clean up
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = re.sub(r" {2,}", " ", full_text)
    return full_text.strip()


# ─── Text Chunking ─────────────────────────────────────────
def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    텍스트를 토큰 근사치(단어 수)로 chunking한다.
    각 chunk에 인덱스와 위치 정보를 포함.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text,
            "word_start": start,
            "word_end": end,
            "chunk_index": len(chunks),
        })

        if end >= len(words):
            break
        start = end - chunk_overlap

    return chunks


# ─── Nemotron Embedding Client ─────────────────────────────
class NemotronEmbedClient:
    """OpenRouter Nemotron Embed VL 1B v2 API 클라이언트."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def embed(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        """
        텍스트 리스트를 임베딩 벡터로 변환한다.
        OpenRouter rate limit을 고려하여 batch 처리.
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.session.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": batch},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            # Sort by index to preserve order
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            for item in sorted_data:
                all_embeddings.append(item["embedding"])

            if i + batch_size < len(texts):
                time.sleep(0.5)  # Rate limit

        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """단일 텍스트를 임베딩."""
        result = self.embed([text])
        return result[0]


# ─── ChromaDB Memory Store ─────────────────────────────────
class MemoryStore:
    """ChromaDB 기반 논문 벡터 저장소."""

    def __init__(self, persist_dir: str = "./memory_store", collection_name: str = "papers"):
        if chromadb is None:
            raise ImportError("chromadb not installed: pip install chromadb")

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

    def add_paper_chunks(
        self,
        doi: str,
        title: str,
        chunks: list[dict],
        embeddings: list[list[float]],
        metadata_extra: dict | None = None,
    ):
        """논문의 chunk들과 임베딩을 저장한다."""
        ids = []
        documents = []
        metadatas = []
        embs = []

        for chunk, emb in zip(chunks, embeddings):
            chunk_id = f"{doi}__chunk_{chunk['chunk_index']}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            meta = {
                "doi": doi,
                "title": title,
                "chunk_index": chunk["chunk_index"],
                "word_start": chunk["word_start"],
                "word_end": chunk["word_end"],
            }
            if metadata_extra:
                meta.update(metadata_extra)
            metadatas.append(meta)
            embs.append(emb)

        # Upsert (add or update)
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embs,
        )

    def has_paper(self, doi: str) -> bool:
        """DOI가 이미 저장되어 있는지 확인."""
        results = self.collection.get(
            where={"doi": doi},
            limit=1,
        )
        return len(results["ids"]) > 0

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> dict:
        """벡터 유사도 검색."""
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def get_paper_dois(self) -> list[str]:
        """저장된 모든 DOI 목록을 반환."""
        results = self.collection.get(include=["metadatas"])
        dois = set()
        for meta in results["metadatas"]:
            if "doi" in meta:
                dois.add(meta["doi"])
        return sorted(dois)

    def get_stats(self) -> dict:
        """저장소 통계를 반환."""
        total = self.collection.count()
        dois = self.get_paper_dois()
        return {
            "total_chunks": total,
            "total_papers": len(dois),
            "papers": dois,
        }


# ─── Core Pipeline ──────────────────────────────────────────
def embed_papers(
    cfg: dict,
    pdf_paths: list[Path],
    paper_metadata: list[dict] | None = None,
    skip_existing: bool = True,
    logger: logging.Logger | None = None,
) -> dict:
    """
    PDF 리스트를 임베딩하여 ChromaDB에 저장한다.

    Args:
        cfg: config dict
        pdf_paths: PDF 파일 경로 리스트
        paper_metadata: [{"doi": ..., "title": ..., ...}] (optional)
        skip_existing: 이미 임베딩된 DOI는 건너뛰기
        logger: logger instance

    Returns:
        {"embedded": N, "skipped": N, "failed": N}
    """
    log = logger or logging.getLogger("stage3a")

    # Init clients
    nem_cfg = cfg["nemotron"]
    mem_cfg = cfg.get("memory", {})

    embed_client = NemotronEmbedClient(
        api_key=nem_cfg["api_key"],
        base_url=nem_cfg["base_url"],
        model=nem_cfg["model"],
    )
    store = MemoryStore(
        persist_dir=mem_cfg.get("persist_dir", "./memory_store"),
    )

    chunk_size = mem_cfg.get("chunk_size", 512)
    chunk_overlap = mem_cfg.get("chunk_overlap", 50)

    # Build metadata index
    meta_index = {}
    if paper_metadata:
        for p in paper_metadata:
            doi = p.get("doi", "")
            if doi:
                meta_index[doi.lower()] = p

    results = {"embedded": 0, "skipped": 0, "failed": 0, "papers": []}
    total = len(pdf_paths)

    for i, pdf_path in enumerate(pdf_paths, 1):
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            log.warning(f"[{i}/{total}] File not found: {pdf_path}")
            results["failed"] += 1
            continue

        # Extract DOI from filename (format: 10.xxxx_xxx.pdf)
        stem = pdf_path.stem
        doi_candidate = stem.replace("_", "/", 1).replace("_", "/")  # rough DOI recovery
        # Try to find in metadata
        meta = None
        for key, val in meta_index.items():
            safe = re.sub(r"[^\w\-.]", "_", key)
            if safe in stem or key in stem:
                meta = val
                doi_candidate = key
                break

        doi = meta.get("doi", doi_candidate) if meta else doi_candidate
        title = meta.get("title", stem) if meta else stem

        log.info(f"[{i}/{total}] {title[:60]}...")

        # Skip if already embedded
        if skip_existing and store.has_paper(doi):
            log.info(f"  → Already embedded, skipping")
            results["skipped"] += 1
            continue

        try:
            # Extract text
            text = extract_text_from_pdf(pdf_path)
            if len(text) < 100:
                log.warning(f"  → Too little text extracted ({len(text)} chars)")
                results["failed"] += 1
                continue
            log.info(f"  Extracted {len(text)} chars, {len(text.split())} words")

            # Chunk
            chunks = chunk_text(text, chunk_size, chunk_overlap)
            log.info(f"  → {len(chunks)} chunks")

            # Embed
            chunk_texts = [c["text"] for c in chunks]
            embeddings = embed_client.embed(chunk_texts)
            log.info(f"  → {len(embeddings)} embeddings generated")

            # Store
            extra_meta = {}
            if meta:
                extra_meta["year"] = str(meta.get("year", ""))
                extra_meta["authors"] = ", ".join(meta.get("authors", []))[:200]

            store.add_paper_chunks(doi, title, chunks, embeddings, extra_meta)
            log.info(f"  ✓ Stored in memory layer")

            results["embedded"] += 1
            results["papers"].append({"doi": doi, "title": title, "chunks": len(chunks)})

        except Exception as e:
            log.error(f"  ✗ Failed: {e}")
            results["failed"] += 1

    return results


def search_memory(
    cfg: dict,
    query: str,
    n_results: int = 10,
    logger: logging.Logger | None = None,
) -> list[dict]:
    """
    Memory layer에서 semantic search를 수행한다.

    Returns:
        [{"doi": ..., "title": ..., "text": ..., "distance": ...}]
    """
    log = logger or logging.getLogger("stage3a")

    nem_cfg = cfg["nemotron"]
    mem_cfg = cfg.get("memory", {})

    embed_client = NemotronEmbedClient(
        api_key=nem_cfg["api_key"],
        base_url=nem_cfg["base_url"],
        model=nem_cfg["model"],
    )
    store = MemoryStore(persist_dir=mem_cfg.get("persist_dir", "./memory_store"))

    # Embed query
    query_emb = embed_client.embed_single(query)

    # Search
    raw = store.search(query_emb, n_results=n_results)

    # Format results
    results = []
    if raw["ids"] and raw["ids"][0]:
        for idx in range(len(raw["ids"][0])):
            meta = raw["metadatas"][0][idx] if raw["metadatas"] else {}
            results.append({
                "doi": meta.get("doi", ""),
                "title": meta.get("title", ""),
                "text": raw["documents"][0][idx][:300] + "..." if raw["documents"] else "",
                "distance": raw["distances"][0][idx] if raw["distances"] else None,
                "chunk_index": meta.get("chunk_index", -1),
            })

    return results


# ─── Main ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 3a: Paper Embedding → Memory Layer")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", "-i", help="Stage 2 output JSON (contains PDF paths/DOIs)")
    parser.add_argument("--pdf-dir", help="Directory containing PDFs to embed")
    parser.add_argument("--search", "-s", help="Semantic search query")
    parser.add_argument("--n-results", type=int, default=10, help="Number of search results")
    parser.add_argument("--no-skip", action="store_true", help="Re-embed existing papers")
    parser.add_argument("--stats", action="store_true", help="Show memory store statistics")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg)

    # Stats mode
    if args.stats:
        mem_cfg = cfg.get("memory", {})
        store = MemoryStore(persist_dir=mem_cfg.get("persist_dir", "./memory_store"))
        stats = store.get_stats()
        print(f"\nMemory Store Statistics:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Total papers: {stats['total_papers']}")
        for doi in stats["papers"]:
            print(f"    - {doi}")
        return

    # Search mode
    if args.search:
        logger.info(f"Searching: {args.search}")
        results = search_memory(cfg, args.search, args.n_results, logger)
        print(f"\nSearch results for: '{args.search}'")
        print(f"{'='*60}")
        seen_dois = set()
        for r in results:
            if r["doi"] not in seen_dois:
                seen_dois.add(r["doi"])
                print(f"\n  [{r['distance']:.4f}] {r['title'][:70]}")
                print(f"  DOI: {r['doi']}")
                print(f"  Preview: {r['text'][:150]}...")
        return

    # Embedding mode
    logger.info("Stage 3a: Embedding → Memory Layer starting...")

    pdf_paths = []
    paper_metadata = None

    if args.input:
        with open(args.input) as f:
            data = json.load(f)
        # Get PDF paths from Stage 2 output items
        items = data.get("items", [])
        pdf_dir = Path(cfg.get("pdf_download", {}).get("temp_dir", "/tmp/research_pipeline_pdfs"))
        for item in items:
            if item.get("has_pdf"):
                doi = item["doi"]
                safe_name = re.sub(r"[^\w\-.]", "_", doi) + ".pdf"
                pdf_path = pdf_dir / safe_name
                if pdf_path.exists():
                    pdf_paths.append(pdf_path)
        paper_metadata = items
        logger.info(f"Loaded {len(pdf_paths)} PDFs from Stage 2 output")

    elif args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_paths)} PDFs in {pdf_dir}")

    else:
        # Default: scan temp dir
        pdf_dir = Path(cfg.get("pdf_download", {}).get("temp_dir", "/tmp/research_pipeline_pdfs"))
        if pdf_dir.exists():
            pdf_paths = sorted(pdf_dir.glob("*.pdf"))
            logger.info(f"Found {len(pdf_paths)} PDFs in default temp dir")

    if not pdf_paths:
        logger.error("No PDFs found to embed")
        return

    results = embed_papers(
        cfg=cfg,
        pdf_paths=pdf_paths,
        paper_metadata=paper_metadata,
        skip_existing=not args.no_skip,
        logger=logger,
    )

    # Summary
    logger.info(f"\nStage 3a complete:")
    logger.info(f"  ✓ Embedded: {results['embedded']}")
    logger.info(f"  → Skipped:  {results['skipped']}")
    logger.info(f"  ✗ Failed:   {results['failed']}")

    # Save results
    out_dir = Path(cfg.get("extraction", {}).get("output_dir", "./outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = out_dir / f"stage3a_embedding_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"  Output: {outfile}")

    return results


if __name__ == "__main__":
    main()
