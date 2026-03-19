#!/usr/bin/env python3
"""
Stage 2: DOI → PDF Download → Zotero Subcollection
====================================================
Stage 1의 출력(DOI list JSON)을 받아 PDF를 다운로드하고 Zotero에 저장한다.

Usage:
    # Stage 1 출력 파일로 실행
    python stage2_doi2zotero.py --input outputs/extractions/stage1_Test_*.json

    # 직접 DOI 지정
    python stage2_doi2zotero.py --dois "10.1234/abc,10.5678/def" --collection "My_Papers"

    # SQLite 모드로 실행 (Zotero 종료 필요)
    python stage2_doi2zotero.py --input stage1_output.json --zotero-mode sqlite

    # PDF만 다운로드, Zotero 저장 안 함
    python stage2_doi2zotero.py --input stage1_output.json --download-only
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import yaml

from utils.pdf_utils import download_pdf
from utils.zotero_utils import (
    PaperMeta,
    ZoteroBackend,
    create_zotero_backend,
    fetch_crossref_metadata,
)


# ─── Logging ────────────────────────────────────────────────
def setup_logging(cfg: dict) -> logging.Logger:
    log_dir = Path(cfg.get("pipeline", {}).get("log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_level = cfg.get("pipeline", {}).get("log_level", "INFO")

    logger = logging.getLogger("stage2")
    logger.setLevel(getattr(logging, log_level))

    fh = logging.FileHandler(
        log_dir / f"stage2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return logger


# ─── Config ─────────────────────────────────────────────────
def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Load DOIs from Stage 1 output ─────────────────────────
def load_stage1_output(filepath: str) -> dict:
    """Load Stage 1 JSON output."""
    with open(filepath) as f:
        data = json.load(f)
    return data


# ─── Core Pipeline ──────────────────────────────────────────
def run_stage2(
    cfg: dict,
    dois: list[str],
    collection_name: str | None = None,
    parent_collection: str | None = None,
    skip_existing: bool = True,
    download_only: bool = False,
    logger: logging.Logger | None = None,
    elicit_papers: list[dict] | None = None,
) -> dict:
    """
    Stage 2 실행: DOI 리스트 → PDF 다운로드 → Zotero 저장.

    Args:
        cfg: config dict
        dois: list of DOI strings
        collection_name: Zotero subcollection name (auto-created)
        parent_collection: parent collection key (for subcollection)
        skip_existing: skip DOIs already in Zotero
        download_only: only download PDFs, don't save to Zotero
        logger: logger instance
        elicit_papers: Stage 1 paper metadata (to avoid redundant CrossRef calls)

    Returns:
        {"ok": N, "fail": N, "skip": N, "items": [...]}
    """
    log = logger or logging.getLogger("stage2")

    # PDF download settings
    pdf_cfg = cfg.get("pdf_download", {})
    email = cfg.get("email", "")
    strategies = pdf_cfg.get("strategies", ["unpaywall", "scihub", "publisher"])
    scihub_mirrors = pdf_cfg.get("scihub_mirrors", [])
    timeout = pdf_cfg.get("timeout", 30)
    delay = pdf_cfg.get("delay_between_requests", 2)
    dl_dir = Path(pdf_cfg.get("temp_dir", "/tmp/research_pipeline_pdfs"))
    dl_dir.mkdir(parents=True, exist_ok=True)

    # Build elicit metadata index (DOI → paper data) to skip CrossRef when possible
    elicit_index = {}
    if elicit_papers:
        for p in elicit_papers:
            if p.get("doi"):
                elicit_index[p["doi"].lower()] = p

    # Zotero backend
    zotero: ZoteroBackend | None = None
    collection_key = None

    if not download_only:
        zotero = create_zotero_backend(cfg)
        zotero.connect()
        log.info(f"Zotero backend: {type(zotero).__name__}")

        # Find or create collection
        if collection_name:
            collection_key = zotero.find_collection(collection_name)
            if collection_key:
                log.info(f"Using existing collection: '{collection_name}' ({collection_key})")
            else:
                collection_key = zotero.create_collection(collection_name, parent_collection)
                log.info(f"Created collection: '{collection_name}' ({collection_key})")

    # Process DOIs
    results = {"ok": 0, "fail": 0, "skip": 0, "items": [], "failed_dois": []}
    total = len(dois)

    for i, doi in enumerate(dois, 1):
        log.info(f"[{i}/{total}] {doi}")

        # Skip existing
        if skip_existing and zotero and zotero.has_doi(doi):
            log.info(f"  → Already exists, skipping")
            results["skip"] += 1
            continue

        # Get metadata (prefer Elicit data, fallback to CrossRef)
        elicit_data = elicit_index.get(doi.lower())
        if elicit_data:
            meta = PaperMeta(
                doi=doi,
                title=elicit_data.get("title", doi),
                authors=[
                    {"firstName": "", "lastName": a, "creatorType": "author"}
                    for a in elicit_data.get("authors", [])
                ],
                abstract=elicit_data.get("abstract", ""),
                date=str(elicit_data.get("year", "")),
                url=f"https://doi.org/{doi}",
            )
            log.info(f"  Metadata from Elicit: {meta.title[:60]}")
        else:
            meta = fetch_crossref_metadata(doi, email, timeout)
            log.info(f"  Metadata from CrossRef: {meta.title[:60]}")
            time.sleep(0.5)  # CrossRef rate limit

        # Download PDF
        ok, pdf_path, source = download_pdf(
            doi=doi,
            download_dir=dl_dir,
            email=email,
            strategies=strategies,
            scihub_mirrors=scihub_mirrors,
            timeout=timeout,
        )

        # Save to Zotero
        if not download_only and zotero:
            try:
                item_key = zotero.add_item(meta, collection_key)
                if ok and pdf_path:
                    att_key = zotero.attach_pdf(item_key, pdf_path)
                    log.info(f"  ✓ Saved [{source}] → item:{item_key} pdf:{att_key}")
                else:
                    log.info(f"  ✓ Item saved (no PDF) → {item_key}")

                results["ok"] += 1
                results["items"].append({
                    "doi": doi,
                    "key": item_key,
                    "title": meta.title,
                    "has_pdf": ok,
                    "pdf_source": source,
                })
            except Exception as e:
                log.error(f"  ✗ Zotero save failed: {e}")
                results["fail"] += 1
                results["failed_dois"].append({"doi": doi, "error": str(e)})
        elif download_only:
            if ok:
                log.info(f"  ✓ Downloaded [{source}] → {pdf_path}")
                results["ok"] += 1
            else:
                log.info(f"  ✗ PDF not found")
                results["fail"] += 1
                results["failed_dois"].append({"doi": doi, "error": "PDF not found"})

        # Rate limit
        if i < total:
            time.sleep(delay)

    # Cleanup
    if zotero:
        zotero.close()

    return results


# ─── Save Results ───────────────────────────────────────────
def save_results(results: dict, collection_name: str, output_dir: str = "./outputs") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = collection_name.replace(" ", "_")[:50]
    filepath = out / f"stage2_{safe}_{timestamp}.json"

    output = {
        "metadata": {
            "collection_name": collection_name,
            "timestamp": datetime.now().isoformat(),
            "total_processed": results["ok"] + results["fail"] + results["skip"],
            "ok": results["ok"],
            "fail": results["fail"],
            "skip": results["skip"],
        },
        "items": results["items"],
        "failed_dois": results["failed_dois"],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return filepath


# ─── Main ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 2: DOI → PDF → Zotero")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--input", "-i", help="Stage 1 output JSON file")
    parser.add_argument("--dois", help="Comma-separated DOI list (overrides --input)")
    parser.add_argument("--collection", "-c", help="Override collection name")
    parser.add_argument("--parent-collection", help="Parent collection key for subcollection")
    parser.add_argument("--zotero-mode", choices=["api", "sqlite"], help="Override Zotero mode")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip existing DOIs")
    parser.add_argument("--download-only", action="store_true", help="Only download PDFs")
    parser.add_argument("--max-papers", "-n", type=int, help="Limit number of papers to process")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg)
    logger.info("Stage 2: DOI → PDF → Zotero starting...")

    # Override Zotero mode if specified
    if args.zotero_mode:
        cfg.setdefault("pipeline", {})["zotero_mode"] = args.zotero_mode

    # Load DOIs
    elicit_papers = None
    if args.dois:
        dois = [d.strip() for d in args.dois.split(",") if d.strip()]
        collection_name = args.collection or "cli_import"
    elif args.input:
        data = load_stage1_output(args.input)
        dois = data.get("dois", [])
        elicit_papers = data.get("papers", [])
        collection_name = args.collection or data.get("metadata", {}).get(
            "collection_name", "stage2_import"
        )
        logger.info(f"Loaded {len(dois)} DOIs from {args.input}")
    else:
        # Use config research_questions
        rqs = cfg.get("research_questions", [])
        if not rqs:
            logger.error("No input specified. Use --input, --dois, or set research_questions in config")
            return
        # Run Stage 1 first
        from stage1_elicit_search import run_stage1
        stage1_results = run_stage1(cfg=cfg, logger=logger)
        if not stage1_results:
            logger.error("Stage 1 returned no results")
            return
        # Process first query result
        r = stage1_results[0]
        dois = [p["doi"] for p in r["papers"] if p.get("doi")]
        elicit_papers = r["papers"]
        collection_name = args.collection or r["collection_name"]

    if not dois:
        logger.error("No DOIs to process")
        return

    # Limit papers if specified
    if args.max_papers:
        dois = dois[:args.max_papers]

    logger.info(f"Processing {len(dois)} DOIs → collection: '{collection_name}'")

    # Find parent collection key from config
    parent_key = args.parent_collection
    if not parent_key:
        default_coll = cfg.get("zotero", {}).get("default_collection")
        if default_coll and not args.download_only:
            # Look up parent collection key
            backend = create_zotero_backend(cfg)
            backend.connect()
            parent_key = backend.find_collection(default_coll)
            backend.close()
            if parent_key:
                logger.info(f"Parent collection: '{default_coll}' ({parent_key})")

    # Run
    results = run_stage2(
        cfg=cfg,
        dois=dois,
        collection_name=collection_name,
        parent_collection=parent_key,
        skip_existing=not args.no_skip,
        download_only=args.download_only,
        logger=logger,
        elicit_papers=elicit_papers,
    )

    # Save results
    output_dir = cfg.get("extraction", {}).get("output_dir", "./outputs")
    outfile = save_results(results, collection_name, output_dir)

    # Summary
    logger.info(f"\nStage 2 complete:")
    logger.info(f"  ✓ Saved:   {results['ok']}")
    logger.info(f"  ✗ Failed:  {results['fail']}")
    logger.info(f"  → Skipped: {results['skip']}")
    logger.info(f"  Output:    {outfile}")

    # Save failed DOIs for retry
    if results["failed_dois"]:
        retry_cfg = cfg.get("pipeline", {}).get("retry", {})
        failed_file = Path(retry_cfg.get("failed_doi_file", "./outputs/failed_dois.json"))
        failed_file.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_file, "w") as f:
            json.dump(results["failed_dois"], f, indent=2)
        logger.info(f"  Failed DOIs saved: {failed_file}")

    return results


if __name__ == "__main__":
    main()
