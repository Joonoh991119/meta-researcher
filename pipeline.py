#!/usr/bin/env python3
"""
Research Paper Pipeline - Orchestrator
=======================================
3개 스테이지를 순차 실행하는 메인 진입점.

Usage:
    # 전체 파이프라인 실행 (config.yaml의 research_questions 사용)
    python pipeline.py

    # 특정 query로 전체 파이프라인
    python pipeline.py --query "Bayesian inference in visual working memory"

    # 특정 스테이지만 실행
    python pipeline.py --stage 1          # Elicit search만
    python pipeline.py --stage 2          # DOI → Zotero만
    python pipeline.py --stage 3a         # Embedding만
    python pipeline.py --stage 3b         # Elicit report만

    # Stage 1 → 2만 실행 (embedding 제외)
    python pipeline.py --stage 1,2

    # 최대 논문 수 제한
    python pipeline.py --max-papers 10

    # dry-run (실제 실행 없이 계획만 출력)
    python pipeline.py --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict) -> logging.Logger:
    log_dir = Path(cfg.get("pipeline", {}).get("log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline")
    logger.setLevel(getattr(logging, cfg.get("pipeline", {}).get("log_level", "INFO")))
    fh = logging.FileHandler(log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    return logger


def run_pipeline(
    cfg: dict,
    stages: list[str],
    query: str | None = None,
    collection: str | None = None,
    max_papers: int | None = None,
    dry_run: bool = False,
    logger: logging.Logger | None = None,
):
    log = logger or logging.getLogger("pipeline")
    log.info("=" * 60)
    log.info("Research Paper Pipeline")
    log.info("=" * 60)

    # Build query list
    if query:
        queries = [{"query": query, "collection_name": collection or "pipeline_run"}]
    else:
        queries = cfg.get("research_questions", [])

    if not queries:
        log.error("No research questions. Use --query or set research_questions in config.yaml")
        return

    log.info(f"Stages: {stages}")
    log.info(f"Queries: {len(queries)}")
    for q in queries:
        log.info(f"  - {q['query'][:70]}...")

    if dry_run:
        log.info("\n[DRY RUN] Would execute:")
        if "1" in stages:
            log.info(f"  Stage 1: Search {len(queries)} queries via Elicit API")
        if "2" in stages:
            log.info(f"  Stage 2: Download PDFs + save to Zotero (max: {max_papers or 'all'})")
        if "3a" in stages:
            log.info(f"  Stage 3a: Embed papers → ChromaDB memory layer")
        if "3b" in stages:
            log.info(f"  Stage 3b: Generate Elicit Reports (on-demand)")
        return

    all_results = {}

    # ─── Stage 1: Elicit Search ─────────────────────────────
    if "1" in stages:
        log.info("\n" + "─" * 60)
        log.info("STAGE 1: Elicit API Search")
        log.info("─" * 60)

        from stage1_elicit_search import run_stage1, build_filters

        # Build filters from config
        filters = {}
        cfg_filters = cfg.get("elicit", {}).get("filters", {})
        if cfg_filters.get("year_from"):
            filters["minYear"] = cfg_filters["year_from"]
        if cfg_filters.get("year_to"):
            filters["maxYear"] = cfg_filters["year_to"]

        stage1_results = run_stage1(
            cfg=cfg,
            queries=queries,
            logger=log,
            filters=filters,
            max_results=max_papers or cfg.get("elicit", {}).get("max_results_per_query", 100),
        )
        all_results["stage1"] = stage1_results

        if not stage1_results:
            log.error("Stage 1 returned no results. Stopping.")
            return all_results

    # ─── Stage 2: DOI → PDF → Zotero ───────────────────────
    if "2" in stages:
        log.info("\n" + "─" * 60)
        log.info("STAGE 2: DOI → PDF → Zotero")
        log.info("─" * 60)

        from stage2_doi2zotero import run_stage2
        from utils.zotero_utils import create_zotero_backend

        # Get DOIs from Stage 1 or config
        s1 = all_results.get("stage1", [])
        stage2_results = []

        for s1_result in s1:
            papers = s1_result["papers"]
            dois = [p["doi"] for p in papers if p.get("doi")]
            coll_name = s1_result["collection_name"]

            if max_papers:
                dois = dois[:max_papers]

            log.info(f"Processing {len(dois)} DOIs → collection: '{coll_name}'")

            # Find parent collection
            parent_key = None
            default_coll = cfg.get("zotero", {}).get("default_collection")
            if default_coll:
                backend = create_zotero_backend(cfg)
                backend.connect()
                parent_key = backend.find_collection(default_coll)
                backend.close()

            result = run_stage2(
                cfg=cfg,
                dois=dois,
                collection_name=coll_name,
                parent_collection=parent_key,
                logger=log,
                elicit_papers=papers,
            )
            stage2_results.append({"collection": coll_name, "result": result})

        all_results["stage2"] = stage2_results

    # ─── Stage 3a: Embedding → Memory Layer ─────────────────
    if "3a" in stages:
        log.info("\n" + "─" * 60)
        log.info("STAGE 3a: Embedding → Memory Layer")
        log.info("─" * 60)

        from stage3a_embedding import embed_papers
        import re

        pdf_dir = Path(cfg.get("pdf_download", {}).get("temp_dir", "/tmp/research_pipeline_pdfs"))
        pdf_paths = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []

        # Collect paper metadata from Stage 2
        paper_metadata = None
        s2 = all_results.get("stage2", [])
        if s2:
            paper_metadata = []
            for s2r in s2:
                paper_metadata.extend(s2r["result"].get("items", []))

        if pdf_paths:
            log.info(f"Embedding {len(pdf_paths)} PDFs...")
            result = embed_papers(
                cfg=cfg,
                pdf_paths=pdf_paths,
                paper_metadata=paper_metadata,
                logger=log,
            )
            all_results["stage3a"] = result
        else:
            log.warning("No PDFs found for embedding")

    # ─── Stage 3b: Elicit Reports (on-demand) ───────────────
    if "3b" in stages:
        log.info("\n" + "─" * 60)
        log.info("STAGE 3b: Elicit Reports")
        log.info("─" * 60)

        from stage3b_elicit_reports import create_and_wait, save_report

        output_dir = cfg.get("extraction", {}).get("output_dir", "./outputs")
        reports = []

        for q in queries:
            try:
                report = create_and_wait(cfg, q["query"], log)
                outfile = save_report(report, q["query"], output_dir)
                log.info(f"  ✓ Report saved: {outfile}")
                reports.append({"query": q["query"], "file": str(outfile)})
            except Exception as e:
                log.error(f"  ✗ Report failed: {e}")

        all_results["stage3b"] = reports

    # ─── Summary ────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)

    if "stage1" in all_results:
        total = sum(len(r["papers"]) for r in all_results["stage1"])
        log.info(f"  Stage 1: {total} papers found")
    if "stage2" in all_results:
        for s2r in all_results["stage2"]:
            r = s2r["result"]
            log.info(f"  Stage 2 [{s2r['collection']}]: {r['ok']} saved, {r['fail']} failed, {r['skip']} skipped")
    if "stage3a" in all_results:
        r = all_results["stage3a"]
        log.info(f"  Stage 3a: {r['embedded']} embedded, {r['skipped']} skipped")
    if "stage3b" in all_results:
        log.info(f"  Stage 3b: {len(all_results['stage3b'])} reports generated")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Research Paper Pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--query", "-q", help="Research question (overrides config)")
    parser.add_argument("--collection", "-c", help="Collection name")
    parser.add_argument("--stage", help="Stages to run: 1,2,3a,3b or 'all' (default: 1,2,3a)")
    parser.add_argument("--max-papers", "-n", type=int, help="Max papers to process")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg)

    # Parse stages
    if args.stage:
        if args.stage == "all":
            stages = ["1", "2", "3a", "3b"]
        else:
            stages = [s.strip() for s in args.stage.split(",")]
    else:
        stages = ["1", "2", "3a"]  # Default: no report generation

    run_pipeline(
        cfg=cfg,
        stages=stages,
        query=args.query,
        collection=args.collection,
        max_papers=args.max_papers,
        dry_run=args.dry_run,
        logger=logger,
    )


if __name__ == "__main__":
    main()
