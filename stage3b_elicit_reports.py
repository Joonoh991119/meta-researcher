#!/usr/bin/env python3
"""
Stage 3b: Elicit Reports API (On-Demand)
==========================================
사용자 요청 시 Elicit Reports API를 호출하여 research question에 대한
자동 분석 보고서를 생성한다.

Usage:
    # Research question으로 report 생성
    python stage3b_elicit_reports.py --question "How does efficient coding explain VWM set size effects?"

    # config의 research_questions 사용
    python stage3b_elicit_reports.py

    # 기존 report 상태 확인
    python stage3b_elicit_reports.py --report-id <id>

    # 모든 reports 목록 조회
    python stage3b_elicit_reports.py --list
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml


# ─── Config / Logging ──────────────────────────────────────
def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict) -> logging.Logger:
    log_dir = Path(cfg.get("pipeline", {}).get("log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage3b")
    logger.setLevel(getattr(logging, cfg.get("pipeline", {}).get("log_level", "INFO")))
    fh = logging.FileHandler(log_dir / f"stage3b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    return logger


# ─── Elicit Reports Client ─────────────────────────────────
class ElicitReportsClient:
    """Elicit /v1/reports API 클라이언트."""

    def __init__(self, api_key: str, base_url: str = "https://elicit.com/api/v1"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def create_report(
        self,
        research_question: str,
        max_search_papers: int = 50,
        max_extract_papers: int = 10,
    ) -> dict:
        """Report 생성 요청. 즉시 reportId 반환."""
        resp = self.session.post(
            f"{self.base_url}/reports",
            json={
                "researchQuestion": research_question,
                "maxSearchPapers": max_search_papers,
                "maxExtractPapers": max_extract_papers,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_report(self, report_id: str, include_body: bool = False) -> dict:
        """Report 상태/내용 조회."""
        url = f"{self.base_url}/reports/{report_id}"
        if include_body:
            url += "?include=reportBody"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def list_reports(self, limit: int = 20, cursor: str | None = None) -> dict:
        """Report 목록 조회."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        resp = self.session.get(f"{self.base_url}/reports", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def wait_for_report(
        self,
        report_id: str,
        poll_interval: int = 10,
        max_wait: int = 600,
        logger: logging.Logger | None = None,
    ) -> dict:
        """Report 완료까지 polling (5~15분 소요)."""
        log = logger or logging.getLogger("stage3b")
        elapsed = 0
        while elapsed < max_wait:
            result = self.get_report(report_id, include_body=True)
            status = result.get("status", "unknown")
            if status == "completed":
                log.info(f"  Report completed in {elapsed}s")
                return result
            elif status == "failed":
                raise RuntimeError(f"Report failed: {result}")
            log.info(f"  Report status: {status} ({elapsed}s / {max_wait}s max)")
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise TimeoutError(f"Report did not complete within {max_wait}s")


# ─── Core Functions ─────────────────────────────────────────
def create_and_wait(
    cfg: dict,
    question: str,
    logger: logging.Logger | None = None,
) -> dict:
    """Report를 생성하고 완료될 때까지 대기한다."""
    log = logger or logging.getLogger("stage3b")
    elicit_cfg = cfg["elicit"]
    reports_cfg = cfg.get("extraction", {}).get("elicit_reports", {})

    client = ElicitReportsClient(elicit_cfg["api_key"], elicit_cfg.get("base_url", "https://elicit.com/api/v1"))

    log.info(f"Creating report: {question[:80]}...")
    result = client.create_report(
        research_question=question,
        max_search_papers=elicit_cfg.get("max_results_per_query", 50),
    )

    report_id = result.get("reportId", result.get("id", ""))
    status = result.get("status", "unknown")
    url = result.get("url", "")
    log.info(f"  Report ID: {report_id}")
    log.info(f"  Status: {status}")
    if url:
        log.info(f"  URL: {url}")

    if status == "completed":
        return client.get_report(report_id, include_body=True)

    # Poll for completion
    poll_interval = reports_cfg.get("poll_interval", 10)
    max_wait = reports_cfg.get("max_wait", 600)
    return client.wait_for_report(report_id, poll_interval, max_wait, log)


def save_report(report: dict, question: str, output_dir: str = "./outputs") -> Path:
    """Report를 파일로 저장한다."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_ " else "" for c in question[:40]).strip().replace(" ", "_")
    filepath = out / f"report_{safe}_{ts}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Also save markdown body separately if available
    body = report.get("reportBody", "")
    if body:
        md_path = out / f"report_{safe}_{ts}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(body)

    return filepath


# ─── Main ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 3b: Elicit Reports (On-Demand)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--question", "-q", help="Research question for report")
    parser.add_argument("--report-id", help="Check status of existing report")
    parser.add_argument("--list", action="store_true", help="List all reports")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for completion")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg)
    elicit_cfg = cfg["elicit"]
    client = ElicitReportsClient(elicit_cfg["api_key"], elicit_cfg.get("base_url", "https://elicit.com/api/v1"))

    # List mode
    if args.list:
        result = client.list_reports()
        reports = result.get("reports", result.get("data", []))
        print(f"\nElicit Reports ({len(reports)} found):")
        for r in reports:
            print(f"  [{r.get('status', '?')}] {r.get('reportId', r.get('id', '?'))}")
            print(f"    Q: {r.get('researchQuestion', '?')[:70]}")
            if r.get("url"):
                print(f"    URL: {r['url']}")
        return

    # Check existing report
    if args.report_id:
        result = client.get_report(args.report_id, include_body=True)
        print(f"\nReport: {args.report_id}")
        print(f"Status: {result.get('status', '?')}")
        if result.get("reportBody"):
            print(f"Body length: {len(result['reportBody'])} chars")
            print(f"\n{result['reportBody'][:500]}...")
        return

    # Create report
    if args.question:
        questions = [args.question]
    else:
        questions = [q["query"] for q in cfg.get("research_questions", [])]

    if not questions:
        logger.error("No research question provided")
        return

    output_dir = cfg.get("extraction", {}).get("output_dir", "./outputs")

    for question in questions:
        logger.info(f"Processing: {question[:80]}...")

        if args.no_wait:
            result = client.create_report(question)
            logger.info(f"  Report created: {result.get('reportId', result.get('id', ''))}")
            logger.info(f"  Status: {result.get('status')}")
            if result.get("url"):
                logger.info(f"  URL: {result['url']}")
        else:
            try:
                report = create_and_wait(cfg, question, logger)
                outfile = save_report(report, question, output_dir)
                logger.info(f"  ✓ Report saved: {outfile}")
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")


if __name__ == "__main__":
    main()
