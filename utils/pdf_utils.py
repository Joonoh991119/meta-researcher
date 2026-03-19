#!/usr/bin/env python3
"""
PDF Download Utilities
======================
3-tier PDF download strategy: Unpaywall → Sci-Hub → Direct Publisher.
Refactored from doi2zotero_app.py into reusable functions.
"""

import logging
import re
import time
import urllib.parse
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("pipeline.pdf")


# ─── HTTP Session ───────────────────────────────────────────
def create_session() -> requests.Session:
    """Browser-mimicking session for publisher sites."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.8,*/*;q=0.7",
    })
    return s


_SESSION = create_session()


# ─── Validation ─────────────────────────────────────────────
def validate_pdf(path: Path) -> bool:
    """Check if file is a valid PDF (header + minimum size)."""
    if not path.exists() or path.stat().st_size < 5000:
        return False
    with open(path, "rb") as f:
        return f.read(4) == b"%PDF"


def _download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a URL to file, validate as PDF."""
    try:
        r = _SESSION.get(url, timeout=timeout, stream=True, allow_redirects=True)
        if r.status_code != 200:
            return False
        ct = r.headers.get("Content-Type", "")
        if "html" in ct.lower() and "pdf" not in ct.lower():
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return validate_pdf(dest)
    except Exception:
        return False


# ─── Strategy 1: Unpaywall (Legal Open Access) ─────────────
def try_unpaywall(doi: str, dest: Path, email: str, timeout: int = 30) -> bool:
    """Try Unpaywall API — iterate ALL OA locations (publisher + repository)."""
    try:
        encoded = urllib.parse.quote(doi, safe="")
        r = requests.get(
            f"https://api.unpaywall.org/v2/{encoded}?email={email}",
            timeout=timeout,
        )
        if r.status_code != 200:
            return False
        data = r.json()

        # Try all OA locations, not just best — repositories (PMC) often
        # succeed when publisher sites (PNAS, Royal Society) block bots
        for loc in data.get("oa_locations", []):
            url = loc.get("url_for_pdf") or loc.get("url")
            if not url:
                continue
            if _download_file(url, dest, timeout):
                return True

        return False
    except Exception as e:
        logger.debug(f"Unpaywall failed for {doi}: {e}")
        return False


# ─── Strategy 1b: EuropePMC (fallback for PMC papers) ──────
def try_europepmc(doi: str, dest: Path, timeout: int = 30) -> bool:
    """Try EuropePMC for papers with PMCID — bypasses some bot blocks."""
    try:
        # Look up PMCID via EuropePMC API
        r = requests.get(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query=DOI:{doi}&format=json&resultType=core",
            timeout=timeout,
        )
        if r.status_code != 200:
            return False
        results = r.json().get("resultList", {}).get("result", [])
        if not results:
            return False
        pmcid = results[0].get("pmcid", "")
        if not pmcid:
            return False

        # EuropePMC render endpoint (often less aggressive bot blocking)
        pdf_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
        if _download_file(pdf_url, dest, timeout):
            return True

        # Also try direct PMC PDF
        pdf_url2 = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"
        return _download_file(pdf_url2, dest, timeout)
    except Exception as e:
        logger.debug(f"EuropePMC failed for {doi}: {e}")
        return False


# ─── Strategy 2: Sci-Hub ───────────────────────────────────
def _scihub_session() -> requests.Session:
    """Sci-Hub requires a clean session without extra Accept headers."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        ),
    })
    return s


_SH_SESSION = _scihub_session()


def try_scihub(
    doi: str, dest: Path, mirrors: list[str], timeout: int = 30
) -> bool:
    """Try Sci-Hub mirrors for PDF with comprehensive URL extraction."""
    for mirror in mirrors:
        try:
            r = _SH_SESSION.get(f"{mirror}/{doi}", timeout=timeout)
            if r.status_code != 200:
                continue

            # Multiple PDF URL extraction patterns (ordered by reliability)
            # Note: Sci-Hub HTML uses spaces around = in attributes (e.g. data = "url")
            patterns = [
                # <object data="/storage/.../paper.pdf"> (with optional spaces)
                r'<object[^>]*data\s*=\s*["\']([^"\']*\.pdf[^"\']*)["\']',
                # citation_pdf_url meta tag (with optional spaces)
                r'citation_pdf_url["\'][^>]*content\s*=\s*["\']([^"\']+)["\']',
                r'content\s*=\s*["\']([^"\']+\.pdf[^"\']*)["\'][^>]*citation_pdf_url',
                # iframe/embed src
                r'(?:iframe|embed)[^>]*src\s*=\s*["\']([^"\']*\.pdf[^"\']*)["\']',
                # Direct PDF links
                r'(https?://[^\s"\'<>]+\.pdf[^\s"\'<>]*)',
            ]
            for pat in patterns:
                match = re.search(pat, r.text, re.I)
                if match:
                    url = match.group(1).split("#")[0]  # Remove URL fragment
                    if url.startswith("//"):
                        url = "https:" + url
                    elif url.startswith("/"):
                        url = mirror + url
                    if _download_file(url, dest, timeout):
                        return True
        except Exception:
            continue
    return False


# ─── Strategy 3: Direct Publisher ───────────────────────────
def _publisher_patterns(url: str) -> list[str]:
    """Generate publisher-specific PDF URL patterns."""
    patterns = []
    u = url.lower()

    if "sciencedirect.com" in u:
        pii = re.search(r"/pii/(\w+)", u)
        if pii:
            patterns.append(
                f"https://www.sciencedirect.com/science/article/pii/{pii.group(1)}/pdfft"
            )
    if "springer.com" in u or "nature.com" in u:
        patterns.append(url.replace("/article/", "/content/pdf/") + ".pdf")
    if "wiley.com" in u:
        patterns.append(url.replace("/abs/", "/pdfdirect/") + "?download=true")
    if "plos" in u:
        match = re.search(r"article\?id=([\d.]+/[\w.]+)", u)
        if match:
            patterns.append(
                f"https://journals.plos.org/plosone/article/file?id={match.group(1)}&type=printable"
            )
    if "pnas.org" in u:
        # PNAS: /doi/full/10.xxx → /doi/pdf/10.xxx
        patterns.append(url.replace("/doi/full/", "/doi/pdf/"))
        patterns.append(url.replace("/doi/abs/", "/doi/pdf/"))
        patterns.append(url.rstrip("/") + ".full.pdf")
    if "academic.oup.com" in u:
        patterns.append(url.rstrip("/") + ".full.pdf")
    if "royalsocietypublishing.org" in u:
        # Royal Society: /doi/full/10.xxx → /doi/pdf/10.xxx
        patterns.append(url.replace("/doi/full/", "/doi/pdf/"))
        patterns.append(url.replace("/doi/abs/", "/doi/pdf/"))
    if "frontiersin.org" in u or "mdpi.com" in u:
        patterns.append(url.rstrip("/") + "/pdf")

    return patterns


def try_direct_publisher(doi: str, dest: Path, timeout: int = 30) -> bool:
    """Try direct publisher download via DOI redirect + meta tags."""
    try:
        r = _SESSION.get(
            f"https://doi.org/{doi}", timeout=timeout, allow_redirects=True
        )
        if r.status_code != 200:
            return False
        final_url = r.url

        # Handle Elsevier redirect
        if "linkinghub.elsevier.com" in final_url:
            match = re.search(
                r'href=["\']([^"\']*sciencedirect[^"\']*)["\']', r.text
            )
            if match:
                r = _SESSION.get(match.group(1), timeout=timeout, allow_redirects=True)
                final_url = r.url

        # Try citation_pdf_url meta tag
        for pat in [
            r'<meta[^>]*name=["\']citation_pdf_url["\'][^>]*content=["\']([^"\']+)["\']',
            r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*name=["\']citation_pdf_url["\']',
        ]:
            match = re.search(pat, r.text, re.I)
            if match and _download_file(match.group(1), dest, timeout):
                return True

        # Try publisher-specific patterns
        for url in _publisher_patterns(final_url):
            if _download_file(url, dest, timeout):
                return True

        return False
    except Exception as e:
        logger.debug(f"Direct publisher failed for {doi}: {e}")
        return False


# ─── Main Download Function ────────────────────────────────
def download_pdf(
    doi: str,
    download_dir: Path,
    email: str = "",
    strategies: list[str] | None = None,
    scihub_mirrors: list[str] | None = None,
    timeout: int = 30,
    delay: float = 0.5,
) -> tuple[bool, Optional[Path], str]:
    """
    Download PDF for a DOI using configured strategies.

    Returns:
        (success, file_path, source_name)
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w\-.]", "_", doi) + ".pdf"
    dest = download_dir / safe_name

    if strategies is None:
        strategies = ["unpaywall", "europepmc", "scihub", "publisher"]
    if scihub_mirrors is None:
        scihub_mirrors = ["https://sci-hub.kr", "https://sci-hub.se", "https://sci-hub.st", "https://sci-hub.ru"]

    strategy_map = {
        "unpaywall": lambda: try_unpaywall(doi, dest, email, timeout),
        "europepmc": lambda: try_europepmc(doi, dest, timeout),
        "scihub": lambda: try_scihub(doi, dest, scihub_mirrors, timeout),
        "publisher": lambda: try_direct_publisher(doi, dest, timeout),
    }

    for name in strategies:
        fn = strategy_map.get(name)
        if fn and fn():
            logger.info(f"  PDF downloaded via {name}")
            return (True, dest, name)
        time.sleep(delay)

    # Clean up failed partial download
    if dest.exists():
        dest.unlink()
    return (False, None, "")
