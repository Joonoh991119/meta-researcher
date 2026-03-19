#!/usr/bin/env python3
"""
Zotero Utilities - Dual Mode (Web API + SQLite)
================================================
Supports both pyzotero (Web API) and direct SQLite manipulation.
Use --zotero-mode api|sqlite to switch.
"""

import hashlib
import logging
import os
import random
import re
import shutil
import sqlite3
import string
import time
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("pipeline.zotero")


# ─── Paper Metadata ────────────────────────────────────────
@dataclass
class PaperMeta:
    """Unified paper metadata structure."""
    doi: str
    title: str = ""
    authors: list = field(default_factory=list)
    date: str = ""
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    abstract: str = ""
    url: str = ""
    item_type: str = "journalArticle"
    publisher: str = ""


# ─── CrossRef Metadata Fetcher ─────────────────────────────
def fetch_crossref_metadata(doi: str, email: str = "", timeout: int = 30) -> PaperMeta:
    """Fetch paper metadata from CrossRef API."""
    encoded = urllib.parse.quote(doi, safe="")
    url = f"https://api.crossref.org/works/{encoded}"
    headers = {"User-Agent": f"research-pipeline/1.0 (mailto:{email})"}

    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return PaperMeta(doi=doi, title=doi, url=f"https://doi.org/{doi}")
        d = r.json()["message"]
    except Exception:
        return PaperMeta(doi=doi, title=doi, url=f"https://doi.org/{doi}")

    authors = [
        {
            "firstName": a.get("given", ""),
            "lastName": a.get("family", ""),
            "creatorType": "author",
        }
        for a in d.get("author", [])
    ]

    dp = (
        d.get("published-print", d.get("published-online", {}))
        .get("date-parts", [[]])[0]
    )
    date_str = "-".join(str(x) for x in dp) if dp else ""
    titles = d.get("title", [])
    title = titles[0] if titles else doi

    ct = d.get("type", "")
    item_type = (
        "bookSection" if "book-chapter" in ct
        else "conferencePaper" if "proceedings" in ct
        else "book" if "book" in ct and "chapter" not in ct
        else "journalArticle"
    )

    jn = d.get("container-title", [])
    abstract = re.sub(r"<[^>]+>", "", d.get("abstract", ""))

    return PaperMeta(
        doi=doi, title=title, authors=authors, date=date_str,
        journal=jn[0] if jn else "", volume=d.get("volume", ""),
        issue=d.get("issue", ""), pages=d.get("page", ""),
        abstract=abstract, url=f"https://doi.org/{doi}",
        item_type=item_type, publisher=d.get("publisher", ""),
    )


# ─── Abstract Base Class ───────────────────────────────────
class ZoteroBackend(ABC):
    """Abstract interface for Zotero storage backends."""

    @abstractmethod
    def connect(self): ...

    @abstractmethod
    def close(self): ...

    @abstractmethod
    def find_collection(self, name: str) -> Optional[str]: ...

    @abstractmethod
    def create_collection(self, name: str, parent_key: Optional[str] = None) -> str: ...

    @abstractmethod
    def has_doi(self, doi: str) -> bool: ...

    @abstractmethod
    def add_item(self, meta: PaperMeta, collection_key: Optional[str] = None) -> str: ...

    @abstractmethod
    def attach_pdf(self, item_key: str, pdf_path: Path) -> str: ...


# ─── Web API Backend (pyzotero) ─────────────────────────────
class ZoteroAPIBackend(ZoteroBackend):
    """Zotero Web API backend using direct HTTP requests."""

    BASE = "https://api.zotero.org"

    # Zotero API item type → field mapping for creators
    ITEM_TYPE_MAP = {
        "journalArticle": "journalArticle",
        "book": "book",
        "bookSection": "bookSection",
        "conferencePaper": "conferencePaper",
        "thesis": "thesis",
        "report": "report",
        "preprint": "preprint",
        "webpage": "webpage",
    }

    def __init__(self, api_key: str, library_id: str, library_type: str = "user"):
        self.api_key = api_key
        self.library_id = library_id
        self.library_type = library_type
        self.session = requests.Session()
        self.session.headers.update({
            "Zotero-API-Key": api_key,
            "Zotero-API-Version": "3",
            "Content-Type": "application/json",
        })
        self._prefix = f"{self.BASE}/{library_type}s/{library_id}"

    def connect(self):
        """Verify API connection."""
        r = self.session.get(f"{self._prefix}/collections", params={"limit": 1})
        r.raise_for_status()
        logger.info("Zotero API connected")

    def close(self):
        """No-op for API backend."""
        pass

    def find_collection(self, name: str) -> Optional[str]:
        """Find collection by name, return key or None."""
        r = self.session.get(f"{self._prefix}/collections")
        r.raise_for_status()
        for col in r.json():
            if col["data"]["name"] == name:
                return col["data"]["key"]
        return None

    def create_collection(self, name: str, parent_key: Optional[str] = None) -> str:
        """Create a new collection, return its key."""
        payload = [{"name": name}]
        if parent_key:
            payload[0]["parentCollection"] = parent_key
        r = self.session.post(f"{self._prefix}/collections", json=payload)
        r.raise_for_status()
        result = r.json()
        # Successful creates return in "successful" dict
        success = result.get("successful", {})
        if success:
            key = list(success.values())[0]["data"]["key"]
            logger.info(f"Created collection '{name}' (key: {key})")
            return key
        # Check for failures
        failed = result.get("failed", {})
        if failed:
            raise RuntimeError(f"Failed to create collection: {failed}")
        raise RuntimeError(f"Unexpected response: {result}")

    def has_doi(self, doi: str) -> bool:
        """Check if DOI already exists in library."""
        r = self.session.get(
            f"{self._prefix}/items",
            params={"q": doi, "qmode": "everything", "limit": 1},
        )
        r.raise_for_status()
        items = r.json()
        for item in items:
            if item.get("data", {}).get("DOI", "").lower() == doi.lower():
                return True
        return False

    def _build_item_data(self, meta: PaperMeta) -> dict:
        """Build Zotero item JSON from PaperMeta."""
        item_type = self.ITEM_TYPE_MAP.get(meta.item_type, "journalArticle")

        creators = []
        for a in meta.authors:
            creators.append({
                "creatorType": a.get("creatorType", "author"),
                "firstName": a.get("firstName", ""),
                "lastName": a.get("lastName", ""),
            })

        data = {
            "itemType": item_type,
            "title": meta.title,
            "creators": creators,
            "DOI": meta.doi,
            "url": meta.url,
            "abstractNote": meta.abstract[:10000],  # Zotero limit
            "date": meta.date,
        }

        # Type-specific fields
        if item_type in ("journalArticle", "conferencePaper"):
            data["publicationTitle"] = meta.journal
            data["volume"] = meta.volume
            data["issue"] = meta.issue
            data["pages"] = meta.pages
        elif item_type in ("book", "bookSection"):
            data["publisher"] = meta.publisher

        return data

    def add_item(self, meta: PaperMeta, collection_key: Optional[str] = None) -> str:
        """Add item to Zotero library, return item key."""
        data = self._build_item_data(meta)
        if collection_key:
            data["collections"] = [collection_key]

        r = self.session.post(f"{self._prefix}/items", json=[data])
        r.raise_for_status()
        result = r.json()

        success = result.get("successful", {})
        if success:
            first = list(success.values())[0]
            key = first.get("key") or first.get("data", {}).get("key")
            if key:
                return key

        failed = result.get("failed", {})
        if failed:
            err = list(failed.values())[0]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            # Retry as journalArticle if item type caused the failure
            if meta.item_type != "journalArticle":
                logger.warning(f"Retrying as journalArticle (was {meta.item_type}): {msg}")
                meta.item_type = "journalArticle"
                return self.add_item(meta, collection_key)
            raise RuntimeError(f"Failed to add item: {msg}")
        raise RuntimeError(f"Unexpected response: {result}")

    def attach_pdf(self, item_key: str, pdf_path: Path) -> str:
        """Upload PDF as child attachment to an item."""
        filename = pdf_path.name
        filesize = pdf_path.stat().st_size
        md5 = hashlib.md5(pdf_path.read_bytes()).hexdigest()

        # Step 1: Create attachment item
        attachment_data = [{
            "itemType": "attachment",
            "parentItem": item_key,
            "linkMode": "imported_file",
            "title": filename,
            "contentType": "application/pdf",
            "filename": filename,
        }]

        r = self.session.post(f"{self._prefix}/items", json=attachment_data)
        r.raise_for_status()
        result = r.json()
        success = result.get("successful", {})
        if not success:
            raise RuntimeError(f"Failed to create attachment: {result}")

        att_key = list(success.values())[0]["key"]

        # Step 2: Get upload authorization
        r = self.session.post(
            f"{self._prefix}/items/{att_key}/file",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "If-None-Match": "*",
            },
            data=f"md5={md5}&filename={filename}&filesize={filesize}&mtime={int(pdf_path.stat().st_mtime * 1000)}",
        )
        r.raise_for_status()
        upload_info = r.json()

        if upload_info.get("exists"):
            logger.debug(f"File already exists on server: {filename}")
            return att_key

        # Step 3: Upload file
        upload_url = upload_info["url"]
        upload_prefix = upload_info.get("prefix", "").encode()
        upload_suffix = upload_info.get("suffix", "").encode()
        content_type = upload_info.get("contentType", "application/pdf")

        file_data = upload_prefix + pdf_path.read_bytes() + upload_suffix

        r = requests.post(
            upload_url,
            data=file_data,
            headers={"Content-Type": content_type},
            timeout=120,
        )
        r.raise_for_status()

        # Step 4: Register upload
        upload_key = upload_info.get("uploadKey", "")
        r = self.session.post(
            f"{self._prefix}/items/{att_key}/file",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "If-None-Match": "*",
            },
            data=f"upload={upload_key}",
        )
        r.raise_for_status()
        logger.debug(f"PDF uploaded: {filename} → {att_key}")
        return att_key


# ─── SQLite Backend (Direct DB) ─────────────────────────────
class ZoteroSQLiteBackend(ZoteroBackend):
    """Direct Zotero SQLite database manipulation (requires Zotero to be closed)."""

    ATT_TYPE = 3
    LINK_MODE = 0
    LIB_ID = 1
    TYPES = {
        "journalArticle": 22, "book": 2, "bookSection": 5,
        "conferencePaper": 27, "thesis": 7, "report": 13,
        "preprint": 39, "webpage": 33,
    }

    def __init__(self, zotero_dir: str):
        self.zdir = Path(zotero_dir).expanduser()
        self.db_path = self.zdir / "zotero.sqlite"
        self.storage_dir = self.zdir / "storage"
        self.conn: Optional[sqlite3.Connection] = None
        self._keys: set = set()
        self._fields: dict = {}
        self._creator_types: dict = {}

    def connect(self):
        if not self.db_path.exists():
            raise FileNotFoundError(f"Zotero DB not found: {self.db_path}")

        # Backup
        backup = self.zdir / f"zotero_backup_{int(time.time())}.sqlite"
        shutil.copy2(self.db_path, backup)
        logger.info(f"DB backup: {backup.name}")

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        c = self.conn.cursor()
        c.execute("SELECT key FROM items")
        self._keys = {r[0] for r in c.fetchall()}
        c.execute("SELECT fieldName, fieldID FROM fields")
        self._fields = {r[0]: r[1] for r in c.fetchall()}
        c.execute("SELECT creatorType, creatorTypeID FROM creatorTypes")
        self._creator_types = {r[0]: r[1] for r in c.fetchall()}
        logger.info("SQLite backend connected")

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def _gen_key(self) -> str:
        chars = string.ascii_uppercase + string.digits
        while True:
            k = "".join(random.choices(chars, k=8))
            if k not in self._keys:
                self._keys.add(k)
                return k

    def _now(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    def _get_value_id(self, cursor, value: str) -> int:
        cursor.execute("SELECT valueID FROM itemDataValues WHERE value=?", (value,))
        r = cursor.fetchone()
        if r:
            return r[0]
        cursor.execute("INSERT INTO itemDataValues (value) VALUES (?)", (value,))
        return cursor.lastrowid

    def find_collection(self, name: str) -> Optional[str]:
        c = self.conn.cursor()
        c.execute(
            "SELECT key FROM collections WHERE collectionName=?", (name,)
        )
        r = c.fetchone()
        return r[0] if r else None

    def create_collection(self, name: str, parent_key: Optional[str] = None) -> str:
        c = self.conn.cursor()
        key = self._gen_key()
        parent_id = None
        if parent_key:
            c.execute("SELECT collectionID FROM collections WHERE key=?", (parent_key,))
            r = c.fetchone()
            parent_id = r[0] if r else None

        c.execute(
            "INSERT INTO collections "
            "(collectionName, parentCollectionID, libraryID, key, clientDateModified, version, synced) "
            "VALUES (?,?,?,?,?,0,0)",
            (name, parent_id, self.LIB_ID, key, self._now()),
        )
        self.conn.commit()
        logger.info(f"Created collection '{name}' (key: {key})")
        return key

    def has_doi(self, doi: str) -> bool:
        fid = self._fields.get("DOI")
        if not fid:
            return False
        c = self.conn.cursor()
        c.execute(
            "SELECT 1 FROM items i "
            "JOIN itemData d ON i.itemID=d.itemID "
            "JOIN itemDataValues v ON d.valueID=v.valueID "
            "WHERE d.fieldID=? AND v.value=?",
            (fid, doi),
        )
        return c.fetchone() is not None

    def add_item(self, meta: PaperMeta, collection_key: Optional[str] = None) -> str:
        c = self.conn.cursor()
        key = self._gen_key()
        ts = self._now()
        type_id = self.TYPES.get(meta.item_type, 22)

        c.execute(
            "INSERT INTO items "
            "(itemTypeID,libraryID,key,dateAdded,dateModified,clientDateModified,version,synced) "
            "VALUES (?,?,?,?,?,?,0,0)",
            (type_id, self.LIB_ID, key, ts, ts, ts),
        )
        item_id = c.lastrowid

        field_map = {
            "title": meta.title, "abstractNote": meta.abstract, "date": meta.date,
            "DOI": meta.doi, "url": meta.url, "volume": meta.volume,
            "issue": meta.issue, "pages": meta.pages, "publicationTitle": meta.journal,
        }
        for fn, fv in field_map.items():
            if not fv:
                continue
            fid = self._fields.get(fn)
            if not fid:
                continue
            c.execute(
                "INSERT OR IGNORE INTO itemData VALUES (?,?,?)",
                (item_id, fid, self._get_value_id(c, fv)),
            )

        for idx, a in enumerate(meta.authors):
            ct_id = self._creator_types.get(a.get("creatorType", "author"), 1)
            fn, ln = a.get("firstName", ""), a.get("lastName", "")
            c.execute(
                "SELECT creatorID FROM creators WHERE firstName=? AND lastName=?",
                (fn, ln),
            )
            r = c.fetchone()
            if r:
                creator_id = r[0]
            else:
                c.execute(
                    "INSERT INTO creators (firstName, lastName) VALUES (?,?)",
                    (fn, ln),
                )
                creator_id = c.lastrowid
            c.execute(
                "INSERT INTO itemCreators VALUES (?,?,?,?)",
                (item_id, creator_id, ct_id, idx),
            )

        if collection_key:
            c.execute(
                "SELECT collectionID FROM collections WHERE key=?",
                (collection_key,),
            )
            r = c.fetchone()
            if r:
                c.execute(
                    "INSERT OR IGNORE INTO collectionItems VALUES (?,?,0)",
                    (r[0], item_id),
                )

        self.conn.commit()
        return key

    def attach_pdf(self, item_key: str, pdf_path: Path) -> str:
        c = self.conn.cursor()
        c.execute("SELECT itemID FROM items WHERE key=?", (item_key,))
        r = c.fetchone()
        if not r:
            return item_key
        parent_id = r[0]

        # Check existing attachment
        c.execute(
            "SELECT 1 FROM items i JOIN itemAttachments a ON i.itemID=a.itemID "
            "WHERE a.parentItemID=? AND a.contentType='application/pdf'",
            (parent_id,),
        )
        if c.fetchone():
            return item_key

        att_key = self._gen_key()
        ts = self._now()
        att_dir = self.storage_dir / att_key
        att_dir.mkdir(parents=True, exist_ok=True)

        filename = pdf_path.name
        shutil.copy2(pdf_path, att_dir / filename)
        md5 = hashlib.md5(pdf_path.read_bytes()).hexdigest()
        mtime = int(pdf_path.stat().st_mtime * 1000)

        c.execute(
            "INSERT INTO items "
            "(itemTypeID,libraryID,key,dateAdded,dateModified,clientDateModified,version,synced) "
            "VALUES (?,?,?,?,?,?,0,0)",
            (self.ATT_TYPE, self.LIB_ID, att_key, ts, ts, ts),
        )
        att_id = c.lastrowid
        c.execute(
            "INSERT INTO itemAttachments "
            "(itemID,parentItemID,linkMode,contentType,path,syncState,storageModTime,storageHash) "
            "VALUES (?,?,?,?,?,0,?,?)",
            (att_id, parent_id, self.LINK_MODE, "application/pdf", f"storage:{filename}", mtime, md5),
        )

        title_fid = self._fields.get("title")
        if title_fid:
            c.execute(
                "INSERT OR IGNORE INTO itemData VALUES (?,?,?)",
                (att_id, title_fid, self._get_value_id(c, filename)),
            )

        self.conn.commit()
        return att_key


# ─── Factory ────────────────────────────────────────────────
def create_zotero_backend(cfg: dict) -> ZoteroBackend:
    """Create appropriate Zotero backend based on config."""
    mode = cfg.get("pipeline", {}).get("zotero_mode", "api")
    zcfg = cfg["zotero"]

    if mode == "api":
        return ZoteroAPIBackend(
            api_key=zcfg["api_key"],
            library_id=zcfg["library_id"],
            library_type=zcfg.get("library_type", "user"),
        )
    elif mode == "sqlite":
        return ZoteroSQLiteBackend(zotero_dir=zcfg.get("data_dir", "~/Zotero"))
    else:
        raise ValueError(f"Unknown zotero_mode: {mode}")
