# Research Paper Pipeline

Automated research pipeline: Elicit API search → PDF download → Zotero storage → Embedding memory layer.

## Architecture

```
[Research Question]
        │
        ▼
┌─── Stage 1: Elicit Search ──────┐
│  /api/v1/search → DOI list      │
└──────────┬──────────────────────┘
           │
           ▼
┌─── Stage 2: DOI → Zotero ───────┐
│  PDF download (3-tier fallback)  │
│  → Zotero subcollection         │
└──────────┬──────────────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─ 3a: Memory ─┐  ┌─ 3b: Reports ──┐
│ Nemotron embed│  │ Elicit Reports │
│ → ChromaDB    │  │ (on-demand)    │
└───────────────┘  └────────────────┘
```

## Setup

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml
# Edit config.yaml with your API keys
```

## Usage

```bash
# Full pipeline (Stage 1 → 2 → 3a)
python pipeline.py

# With custom query
python pipeline.py --query "Bayesian inference in VWM" --max-papers 20

# Run specific stages
python pipeline.py --stage 1           # Elicit search only
python pipeline.py --stage 2           # DOI → Zotero only
python pipeline.py --stage 3a          # Embedding only
python pipeline.py --stage 3b          # Elicit report only
python pipeline.py --stage 1,2         # Search + Zotero
python pipeline.py --stage all         # Everything including reports

# Dry run
python pipeline.py --dry-run

# Individual stage scripts
python stage1_elicit_search.py --query "..." --max-results 50
python stage2_doi2zotero.py --input stage1_output.json --zotero-mode api
python stage3a_embedding.py --search "query text"    # semantic search
python stage3a_embedding.py --stats                  # show memory stats
python stage3b_elicit_reports.py --question "..."     # on-demand report
```

## Project Structure

```
research_pipeline/
├── pipeline.py              # Orchestrator
├── stage1_elicit_search.py  # Elicit API → DOI list
├── stage2_doi2zotero.py     # DOI → PDF → Zotero
├── stage3a_embedding.py     # PDF → Nemotron embed → ChromaDB
├── stage3b_elicit_reports.py# Elicit Reports (on-demand)
├── utils/
│   ├── pdf_utils.py         # 3-tier PDF download
│   └── zotero_utils.py      # Dual backend (API + SQLite)
├── config.example.yaml      # Config template
├── requirements.txt
└── README.md
```

## API Keys Required

| Service | Where to get | Used for |
|---------|-------------|----------|
| Elicit | elicit.com → Settings → API | Paper search + Reports |
| OpenRouter | openrouter.ai → API Keys | Nemotron embedding |
| Zotero | zotero.org/settings/keys | Library storage |

## License

MIT
