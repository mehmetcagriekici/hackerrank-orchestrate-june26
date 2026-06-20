# Damage Claim Verification System

Processes damage claims (images + user description) and decides whether submitted evidence **supports**, **contradicts**, or provides **not enough information** to evaluate the claim.

## Prerequisites

- Docker + Docker Compose
- Python 3.9+
- `pip install sentence-transformers pillow requests`

## Setup

### 1. Start Ollama

```bash
cd code
docker compose up -d
```

Wait ~30 seconds for the container to be healthy, then pull the model:

```bash
docker exec ollama-server ollama pull llama3.2:1b
```

Verify it's ready:

```bash
curl http://localhost:11434/api/tags
```

### 2. Install Python dependencies

From the repo root:

```bash
pip install sentence-transformers pillow requests
```

The CLIP model (`clip-ViT-L-14`) downloads automatically on first run (~900 MB).

## Running

All commands must be run from the **repo root** (the folder containing `dataset/` and `code/`).

### Production — full dataset

```bash
python run.py
```

Reads `dataset/claims.csv`, writes `output.csv`.

### Development — sample dataset (labeled, for validation)

`dataset/sample_claims.csv` has ground-truth labels in all 14 columns. Run against it to spot-check quality before the full run:

```bash
python - <<'EOF'
import sys
sys.path.insert(0, 'code')
from agent.agent import Agent

agent = Agent(ollama_url="http://localhost:11434", model_name="llama3.2:1b")
agent.run(input_csv='dataset/sample_claims.csv', output_csv='sample_output.csv')
EOF
```

Then compare `sample_output.csv` against `dataset/sample_claims.csv`.

## Output

`output.csv` — 14 columns, one row per claim:

| Column | Values |
|---|---|
| `user_id` | passthrough |
| `image_paths` | passthrough |
| `user_claim` | passthrough |
| `claim_object` | passthrough |
| `evidence_standard_met` | `true` / `false` |
| `evidence_standard_met_reason` | short explanation |
| `risk_flags` | semicolon-separated or `none` |
| `issue_type` | dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown |
| `object_part` | e.g. door, screen, box, unknown |
| `claim_status` | `supported` / `contradicted` / `not_enough_information` |
| `claim_status_justification` | narrative explanation |
| `supporting_image_ids` | semicolon-separated image IDs or `none` |
| `valid_image` | `true` / `false` |
| `severity` | none, low, medium, high, unknown |

## Logs

Appended to `~/hackerrank_orchestrate/log.txt` and printed to stdout.

## Architecture

```
Agent (orchestrator)
  └── Organizer (data assembly)
        └── RAG (CLIP vision layer — clip-ViT-L-14)
  └── LLMReasoner (Ollama — llama3.2:1b)
```

- **RAG** — encodes claim text and images with CLIP, classifies damage type, severity, object/part, quality flags
- **Organizer** — loads user history + evidence requirements, assembles context dict, computes `evidence_standard_met`
- **LLMReasoner** — sends organized context to Ollama, parses and validates the 14-field JSON response
- **Agent** — iterates claims CSV, coordinates the pipeline, writes output CSV

## Tuning

Three thresholds control evidence sensitivity:

| Constant | File | Default | Effect |
|---|---|---|---|
| `QUALITY_THRESHOLDS` | `code/rag/rag.py` | various | CLIP score cutoffs for blurry/dark/cropped flags |
| `CLAIM_SIMILARITY_THRESHOLD` | `code/organizer/organizer.py` | `0.20` | Min claim-image similarity to count a damage detection as valid |
| `EVIDENCE_MATCH_THRESHOLD` | `code/organizer/organizer.py` | `0.22` | Min evidence-requirement match score |

Use `dataset/sample_claims.csv` (ground truth included) to calibrate these before running on the full dataset.
