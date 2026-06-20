# HackerRank Orchestrate — Multi-Modal Evidence Review
## Project Context for External LLMs

---

## Project Overview

Build a damage claim verification system for a 24-hour hackathon. The system reads damage claims (images + user description) and decides whether submitted evidence supports, contradicts, or is insufficient to evaluate the claim.

**Challenge:** Process `dataset/claims.csv`, produce `output.csv` with structured predictions.

**Constraints:**
- 24-hour timeline
- Must handle 3 object types: car, laptop, package
- Must use image evidence as primary source of truth
- Must consider user history and evidence requirements
- All predictions must be reproducible and evaluable

---

## Architecture

```
Agent (orchestrator)
  ↓
  RAG (vision/structure layer)
    - CLIP multimodal embeddings
    - Per-image classifiers (heuristics)
    - Returns: object, part, damage, severity, quality flags, similarities
  ↓
  Organizer (data assembly layer)
    - Loads RAG results + user history + evidence requirements
    - Assembles clean context dict
    - Computes: evidence_standard_met, valid_image, aggregated flags
    - NO reasoning; just plumbing
  ↓
  LLM (reasoning layer)
    - Receives organized context
    - Produces all 14 output fields
    - Handles: claim_status, justification, risk_flags, supporting_image_ids
  ↓
  CSV Output (14 columns, one row per claim)
```

---

## What's Built

### 1. RAG (rag.py) — COMPLETE

**Class:** `Rag`

**Purpose:** Extract structured information from images using CLIP embeddings + heuristic classifiers.

**Public method:**
```python
def process(self, claim_text: str, image_paths: list, claim_object: str, evidence_requirements: dict):
    """
    Returns dict:
    {
      'claim_similarities': [scores...],
      'evidence_matches': [scores...],
      'quality_flags': [[flags per image]...],
      'object_parts': [{'object': 'car', 'object_part': 'door', 'object_mismatch': False}, ...],
      'damage_types': ['dent', 'scratch', ...],
      'severities': ['medium', 'low', ...]
    }
    """
```

**Classifiers (heuristic-based):**
- `detect_quality_flags()` — blurry, low light, cropped, text present
- `classify_object_and_part()` — car/laptop/package → specific part (door, screen, box, etc.)
- `classify_damage_type()` — dent, scratch, crack, water_damage, etc.
- `classify_severity()` — low, medium, high, unknown

**Tech:** Sentence Transformers CLIP model (`clip-ViT-L-14`)

---

### 2. Organizer (organizer.py) — COMPLETE

**Class:** `Organizer`

**Purpose:** Assemble RAG output + user history + evidence requirements into a single context dict for LLM.

**Public method:**
```python
def process_claim(self, row):
    """
    Input: One row from claims.csv
    Output: Dict with all context needed for LLM
    {
      'user_id': '...',
      'image_ids': ['img_1', 'img_2'],
      'user_claim': '...',
      'claim_object': 'car',
      'object_parts': [...],
      'damage_types': [...],
      'severities': [...],
      'quality_flags_per_image': [...],
      'claim_similarities': [...],
      'evidence_matches': [...],
      'quality_flags_aggregated': '...',
      'evidence_standard_met': True/False,
      'valid_image': True/False,
      'user_history': {...},
      'evidence_requirements': [...],
      'error': None or error_msg
    }
    """
```

**No reasoning logic.** Pure data assembly.

---

## What's Needed

### 3. LLM Integration Layer (NEEDS TO BE BUILT)

**Purpose:** Take organized context and produce all 14 output fields.

**Responsibilities:**
- Load llm_context.md (decision framework)
- Call LLM API (Ollama local, OpenAI, Anthropic, etc.)
- Receive decision dict from LLM
- Validate output (all 14 fields, correct enums)
- Handle errors gracefully

**Suggested method:**
```python
class LLMReasoner:
    def __init__(self, model_name='local', api_key=None):
        # Initialize connection to LLM (Ollama, OpenAI, Anthropic, etc.)
        pass
    
    def reason(self, organized_context):
        """
        Input: Dict from Organizer.process_claim()
        Output: Dict with 14 fields:
        {
          'user_id': '...',
          'image_paths': '...',
          'user_claim': '...',
          'claim_object': '...',
          'evidence_standard_met': 'true'/'false',
          'evidence_standard_met_reason': '...',
          'risk_flags': '...;...;...|none',
          'issue_type': 'dent|scratch|...|none|unknown',
          'object_part': '...',
          'claim_status': 'supported|contradicted|not_enough_information',
          'claim_status_justification': '...',
          'supporting_image_ids': 'img_1;img_2|none',
          'valid_image': 'true'/'false',
          'severity': 'low|medium|high|unknown'
        }
        """
        pass
```

**Input to LLM:**
```
System prompt: [contents of llm_context.md]

User message: 
{
  Organize context dict from Organizer
  (all fields listed above)
}

Expected response:
JSON or structured dict with 14 fields.
```

---

### 4. Agent / Main Orchestrator (NEEDS TO BE BUILT)

**Purpose:** Tie everything together.

**Responsibilities:**
- Load claims.csv
- For each row:
  - Call Organizer.process_claim(row) → get organized_context
  - Call LLMReasoner.reason(organized_context) → get output_row
  - Accumulate output rows
- Write output.csv with exact column order
- Handle errors per row (don't crash on one bad claim)

**Suggested structure:**
```python
class Agent:
    def __init__(self):
        self.organizer = Organizer()
        self.llm = LLMReasoner(model='ollama', api_key=...)
    
    def run(self, input_csv='dataset/claims.csv', output_csv='output.csv'):
        """Load, process, write output."""
        rows = self._load_csv(input_csv)
        output_rows = []
        
        for row in rows:
            try:
                organized = self.organizer.process_claim(row)
                output = self.llm.reason(organized)
                output_rows.append(output)
            except Exception as e:
                output_rows.append(self._error_fallback(row, e))
        
        self._write_csv(output_csv, output_rows)
```

---

### 5. Evaluation Framework (NEEDS TO BE BUILT)

**Purpose:** Evaluate system on `dataset/sample_claims.csv` (labeled), measure accuracy, generate report.

**Responsibilities:**
- Run Agent on sample_claims.csv
- Compare predictions to expected_output (labels in CSV)
- Calculate metrics:
  - Evidence standard accuracy
  - Claim status F1 / precision / recall
  - Risk flag coverage
  - Per-field agreement rates
- Document model calls, token usage, cost estimate
- Try 2+ strategies (different LLM prompts, model choices) and compare
- Generate `evaluation/evaluation_report.md`

**Suggested metrics:**
```python
metrics = {
  'accuracy_evidence_standard_met': float,
  'accuracy_claim_status': float,
  'f1_claim_status': float,
  'precision_by_status': {'supported': X, 'contradicted': Y, ...},
  'recall_by_status': {...},
  'model_calls_total': int,
  'model_calls_per_claim': float,
  'estimated_cost': float,
  'token_usage': {'input': X, 'output': Y},
  'runtime_seconds': float
}
```

---

## Input/Output Schema

### Input: dataset/claims.csv

Columns:
- `user_id` — user submitting claim
- `image_paths` — semicolon-separated paths (e.g., `images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg`)
- `user_claim` — chat transcript describing damage
- `claim_object` — `car`, `laptop`, or `package`

### Output: output.csv

14 columns (exact order required):
1. `user_id`
2. `image_paths`
3. `user_claim`
4. `claim_object`
5. `evidence_standard_met` — `true` or `false`
6. `evidence_standard_met_reason` — short explanation
7. `risk_flags` — semicolon-separated or `none`
8. `issue_type` — dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown
9. `object_part` — specific part (door, screen, box, unknown, etc.)
10. `claim_status` — `supported`, `contradicted`, `not_enough_information`
11. `claim_status_justification` — explanation grounded in images
12. `supporting_image_ids` — semicolon-separated image IDs or `none`
13. `valid_image` — `true` or `false`
14. `severity` — `none`, `low`, `medium`, `high`, `unknown`

### Supporting Data Files

- `dataset/user_history.csv` — past claim counts, flags for each user
- `dataset/evidence_requirements.csv` — minimum evidence rules per object/issue type
- `dataset/sample_claims.csv` — labeled examples (for evaluation)
- `dataset/images/sample/` and `dataset/images/test/` — actual images

---

## Current Status

✅ RAG layer — complete, tested  
✅ Organizer layer — complete, tested  
❌ LLM integration — **needs implementation**  
❌ Agent orchestrator — **needs implementation**  
❌ Evaluation framework — **needs implementation**  

---

## How to Run (Once Complete)

```bash
# Development & testing
python -m code.evaluation.main  # Run on sample_claims.csv, generate metrics

# Production
python -m code.main             # Run on full claims.csv, generate output.csv
```

Both should output structured CSV and print summary stats.

---

## Technology Stack

- **Vision/RAG:** Sentence Transformers CLIP (`clip-ViT-L-14`)
- **LLM options:** Ollama (local), OpenAI (API), Anthropic Claude (API), Gemini (API)
- **Env config:** `.env` file with API keys (never hardcode)
- **Python 3.9+**
- **Dependencies:** `sentence-transformers`, `pillow`, `requests` (for API calls), etc.

---

## Key Decision Points for Implementation

### 1. LLM Choice
- **Ollama local:** No API cost, but slower inference (~10s per claim), limited model quality
- **OpenAI/Anthropic/Gemini API:** Faster, better quality, but API costs and rate limits
- **Recommendation:** Start with Ollama for free tier testing, switch to API if time permits

### 2. Prompt Strategy
- Use llm_context.md as system prompt
- Structure user message as JSON dump of organized_context
- Request JSON output (not free text)
- Example:
  ```
  System: [llm_context.md content]
  User: {json dump of context}
  Expected: JSON with 14 fields
  ```

### 3. Error Handling
- If RAG fails (missing image): Organizer sets error flag, LLM receives it, outputs graceful fallback
- If LLM fails (timeout, invalid response): Agent catches, logs, writes fallback row
- No crashes; every row gets an output (even if uncertain)

### 4. Batching & Caching
- CLIP embeddings: Batch encode all images + claims (done in RAG)
- Evidence requirements: Batch encode once at startup (done in Organizer)
- LLM calls: No native batching for most APIs, but consider:
  - Cache system prompt in context
  - Use streaming for long justifications
  - Implement retry logic for transient failures

---

## File Structure

```
code/
├── main.py                    # Entry point (Agent + pipeline)
├── rag.py                     # RAG class (COMPLETE)
├── organizer.py               # Organizer class (COMPLETE)
├── llm_reasoner.py            # LLMReasoner class (TODO)
├── agent.py                   # Agent orchestrator (TODO)
├── evaluation/
│   ├── main.py                # Evaluation runner (TODO)
│   ├── metrics.py             # Metric calculation (TODO)
│   └── evaluation_report.md   # Final report (TODO, generated)
├── config.py                  # Config, env loading (TODO)
├── utils.py                   # Helpers (TODO if needed)
└── README.md                  # Setup & usage (TODO)

dataset/
├── claims.csv
├── sample_claims.csv
├── user_history.csv
├── evidence_requirements.csv
└── images/
    ├── sample/
    └── test/

output.csv                      # Generated
llm_context.md                  # (provided)
```

---

## Next Steps for Whoever Implements This

1. **Implement LLMReasoner class:**
   - Choose LLM (Ollama/OpenAI/Anthropic)
   - Write prompt construction
   - Parse LLM response into 14 fields
   - Validate output

2. **Implement Agent:**
   - Load input CSV
   - Loop: Organizer → LLMReasoner → accumulate
   - Write output CSV with exact column order
   - Log errors and stats

3. **Implement Evaluation:**
   - Load sample_claims.csv with labels
   - Run Agent
   - Calculate precision/recall/F1
   - Compare 2+ LLM strategies
   - Write report

4. **Test:**
   - Run on small subset (5-10 claims) to verify output shape
   - Check output.csv is valid and readable
   - Spot-check justifications make sense

5. **Optimize (if time):**
   - Tune RAG thresholds on sample data
   - Refine LLM prompt based on errors
   - Add caching/batching if slow

---

## Important Notes

- **Do not hardcode API keys.** Use `.env` file and read with `python-dotenv`
- **Logging is mandatory.** Append to `$HOME/hackerrank_orchestrate/log.txt` per AGENTS.md
- **All 14 fields required.** No partial output; use `unknown`/`none` as fallbacks
- **Test on sample_claims.csv first.** Only run on claims.csv when confident
- **Keep RAG and Organizer unchanged.** They're working; focus on LLM + Agent layers

---

## Contact

This context is self-contained. You should have enough to implement the missing pieces independently.

Good luck!
