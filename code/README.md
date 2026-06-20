### Pipeline:

1. Embed user claim + all images using CLIP (same embedding space)
2. Compute similarity between claim embedding and each image embedding
3. RAG layer output: for each image-similarity score + image quality flags + object/part mismatch detection
4. Also compute evidence requirement match (embed requirements, score against images)
5. Send to LLM: claim + all image data + RAG scores + evidence sufficiency + user history + error messages
6. LLM produces all 14 output fields + final risk_flags

------------------------------------------------------------

### Architecture

Agent (orchestrator)
|___ RAG (vision/structure)
|      1. object_part
|      2. issue_type
|      3. severity
|      4. valid_image (based on quality flags)
|      5. per image: similarity, quality flags, metadata
|___ Organizer (logic/aggregation)
|      1. evidence_standard_met (true/false from evidence requirement match + image quality)
|      2. evidence_standard_met_reason (logic: "requires X views, got Y; quality threshold met/failed")
|      3. supporting_image_ids (aggregate from similarity scores: which images actually support?)
|      4. Error normalization (missing images → metadata)
|      5. Prepares clean data structure for LLM
|___ LLM (reasoning/formatting)
|      1. risk_flags (uses quality metadata + user history)
|      2. claim_status (supported/contradicted/not_enough_information)
|      3. claim_status_justification (narrative reasoning)
|      4. Formats all 14 fields

---------------------------------------------------------------

### Agent (write output)
    1. Load CSVs + images
    2. Call RAG.process()
    3. Call Organizer.prepare()
    4. Call LLM.reason()
    5. Write output.csv
    6. Handle & log errors

-----------------------

### RAG

### Organizer

### LLM
