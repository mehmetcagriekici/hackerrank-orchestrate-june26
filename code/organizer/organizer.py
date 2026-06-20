import logging
import csv
from pathlib import Path

from rag.rag import Rag

logger = logging.getLogger(__name__)


class Organizer:
    """
    Data assembly layer.
    Takes RAG output + user history + evidence requirements.
    Assembles clean context dict for LLM (no reasoning, just plumbing).
    """
    
    # Evidence match threshold (tunable on sample_claims.csv)
    EVIDENCE_MATCH_THRESHOLD = 0.22
    
    def __init__(self):
        logger.info("Initializing Organizer")
        try:
            self.rag = Rag()
            logger.info("✓ RAG initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {str(e)}")
            raise
        
        try:
            self.evidence_requirements = self._load_evidence_requirements()
            logger.info(f"✓ Loaded {len(self.evidence_requirements)} evidence requirements")
        except Exception as e:
            logger.error(f"Failed to load evidence requirements: {str(e)}")
            raise
        
        try:
            self.user_history = self._load_all_user_history()
            logger.info(f"✓ Loaded history for {len(self.user_history)} users")
        except Exception as e:
            logger.error(f"Failed to load user history: {str(e)}")
            raise
    
    def _load_evidence_requirements(self):
        """Load evidence requirements from CSV."""
        req_path = Path("dataset/evidence_requirements.csv")
        if not req_path.exists():
            logger.warning(f"Evidence requirements file not found: {req_path}")
            return []
        
        try:
            with open(req_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                reqs = list(reader)
            logger.debug(f"Loaded {len(reqs)} evidence requirement records")
            return reqs
        except Exception as e:
            logger.error(f"Failed to parse evidence requirements CSV: {str(e)}")
            raise
    
    def _load_all_user_history(self):
        """Load user history indexed by user_id."""
        hist_path = Path("dataset/user_history.csv")
        history = {}
        
        if not hist_path.exists():
            logger.warning(f"User history file not found: {hist_path}")
            return history
        
        try:
            with open(hist_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    user_id = row.get('user_id')
                    if not user_id:
                        logger.warning("Row missing user_id, skipping")
                        continue
                    
                    history[user_id] = {
                        "past_claim_count": int(row.get('past_claim_count', 0)),
                        "last_90_days_claim_count": int(row.get('last_90_days_claim_count', 0)),
                        "accept_claim": int(row.get('accept_claim', 0)),
                        "manual_review_claim": int(row.get('manual_review_claim', 0)),
                        "rejected_claim": int(row.get('rejected_claim', 0)),
                        "history_flags": row.get('history_flags', 'none'),
                        "history_summary": row.get('history_summary', '')
                    }
            
            logger.debug(f"Loaded {len(history)} user history records")
            return history
            
        except Exception as e:
            logger.error(f"Failed to parse user history CSV: {str(e)}")
            raise
    
    def process_claim(self, row):
        """
        Process a single claim: run RAG, assemble data for LLM.
        Returns a dict with all context needed for LLM to decide.
        
        Args:
            row: Single claim row from input CSV
        
        Returns:
            Dict with all RAG analysis + context for LLM
        """
        # Extract and validate input fields
        try:
            user_id = row.get('user_id', '').strip()
            image_paths_str = row.get('image_paths', '').strip()
            user_claim = row.get('user_claim', '').strip()
            claim_object = row.get('claim_object', '').strip()
            
            # Validate required fields
            if not user_id:
                logger.error("Missing user_id in row")
                return self._assemble_error_context(
                    'unknown', '', '', 'unknown', [], {}, "Missing user_id"
                )
            
            if not claim_object:
                logger.error(f"Missing claim_object for user {user_id}")
                return self._assemble_error_context(
                    user_id, image_paths_str, user_claim, 'unknown', [], {}, "Missing claim_object"
                )
            
            logger.info(f"Processing claim for user {user_id} | object={claim_object}")
            
        except Exception as e:
            logger.error(f"Error parsing input row: {str(e)}")
            return self._assemble_error_context(
                'unknown', '', '', 'unknown', [], {}, f"Input parse error: {str(e)}"
            )
        
        # Parse image paths
        if not image_paths_str:
            logger.warning(f"No image paths for user {user_id}")
            image_paths = []
            image_ids = []
        else:
            try:
                image_paths = [p.strip() for p in image_paths_str.split(';') if p.strip()]
                image_ids = [Path(p).stem for p in image_paths]
                logger.debug(f"Parsed {len(image_paths)} image paths")
            except Exception as e:
                logger.error(f"Error parsing image paths for user {user_id}: {str(e)}")
                return self._assemble_error_context(
                    user_id, image_paths_str, user_claim, claim_object, [], {}, f"Image path parse error: {str(e)}"
                )
        
        # Get user history (fallback to defaults if not found)
        history = self.user_history.get(user_id, {
            "past_claim_count": 0,
            "last_90_days_claim_count": 0,
            "accept_claim": 0,
            "manual_review_claim": 0,
            "rejected_claim": 0,
            "history_flags": "none",
            "history_summary": ""
        })
        logger.debug(f"User history: {history.get('past_claim_count')} past claims")
        
        # Run RAG analysis
        if not image_paths:
            logger.warning(f"No images to analyze for user {user_id}")
            return self._assemble_error_context(
                user_id, image_paths_str, user_claim, claim_object, [], history, "No images submitted"
            )
        
        try:
            rag_result = self.rag.process(
                claim_text=user_claim,
                image_paths=image_paths,
                claim_object=claim_object,
                evidence_requirements=self.evidence_requirements
            )
            logger.info(f"RAG analysis complete for user {user_id}")
        except Exception as e:
            logger.error(f"RAG processing failed for user {user_id}: {str(e)}", exc_info=True)
            return self._assemble_error_context(
                user_id, image_paths_str, user_claim, claim_object, image_ids, history, f"RAG error: {str(e)}"
            )
        
        # Assemble all data for LLM
        try:
            return self._assemble_llm_context(
                user_id=user_id,
                image_paths_str=image_paths_str,
                image_ids=image_ids,
                user_claim=user_claim,
                claim_object=claim_object,
                rag_result=rag_result,
                history=history,
                error=None
            )
        except Exception as e:
            logger.error(f"Failed to assemble LLM context for user {user_id}: {str(e)}", exc_info=True)
            return self._assemble_error_context(
                user_id, image_paths_str, user_claim, claim_object, image_ids, history, f"Assembly error: {str(e)}"
            )
    
    def _assemble_llm_context(self, user_id, image_paths_str, image_ids, user_claim, claim_object, rag_result, history, error=None):
        """
        Assemble all context into a single dict for LLM.
        LLM will use this to produce all 14 output fields.
        
        Args:
            Various context pieces from processing
        
        Returns:
            Dict with all data for LLM reasoning
        """
        # Aggregate quality flags
        all_quality_flags = self._aggregate_quality_flags(rag_result.get('quality_flags', []))
        
        # Extract per-image analysis
        object_parts = rag_result.get('object_parts', [])
        damage_types = rag_result.get('damage_types', [])
        severities = rag_result.get('severities', [])
        claim_similarities = rag_result.get('claim_similarities', [])
        evidence_matches = rag_result.get('evidence_matches', [])
        
        # Compute evidence_standard_met
        evidence_standard_met = self._compute_evidence_met(damage_types, evidence_matches)
        
        # Compute valid_image (no invalid_image flags)
        valid_image = all("invalid_image" not in f for f in rag_result.get('quality_flags', []))
        
        # Filter evidence requirements to only relevant ones
        relevant_reqs = [req for req in self.evidence_requirements 
                        if req.get('claim_object') in [claim_object, 'all']]
        
        logger.debug(f"Assembled context: evidence_met={evidence_standard_met}, valid_image={valid_image}, relevant_reqs={len(relevant_reqs)}")
        
        return {
            # Input fields (for CSV output)
            'user_id': user_id,
            'image_paths': image_paths_str,
            'image_ids': image_ids,
            'user_claim': user_claim,
            'claim_object': claim_object,
            
            # RAG analysis per image
            'object_parts': object_parts,
            'damage_types': damage_types,
            'severities': severities,
            'quality_flags_per_image': rag_result.get('quality_flags', []),
            'claim_similarities': claim_similarities,
            'evidence_matches': evidence_matches,
            
            # Aggregated signals
            'quality_flags_aggregated': all_quality_flags,
            'evidence_standard_met': evidence_standard_met,
            'valid_image': valid_image,
            
            # Context for LLM (only relevant evidence requirements)
            'user_history': history,
            'evidence_requirements': relevant_reqs,
            
            # Error context
            'error': error
        }
    
    def _assemble_error_context(self, user_id, image_paths_str, user_claim, claim_object, image_ids, history, error_msg):
        """
        When processing fails, assemble minimal error context for LLM to handle gracefully.
        
        Args:
            Various input fields
            error_msg: Description of error
        
        Returns:
            Dict with error flag set for LLM
        """
        logger.warning(f"Assembling error context for user {user_id}: {error_msg}")
        
        return {
            'user_id': user_id,
            'image_paths': image_paths_str,
            'image_ids': image_ids,
            'user_claim': user_claim,
            'claim_object': claim_object,
            'object_parts': [],
            'damage_types': [],
            'severities': [],
            'quality_flags_per_image': [],
            'claim_similarities': [],
            'evidence_matches': [],
            'quality_flags_aggregated': 'none',
            'evidence_standard_met': False,
            'valid_image': False,
            'user_history': history,
            'evidence_requirements': [],
            'error': error_msg
        }
    
    def _aggregate_quality_flags(self, flags_per_image):
        """
        Combine quality flags from all images into semicolon-separated string.
        
        Args:
            flags_per_image: List of lists of flags, one list per image
        
        Returns:
            Semicolon-separated flag string or 'none'
        """
        all_flags = set()
        for flags in flags_per_image:
            for f in flags:
                if f not in ("none", "invalid_image"):
                    all_flags.add(f)
        
        result = ";".join(sorted(all_flags)) if all_flags else "none"
        logger.debug(f"Aggregated quality flags: {result}")
        return result
    
    def _compute_evidence_met(self, damage_types, evidence_matches):
        """
        Check if image evidence meets minimum standard for claim evaluation.
        
        Logic:
        - True if any visible damage detected
        - Or if evidence requirement match is strong
        
        Args:
            damage_types: List of detected damage types per image
            evidence_matches: List of evidence match scores per image
        
        Returns:
            Boolean indicating if evidence standard is met
        """
        # Check for valid damage
        has_valid_damage = any(d not in ("none", "unknown") for d in damage_types)
        if has_valid_damage:
            logger.debug("Evidence standard met: valid damage detected")
            return True
        
        # Check evidence requirement matches
        if evidence_matches:
            for idx, match in enumerate(evidence_matches):
                if match is not None:
                    try:
                        max_sim = float(match) if isinstance(match, (int, float)) else float(match.max()) if hasattr(match, 'max') else 0
                        if max_sim > self.EVIDENCE_MATCH_THRESHOLD:
                            logger.debug(f"Evidence standard met: image {idx} has strong evidence match ({max_sim:.3f})")
                            return True
                    except Exception as e:
                        logger.warning(f"Error computing evidence match for image {idx}: {str(e)}")
                        continue  # Check next match, don't fail entire check
        
        logger.debug("Evidence standard not met: no valid damage or strong evidence match")
        return False
