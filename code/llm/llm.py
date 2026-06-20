import json
import re
import time
import requests
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LLMReasoner:
    """
    Reasoning layer for damage claim verification.
    Takes organized context from Organizer and calls Ollama to produce
    all 14 output fields.
    """
    
    # Exact output column order (single source of truth)
    OUTPUT_COLUMNS = [
        "user_id",
        "image_paths",
        "user_claim",
        "claim_object",
        "evidence_standard_met",
        "evidence_standard_met_reason",
        "risk_flags",
        "issue_type",
        "object_part",
        "claim_status",
        "claim_status_justification",
        "supporting_image_ids",
        "valid_image",
        "severity",
    ]
    
    # Valid enum values
    VALID_STATUSES = {"supported", "contradicted", "not_enough_information"}
    
    VALID_SEVERITIES = {"none", "low", "medium", "high", "unknown"}
    
    VALID_ISSUES = {
        "dent", "scratch", "crack", "glass_shatter", "broken_part",
        "missing_part", "torn_packaging", "crushed_packaging",
        "water_damage", "stain", "none", "unknown",
    }
    
    VALID_RISK_FLAGS = {
        "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
        "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
        "possible_manipulation", "non_original_image", "text_instruction_present",
        "user_history_risk", "manual_review_required",
    }
    
    # Configuration
    MAX_RETRIES = 3
    RETRY_BACKOFF_SECONDS = 2  # Exponential: 2s, 4s, 8s
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "mistral"):
        """
        Initialize LLM reasoner.
        
        Args:
            ollama_url: Base URL for Ollama API
            model_name: LLM model name (e.g., 'mistral', 'llama3')
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        
        logger.info(f"Initializing LLMReasoner | url={self.ollama_url} | model={model_name}")
        
        # Test Ollama connection
        self._test_ollama_connection()
        
        # Build system prompt once
        self.system_prompt = self._build_system_prompt()
        logger.info("✓ LLMReasoner initialized")
    
    def _test_ollama_connection(self):
        """Test if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"✓ Ollama is running at {self.ollama_url}")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama at {self.ollama_url}. Ensure Docker container is running.")
        except Exception as e:
            raise Exception(f"Ollama connection error: {str(e)}")
    
    def reason(self, organized_context: dict[str, Any]) -> dict[str, str]:
        """
        Take organized context and produce 14-field output dict.
        
        Args:
            organized_context: Dict from Organizer.process_claim()
        
        Returns:
            Dict with exactly 14 fields ready for CSV output
        """
        try:
            # Handle error context gracefully
            if organized_context.get('error'):
                logger.warning(f"Context has error: {organized_context['error']}")
                return self._error_output(organized_context, organized_context['error'])
            
            user_id = organized_context.get('user_id', 'unknown')
            logger.info(f"Processing LLM reasoning for user {user_id}")
            
            # Build user message from context
            user_message = self._build_user_message(organized_context)
            
            # Call Ollama with retry logic
            llm_response = self._call_ollama_with_retry(user_message)
            logger.debug(f"Ollama response length: {len(llm_response)} chars")
            
            # Parse and validate output
            output_dict = self._parse_response(llm_response)
            
            # Normalize and validate all 14 fields
            output_dict = self._validate_and_normalize(output_dict, organized_context)
            
            logger.info(f"✓ LLM reasoning complete | status={output_dict.get('claim_status')}")
            return output_dict
            
        except Exception as e:
            logger.error(f"LLM reasoning failed: {str(e)}", exc_info=True)
            return self._error_output(organized_context, str(e))
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt with decision logic and examples."""
        return """You are an expert damage claim verification system. Your task is to analyze damage claims with image evidence and make structured decisions.

You will receive:
1. A user's damage claim description
2. RAG analysis of images (object detection, damage type detection, severity classification, quality flags)
3. User history and evidence requirements
4. Aggregated evidence assessment

Your job is to produce a structured JSON response with exactly these 14 fields:

1. evidence_standard_met: 'true' or 'false' - Does image evidence meet minimum standard?
2. evidence_standard_met_reason: Brief explanation of evidence decision
3. risk_flags: Semicolon-separated risk flags or 'none'. 
   ALLOWED VALUES: blurry_image, cropped_or_obstructed, low_light_or_glare, wrong_angle, 
   wrong_object, wrong_object_part, damage_not_visible, claim_mismatch, possible_manipulation, 
   non_original_image, text_instruction_present, user_history_risk, manual_review_required
4. issue_type: ENUM: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown
5. object_part: Specific part affected (e.g., 'door', 'screen', 'box') or 'unknown'
6. claim_status: ENUM: 'supported', 'contradicted', 'not_enough_information'
7. claim_status_justification: Detailed explanation grounded in image evidence
8. supporting_image_ids: Semicolon-separated image IDs supporting the claim or 'none'
9. valid_image: 'true' or 'false' - Are images of sufficient quality for decision-making?
10. severity: ENUM: none, low, medium, high, unknown

DECISION LOGIC:

claim_status Decision Tree:
- IF images show damage AND damage matches claim description AND object matches claimed object AND quality is good
  → 'supported'
- IF images show NO damage OR different damage type AND claim describes definite damage
  → 'contradicted'
- IF images unclear/blurry/cropped OR insufficient evidence OR ambiguous
  → 'not_enough_information'

evidence_standard_met Logic:
- TRUE if: images have valid damage OR strong evidence requirement match
- FALSE if: no images, invalid images, or insufficient visual evidence

risk_flags Guidelines:
- Add 'blurry_image' if quality_flags includes blurry
- Add 'wrong_object' if object_mismatch is true
- Add 'damage_not_visible' if no damage detected
- Add 'user_history_risk' if user has rejection history or many manual review claims
- Add 'manual_review_required' if claim_status is not 'supported'
- Separate multiple flags with semicolons

EXAMPLES:

Example 1 - Supported:
  Claim: "car door has dent"
  RAG: object=['car'], part=['door'], damage=['dent'], quality_flags=['none'], 
       claim_similarity=0.89, valid_image=true
  Decision:
    claim_status: 'supported'
    issue_type: 'dent'
    object_part: 'door'
    risk_flags: 'none'
    severity: 'medium'
    justification: 'Images clearly show a dent on the car door matching the claim.'

Example 2 - Contradicted:
  Claim: "laptop screen is broken"
  RAG: object=['laptop'], part=['screen'], damage=['none'], quality_flags=['none'],
       claim_similarity=0.75, valid_image=true
  Decision:
    claim_status: 'contradicted'
    issue_type: 'none'
    object_part: 'screen'
    risk_flags: 'damage_not_visible'
    severity: 'none'
    justification: 'Images show laptop screen is intact and undamaged, contradicting the claim.'

Example 3 - Not Enough Information:
  Claim: "package was crushed"
  RAG: object=['package'], damage=['unknown'], quality_flags=['blurry_image', 'low_light'],
       claim_similarity=0.45, valid_image=false
  Decision:
    claim_status: 'not_enough_information'
    issue_type: 'unknown'
    object_part: 'unknown'
    risk_flags: 'blurry_image;low_light_or_glare;manual_review_required'
    severity: 'unknown'
    justification: 'Image quality is too poor to reliably assess the claimed damage.'

IMPORTANT NOTES:
- Ground ALL decisions in the RAG analysis and image evidence
- Be conservative: when uncertain, use 'not_enough_information'
- Consider user history and risk factors in risk_flags
- Always respond with valid JSON
- Use 'none' or 'unknown' as fallbacks
- Return ONLY valid JSON, no explanatory text"""
    
    def _build_user_message(self, context: dict[str, Any]) -> str:
        """Build user message containing organized context."""
        
        # Format quality flags per image
        quality_flags_formatted = {}
        for img_id, flags in zip(context.get('image_ids', []), context.get('quality_flags_per_image', [])):
            quality_flags_formatted[img_id] = flags
        
        # Format object parts per image
        object_parts_formatted = {}
        for img_id, part_info in zip(context.get('image_ids', []), context.get('object_parts', [])):
            object_parts_formatted[img_id] = part_info
        
        # Format evidence matches (convert to float if needed)
        evidence_matches_formatted = {}
        for img_id, match in zip(context.get('image_ids', []), context.get('evidence_matches', [])):
            if match is not None:
                try:
                    match_val = float(match) if isinstance(match, (int, float)) else float(match.max())
                    evidence_matches_formatted[img_id] = round(match_val, 3)
                except Exception:
                    evidence_matches_formatted[img_id] = None
            else:
                evidence_matches_formatted[img_id] = None
        
        # Format claim similarities (should already be float)
        claim_sims_formatted = {}
        for img_id, sim in zip(context.get('image_ids', []), context.get('claim_similarities', [])):
            try:
                sim_val = float(sim)
                claim_sims_formatted[img_id] = round(sim_val, 3)
            except Exception:
                claim_sims_formatted[img_id] = None
        
        # Build message dict
        message = {
            "claim_summary": {
                "user_id": context.get('user_id'),
                "claim_object": context.get('claim_object'),
                "user_claim": context.get('user_claim'),
                "image_ids": context.get('image_ids', []),
            },
            "rag_analysis": {
                "damage_types": context.get('damage_types', []),
                "severities": context.get('severities', []),
                "object_parts": object_parts_formatted,
                "quality_flags_per_image": quality_flags_formatted,
                "quality_flags_aggregated": context.get('quality_flags_aggregated', 'none'),
                "claim_similarities": claim_sims_formatted,
                "evidence_requirement_matches": evidence_matches_formatted
            },
            "computed_signals": {
                "evidence_standard_met": context.get('evidence_standard_met'),
                "valid_image": context.get('valid_image')
            },
            "user_history": context.get('user_history', {}),
            "evidence_requirements": context.get('evidence_requirements', [])
        }
        
        return json.dumps(message, indent=2)
    
    def _call_ollama_with_retry(self, user_message: str) -> str:
        """
        Call Ollama API with exponential backoff retry logic.

        Args:
            user_message: JSON-formatted user message

        Returns:
            LLM response text
        """
        retryable = (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
        for attempt in range(self.MAX_RETRIES):
            try:
                return self._call_ollama(user_message)
            except retryable as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise Exception(f"Ollama unreachable after {self.MAX_RETRIES} attempts: {e}")
                wait_time = self.RETRY_BACKOFF_SECONDS * (2 ** attempt)
                logger.warning(f"Ollama error ({type(e).__name__}), retrying in {wait_time}s (attempt {attempt+1}/{self.MAX_RETRIES})")
                time.sleep(wait_time)
        raise Exception(f"Ollama call failed after {self.MAX_RETRIES} attempts")

    def _call_ollama(self, user_message: str) -> str:
        """
        Call Ollama API.
        
        Args:
            user_message: JSON-formatted user message
        
        Returns:
            LLM response text
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "stream": False,
            "format": "json"
        }
        
        logger.debug(f"Calling Ollama /api/chat with model {self.model_name}")
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get('message', {}).get('content', '')
            
            logger.debug(f"Ollama response: {len(content)} chars")
            return content
            
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise
    
    def _parse_response(self, response_text: str) -> dict[str, str]:
        """
        Parse LLM JSON response, handling markdown code blocks and edge cases.
        
        Args:
            response_text: Raw response from LLM
        
        Returns:
            Parsed dict
        """
        try:
            # Try direct JSON parse first
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            logger.debug("Direct JSON parse failed, trying markdown extraction")
        
        # Try markdown code block extraction
        try:
            match = re.search(r'```(?:json)?\s*({.*?})\s*```', response_text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                logger.debug("Extracted JSON from markdown code block")
                return json.loads(json_str)
        except Exception as e:
            logger.debug(f"Markdown extraction failed: {str(e)}")
        
        # Try to find JSON object anywhere in response
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_str = match.group(0).strip()
                logger.debug("Extracted JSON from response text")
                return json.loads(json_str)
        except Exception as e:
            logger.debug(f"Text extraction failed: {str(e)}")
        
        # All parsing attempts failed
        logger.error(f"Cannot parse LLM response as JSON: {response_text[:500]}")
        raise Exception(f"Invalid JSON from LLM: {response_text[:500]}")
    
    @staticmethod
    def _normalize_bool(val) -> str:
        if isinstance(val, bool):
            return 'true' if val else 'false'
        s = str(val).lower().strip()
        return 'true' if s in ('true', 'yes', '1') else 'false'

    def _validate_and_normalize(self, output_dict: dict[str, Any], context: dict[str, Any]) -> dict[str, str]:
        """
        Validate all 14 fields are present and normalized to correct types/enums.
        
        Args:
            output_dict: Dict from LLM (may be incomplete or have invalid values)
            context: Original context for fallback values
        
        Returns:
            Normalized dict with exactly 14 fields in correct order
        """
        logger.debug(f"Validating LLM output with {len(output_dict)} fields")
        
        # Start with input fields from context
        normalized = {
            'user_id': str(context.get('user_id', 'unknown')),
            'image_paths': str(context.get('image_paths', 'none')),
            'user_claim': str(context.get('user_claim', '')),
            'claim_object': str(context.get('claim_object', 'unknown')),
        }
        
        # Validate and normalize LLM output fields
        
        # evidence_standard_met: bool → 'true'/'false'
        normalized['evidence_standard_met'] = self._normalize_bool(output_dict.get('evidence_standard_met', 'false'))
        
        # evidence_standard_met_reason
        normalized['evidence_standard_met_reason'] = str(output_dict.get('evidence_standard_met_reason', 'No reason provided'))[:200]
        
        # risk_flags: validate against enum
        risk_flags = output_dict.get('risk_flags', 'none')
        if isinstance(risk_flags, list):
            risk_flags = ';'.join(str(f).strip() for f in risk_flags if f)
        risk_flags = str(risk_flags).strip()
        
        if not risk_flags or risk_flags.lower() == 'none':
            normalized['risk_flags'] = 'none'
        else:
            provided_flags = [f.strip() for f in risk_flags.split(';')]
            valid_flags = [f for f in provided_flags if f in self.VALID_RISK_FLAGS]
            if not valid_flags:
                logger.warning(f"Invalid risk flags from LLM: {provided_flags}, using 'none'")
                normalized['risk_flags'] = 'none'
            else:
                normalized['risk_flags'] = ';'.join(valid_flags)
        
        # issue_type: validate enum
        issue_type = self._normalize_issue_type(output_dict.get('issue_type', 'unknown'))
        normalized['issue_type'] = issue_type
        
        # object_part
        normalized['object_part'] = str(output_dict.get('object_part', 'unknown')).strip()[:50]
        
        # claim_status: validate enum
        claim_status = str(output_dict.get('claim_status', 'not_enough_information')).lower().strip()
        if claim_status not in self.VALID_STATUSES:
            logger.warning(f"Invalid claim_status from LLM: {claim_status}, using 'not_enough_information'")
            claim_status = 'not_enough_information'
        normalized['claim_status'] = claim_status
        
        # claim_status_justification
        normalized['claim_status_justification'] = str(output_dict.get('claim_status_justification', 'Unable to determine'))[:500]
        
        # supporting_image_ids: semicolon-separated or 'none'
        supporting = output_dict.get('supporting_image_ids', 'none')
        if isinstance(supporting, list):
            supporting = ';'.join(str(s).strip() for s in supporting if s)
        supporting = str(supporting).strip()
        if not supporting or supporting.lower() == 'none':
            normalized['supporting_image_ids'] = 'none'
        else:
            normalized['supporting_image_ids'] = supporting
        
        # valid_image: bool → 'true'/'false'
        normalized['valid_image'] = self._normalize_bool(output_dict.get('valid_image', 'false'))
        
        # severity: validate enum
        severity = str(output_dict.get('severity', 'unknown')).lower().strip()
        if severity not in self.VALID_SEVERITIES:
            logger.warning(f"Invalid severity from LLM: {severity}, using 'unknown'")
            severity = 'unknown'
        normalized['severity'] = severity
        
        result = {field: normalized.get(field, 'unknown') for field in self.OUTPUT_COLUMNS}
        
        logger.debug(f"Validation complete: {list(result.keys())}")
        return result
    
    def _normalize_issue_type(self, issue_type: str) -> str:
        normalized = str(issue_type).lower().strip()
        if normalized in self.VALID_ISSUES:
            return normalized
        underscored = normalized.replace(' ', '_')
        if underscored in self.VALID_ISSUES:
            return underscored
        logger.warning(f"Unknown issue_type from LLM: {issue_type}, using 'unknown'")
        return 'unknown'
    
    def _error_output(self, context: dict[str, Any], error_msg: str) -> dict[str, str]:
        """
        When LLM fails, return graceful fallback output.
        
        Args:
            context: Original organized context
            error_msg: Description of error
        
        Returns:
            Dict with all 14 fields, error indicators set
        """
        logger.warning(f"Generating error fallback: {error_msg}")
        
        return {
            'user_id': str(context.get('user_id', 'unknown')),
            'image_paths': str(context.get('image_paths', 'none')),
            'user_claim': str(context.get('user_claim', '')),
            'claim_object': str(context.get('claim_object', 'unknown')),
            'evidence_standard_met': 'false',
            'evidence_standard_met_reason': f'Processing error: {error_msg[:80]}',
            'risk_flags': 'manual_review_required',
            'issue_type': 'unknown',
            'object_part': 'unknown',
            'claim_status': 'not_enough_information',
            'claim_status_justification': 'Unable to process claim due to technical error. Manual review required.',
            'supporting_image_ids': 'none',
            'valid_image': 'false',
            'severity': 'unknown',
        }
