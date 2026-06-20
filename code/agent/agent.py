import csv
import logging
import sys
from pathlib import Path
from datetime import datetime

from organizer.organizer import Organizer
from llm.llm import LLMReasoner

# Setup logging
log_dir = Path.home() / 'hackerrank_orchestrate'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'log.txt'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class Agent:
    """
    Main orchestrator that ties together:
    - Organizer: Assembles context from RAG + user history + evidence requirements
    - LLMReasoner: Makes decisions and produces 14-field output
    - CSV I/O: Reads input, writes output
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "mistral"):
        """
        Initialize Agent with Organizer and LLMReasoner.
        
        Args:
            ollama_url: Ollama API base URL
            model_name: LLM model name
        """
        logger.info(f"Initializing Agent | ollama_url={ollama_url} | model={model_name}")
        
        try:
            self.organizer = Organizer()
            logger.info("� Organizer initialized")
        except Exception as e:
            logger.error(f"? Failed to initialize Organizer: {str(e)}")
            raise e
        
        try:
            self.llm = LLMReasoner(ollama_url=ollama_url, model_name=model_name)
            logger.info("� LLMReasoner initialized")
        except Exception as e:
            logger.error(f"? Failed to initialize LLMReasoner: {str(e)}")
            raise e
        
        self.output_rows: list[dict[str, str]] = []
        self.stats = {
            'total_claims': 0,
            'processed_successfully': 0,
            'processed_with_error': 0,
            'claims_supported': 0,
            'claims_contradicted': 0,
            'claims_not_enough_info': 0
        }
    
    def run(self, input_csv: str = 'dataset/claims.csv', output_csv: str = 'output.csv') -> bool:
        """
        Load input CSV, process each claim, write output CSV.
        
        Args:
            input_csv: Path to input claims CSV
            output_csv: Path to output CSV
        
        Returns:
            True if successful, False if failed
        """
        start_time = datetime.now()
        logger.info("="*80)
        logger.info(f"Starting Agent.run() | input={input_csv} | output={output_csv}")
        logger.info("="*80)
    
        try:
            # Load input CSV
            rows = self._load_input_csv(input_csv)
            if not rows:
                logger.error(f"No claims found in {input_csv}")
                return False
            
            self.stats['total_claims'] = len(rows)
            logger.info(f"Loaded {len(rows)} claims from {input_csv}")
            
            # Process each claim
            for idx, row in enumerate(rows, 1):
                self._process_claim(row, idx, len(rows))
            
            # Write output CSV
            success = self._write_output_csv(output_csv)
            if not success:
                return False
            
            # Log summary statistics
            elapsed = (datetime.now() - start_time).total_seconds()
            self._log_summary(elapsed)
            
            return True
            
        except Exception as e:
            logger.error(f"? Agent.run() failed with exception: {str(e)}", exc_info=True)
            return False
    
    def _load_input_csv(self, input_path: str) -> list[dict[str, str]]:
        """
        Load claims from input CSV.
        
        Args:
            input_path: Path to input CSV
        
        Returns:
            List of claim dicts
        """
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            logger.info(f"Loaded input CSV | {len(rows)} rows")
            return rows
            
        except Exception as e:
            logger.error(f"? Failed to load input CSV: {str(e)}")
            raise e
    
    def _process_claim(self, row: dict[str, str], claim_num: int, total_claims: int):
        """
        Process a single claim through Organizer LLMReasoner pipeline.
        
        Args:
            row: Single claim row from input CSV
            claim_num: Claim number (for logging)
            total_claims: Total number of claims
        """
        user_id = row.get('user_id', 'unknown')
        
        try:
            # Step 1: Organizer assembles context
            logger.info(f"[{claim_num}/{total_claims}] Processing user_id={user_id}")
            organized_context = self.organizer.process_claim(row)
            
            # Check if Organizer had an error
            organizer_had_error = bool(organized_context.get('error'))
            if organizer_had_error:
                logger.warning(f"Organizer error: {organized_context['error']}")

            # Step 2: LLMReasoner produces decision
            output_row = self.llm.reason(organized_context)

            # Add to output rows
            self.output_rows.append(output_row)

            # Update stats (an organizer error context is not a successful claim)
            if organizer_had_error:
                self.stats['processed_with_error'] += 1
            else:
                self.stats['processed_successfully'] += 1
            claim_status = output_row.get('claim_status', 'unknown')
            if claim_status == 'supported':
                self.stats['claims_supported'] += 1
            elif claim_status == 'contradicted':
                self.stats['claims_contradicted'] += 1
            else:
                self.stats['claims_not_enough_info'] += 1
            
            logger.info(f"Complete | status={claim_status}")
            
        except Exception as e:
            logger.error(f"  ? Failed: {str(e)}", exc_info=False)
            self.stats['processed_with_error'] += 1
            
            # Add fallback output
            fallback = self._error_fallback(row, str(e))
            self.output_rows.append(fallback)
            logger.info("Using fallback output")
    
    def _write_output_csv(self, output_path: str) -> bool:
        """
        Write output rows to CSV in exact column order.
      
        Args:
            output_path: Path to output CSV
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.output_rows:
                logger.error("No output rows to write")
                return False
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write CSV with exact column order
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.llm.OUTPUT_COLUMNS)
                writer.writeheader()

                for row in self.output_rows:
                    clean_row = {col: row.get(col, '') for col in self.llm.OUTPUT_COLUMNS}
                    writer.writerow(clean_row)
            
            logger.info(f"Wrote output CSV | {len(self.output_rows)} rows | path={output_path}")
            return True
            
        except Exception as e:
            logger.error(f"? Failed to write output CSV: {str(e)}", exc_info=True)
            return False
    
    def _error_fallback(self, row: dict[str, str], error_msg: str) -> dict[str, str]:
        """
        Generate fallback output when claim processing fails.
        
        Args:
            row: Original input row
            error_msg: Error message
        
        Returns:
            Fallback output dict
        """
        return {
            'user_id': str(row.get('user_id', 'unknown')),
            'image_paths': str(row.get('image_paths', 'none')),
            'user_claim': str(row.get('user_claim', '')),
            'claim_object': str(row.get('claim_object', 'unknown')),
            'evidence_standard_met': 'false',
            'evidence_standard_met_reason': f'Processing error: {error_msg[:80]}',
            'risk_flags': 'manual_review_required',
            'issue_type': 'unknown',
            'object_part': 'unknown',
            'claim_status': 'not_enough_information',
            'claim_status_justification': 'Unable to process claim due to technical error. Manual review required.',
            'supporting_image_ids': 'none',
            'valid_image': 'false',
            'severity': 'unknown'
        }
    
    def _log_summary(self, elapsed_seconds: float):
        """
        Log summary statistics.
        
        Args:
            elapsed_seconds: Total elapsed time
        """
        logger.info("="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        logger.info(f"Total claims processed: {self.stats['total_claims']}")
        logger.info(f"Successful: {self.stats['processed_successfully']}")
        logger.info(f"  ? With errors: {self.stats['processed_with_error']}")
        logger.info("")
        logger.info("Claim status breakdown:")
        logger.info(f"  Supported: {self.stats['claims_supported']}")
        logger.info(f"  Contradicted: {self.stats['claims_contradicted']}")
        logger.info(f"  Not enough information: {self.stats['claims_not_enough_info']}")
        logger.info("")
        logger.info(f"Elapsed time: {elapsed_seconds:.1f}s ({elapsed_seconds/max(1, self.stats['total_claims']):.1f}s per claim)")
        logger.info("="*80)
