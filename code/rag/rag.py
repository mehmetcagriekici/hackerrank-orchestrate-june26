import logging
from sentence_transformers import SentenceTransformer
from PIL import Image

logger = logging.getLogger(__name__)


class Rag:
    """
    Vision/Structure layer for damage claim analysis.
    Uses CLIP multimodal embeddings + heuristic classifiers.
    """
    
    # Quality flag detection thresholds (tunable on sample_claims.csv)
    QUALITY_THRESHOLDS = {
        "blurry image": ("blurry_image", 0.25),
        "low light or dark": ("low_light_or_glare", 0.28),
        "cropped or obstructed view": ("cropped_or_obstructed", 0.27),
        "text or instructions visible": ("text_instruction_present", 0.22),
    }
    
    def __init__(self, model_name='clip-ViT-L-14') -> None:
        logger.info(f"Initializing RAG with model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("✓ RAG initialized successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise

    def process(self, claim_text: str, image_paths: list, claim_object: str, evidence_requirements: list):
        """
        Main RAG process: embed and classify claim + images.
        
        Args:
            claim_text: User's claim description
            image_paths: List of image file paths
            claim_object: Object type (car, laptop, package)
            evidence_requirements: Evidence requirement records
        
        Returns:
            Dict with RAG analysis results (embeddings as floats, not numpy arrays)
        """
        logger.info(f"Processing claim for {claim_object} with {len(image_paths)} images")
        
        try:
            # Embed claim
            claim_emb = self.model.encode(claim_text)
            logger.debug(f"Encoded claim: shape {claim_emb.shape}")
            
            # Embed evidence requirements relevant to this claim_object
            relevant_reqs = [req for req in evidence_requirements if req['claim_object'] in [claim_object, 'all']]
            if relevant_reqs:
                req_embs = self.model.encode([req['applies_to'] for req in relevant_reqs])
                logger.debug(f"Encoded {len(relevant_reqs)} evidence requirements")
            else:
                req_embs = []
                logger.debug("No evidence requirements for this object type")
            
            # Load images individually so one bad image doesn't fail the whole claim
            n_images = len(image_paths)
            valid_images = []
            valid_indices = []
            for i, path in enumerate(image_paths):
                try:
                    img = Image.open(path)
                    img.load()  # force decode now to catch missing/corrupt files
                    valid_images.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"Image {i} ({path}) could not be loaded: {str(e)}")

            # Full-length, index-aligned results with invalid-image placeholders
            claim_similarities = [None] * n_images
            evidence_matches = [None] * n_images
            quality_flags = [["invalid_image"] for _ in range(n_images)]
            object_parts = [
                {"object": "unknown", "object_part": "unknown", "object_mismatch": False}
                for _ in range(n_images)
            ]
            damage_types = ["unknown"] * n_images
            severities = ["unknown"] * n_images

            if valid_images:
                image_embs = self.model.encode(valid_images)
                logger.debug(f"Encoded {len(valid_images)}/{n_images} images")

                # Ensure image_embs is always 2D
                if len(image_embs.shape) == 1:
                    image_embs = image_embs.reshape(1, -1)

                # Score valid images against claim (convert to float)
                valid_claim_sims = [float(self.model.similarity(claim_emb, img_emb)) for img_emb in image_embs]

                # Score valid images against evidence requirements (convert to float)
                if len(req_embs) > 0:
                    valid_evidence = []
                    for img_emb in image_embs:
                        match = self.model.similarity(req_embs, img_emb)
                        max_match = float(match.max()) if hasattr(match, 'max') else float(match)
                        valid_evidence.append(max_match)
                else:
                    valid_evidence = [None] * len(image_embs)

                # Classifiers on valid embedded images
                valid_quality = self.detect_quality_flags(image_embs)
                valid_parts = self.classify_object_and_part(image_embs, claim_object)
                valid_damage = self.classify_damage_type(image_embs)
                valid_severities = self.classify_severity(image_embs)

                # Weave valid results back into the full-length, index-aligned lists
                for j, orig_idx in enumerate(valid_indices):
                    claim_similarities[orig_idx] = valid_claim_sims[j]
                    evidence_matches[orig_idx] = valid_evidence[j]
                    quality_flags[orig_idx] = valid_quality[j]
                    object_parts[orig_idx] = valid_parts[j]
                    damage_types[orig_idx] = valid_damage[j]
                    severities[orig_idx] = valid_severities[j]
            else:
                logger.warning("No loadable images for this claim; all marked invalid_image")

            logger.info(f"RAG analysis complete | damages: {damage_types} | severities: {severities}")
            
            return {
                "claim_similarities": claim_similarities,
                "evidence_matches": evidence_matches,
                "quality_flags": quality_flags,
                "object_parts": object_parts,
                "damage_types": damage_types,
                "severities": severities,
            }
            
        except Exception as e:
            logger.error(f"RAG processing failed: {str(e)}", exc_info=True)
            raise

    def detect_quality_flags(self, image_embs):
        """
        Detect image quality issues using CLIP similarity to quality descriptors.
        Uses enum-safe dictionary mapping to avoid index brittle-ness.
        
        Returns:
            List of lists of quality flags per image
        """
        flags_list = []
        
        # Pre-encode quality candidates once
        candidates = list(self.QUALITY_THRESHOLDS.keys())
        candidates_emb = self.model.encode(candidates)
        
        for idx, img_emb in enumerate(image_embs):
            try:
                sims = self.model.similarity(candidates_emb, img_emb).flatten()
                
                flags = []
                for sim_idx, (candidate, (flag_name, threshold)) in enumerate(self.QUALITY_THRESHOLDS.items()):
                    if sims[sim_idx] > threshold:
                        flags.append(flag_name)
                
                if not flags:
                    flags.append("none")
                
                flags_list.append(flags)
                logger.debug(f"Image {idx}: quality flags {flags}")
                
            except FileNotFoundError as e:
                logger.warning(f"Image {idx} not found: {str(e)}")
                flags_list.append(["invalid_image"])
            except Exception as e:
                logger.error(f"Error classifying quality for image {idx}: {str(e)}", exc_info=False)
                flags_list.append(["invalid_image"])
        
        return flags_list

    def classify_object_and_part(self, image_embs, claim_object: str):
        """
        Classify object and part for each image.
        Pre-encodes all part candidates to avoid redundant encoding.
        
        Returns:
            List of dicts with object, object_part, object_mismatch per image
        """
        results = []
        
        # Define candidates per object type
        car_parts = ["front bumper", "rear bumper", "door", "hood", "windshield", "side mirror", "headlight", "taillight", "fender", "body"]
        laptop_parts = ["screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base", "body"]
        package_parts = ["box", "package_corner", "package_side", "seal", "label", "contents"]
        
        # Pre-encode all candidates once
        obj_candidates = ["car", "laptop", "package", "other object"]
        obj_candidates_emb = self.model.encode(obj_candidates)
        
        car_parts_emb = self.model.encode(car_parts)
        laptop_parts_emb = self.model.encode(laptop_parts)
        package_parts_emb = self.model.encode(package_parts)
        
        logger.debug("Pre-encoded object and part candidates")
        
        for idx, img_emb in enumerate(image_embs):
            try:
                # Classify object
                obj_sims = self.model.similarity(obj_candidates_emb, img_emb).flatten()
                detected_obj = obj_candidates[obj_sims.argmax()]
                
                # Classify part based on object
                if detected_obj == "car":
                    part_candidates = car_parts
                    candidates_emb = car_parts_emb
                elif detected_obj == "laptop":
                    part_candidates = laptop_parts
                    candidates_emb = laptop_parts_emb
                elif detected_obj == "package":
                    part_candidates = package_parts
                    candidates_emb = package_parts_emb
                else:
                    part_candidates = ["unknown"]
                    candidates_emb = self.model.encode(part_candidates)
                
                part_sims = self.model.similarity(candidates_emb, img_emb).flatten()
                detected_part = part_candidates[part_sims.argmax()]
                
                # Check for object/part mismatch
                detected_obj_clean = detected_obj if detected_obj != "other object" else "unknown"
                mismatch = detected_obj_clean != claim_object and detected_obj_clean != "unknown"
                
                result = {
                    "object": detected_obj_clean,
                    "object_part": detected_part,
                    "object_mismatch": mismatch
                }
                
                results.append(result)
                logger.debug(f"Image {idx}: object={detected_obj_clean}, part={detected_part}, mismatch={mismatch}")
                
            except FileNotFoundError as e:
                logger.warning(f"Image {idx} not found: {str(e)}")
                results.append({"object": "unknown", "object_part": "unknown", "object_mismatch": False})
            except Exception as e:
                logger.error(f"Error classifying object/part for image {idx}: {str(e)}", exc_info=False)
                results.append({"object": "unknown", "object_part": "unknown", "object_mismatch": False})
        
        return results

    def classify_damage_type(self, image_embs):
        """
        Classify damage type for each image.
        
        Returns:
            List of issue_type strings per image
        """
        issue_candidates = [
            "dent", "scratch", "crack", "glass shatter", "broken part", 
            "missing part", "torn packaging", "crushed packaging", 
            "water damage", "stain", "no visible damage", "unknown damage"
        ]
        
        # Pre-encode candidates once
        candidates_emb = self.model.encode(issue_candidates)
        
        results = []
        for idx, img_emb in enumerate(image_embs):
            try:
                sims = self.model.similarity(candidates_emb, img_emb).flatten()
                detected = issue_candidates[sims.argmax()]
                
                # Normalize output
                if detected == "no visible damage":
                    detected = "none"
                elif detected == "unknown damage":
                    detected = "unknown"
                
                results.append(detected)
                logger.debug(f"Image {idx}: damage_type={detected}")
                
            except FileNotFoundError as e:
                logger.warning(f"Image {idx} not found: {str(e)}")
                results.append("unknown")
            except Exception as e:
                logger.error(f"Error classifying damage for image {idx}: {str(e)}", exc_info=False)
                results.append("unknown")
        
        return results

    def classify_severity(self, image_embs):
        """
        Classify damage severity for each image.
        
        Returns:
            List of severity levels per image
        """
        severity_candidates = ["minor damage", "moderate damage", "severe damage", "no damage", "unknown severity"]
        
        # Pre-encode candidates once
        candidates_emb = self.model.encode(severity_candidates)
        
        results = []
        for idx, img_emb in enumerate(image_embs):
            try:
                sims = self.model.similarity(candidates_emb, img_emb).flatten()
                detected = severity_candidates[sims.argmax()]
                
                # Normalize to output enum
                if "no damage" in detected:
                    sev = "none"
                elif "minor" in detected:
                    sev = "low"
                elif "moderate" in detected:
                    sev = "medium"
                elif "severe" in detected:
                    sev = "high"
                else:
                    sev = "unknown"
                
                results.append(sev)
                logger.debug(f"Image {idx}: severity={sev}")
                
            except FileNotFoundError as e:
                logger.warning(f"Image {idx} not found: {str(e)}")
                results.append("unknown")
            except Exception as e:
                logger.error(f"Error classifying severity for image {idx}: {str(e)}", exc_info=False)
                results.append("unknown")
        
        return results
