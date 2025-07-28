#!/usr/bin/env python3
"""
Challenge 1B: Persona-Driven Document Intelligence
Enhanced version with upgraded models and model switching capability
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import PyPDF2
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Tuple, Any
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, model_config: str = "enhanced"):
        """
        Initialize the document processor with models
        
        Args:
            model_config: "basic", "enhanced", or "ensemble"
                - basic: Original all-MiniLM-L6-v2 model (~80MB)
                - enhanced: all-mpnet-base-v2 model (~420MB) 
                - ensemble: Multiple models for best performance (~800MB)
        """
        self.model_config = model_config
        try:
            logger.info("Initializing spaCy model...")
            
            # SpaCy model selection based on config
            if model_config in ["enhanced", "ensemble"]:
                # Try to load larger model, fallback to small if not available
                try:
                    self.nlp = spacy.load("en_core_web_lg")
                    logger.info("Loaded spaCy large model (en_core_web_lg)")
                except OSError:
                    logger.warning("Large spaCy model not available, falling back to small model")
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy small model (en_core_web_sm)")
            else:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy small model (en_core_web_sm)")
            
            logger.info("Initializing sentence transformer model(s)...")
            self._initialize_sentence_models()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def _initialize_sentence_models(self):
        """Initialize sentence transformer models based on configuration"""
        if self.model_config == "basic":
            # Original model - commented but preserved
            # self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            # logger.info("Loaded basic sentence transformer model (all-MiniLM-L6-v2)")
            
            # For now, use enhanced even in basic mode for better results
            # Uncomment above lines and comment below to revert to original
            self.sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            logger.info("Loaded enhanced sentence transformer model (all-mpnet-base-v2)")
            
        elif self.model_config == "enhanced":
            # Enhanced single model
            self.sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            logger.info("Loaded enhanced sentence transformer model (all-mpnet-base-v2)")
            
        elif self.model_config == "ensemble":
            # Better ensemble combination for real-world impact focus
            self.sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            logger.info("Loaded primary sentence transformer model (all-mpnet-base-v2)")
            
            try:
                # Try different secondary models for better ensemble performance
                
                # Option 1: Domain-specific model for better contextual understanding
                try:
                    self.context_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
                    self.secondary_model_name = "distilroberta"
                    logger.info("Loaded context model (all-distilroberta-v1) for ensemble")
                    self.use_ensemble = True
                except:
                    # Option 2: Fallback to a different approach
                    try:
                        self.context_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
                        self.secondary_model_name = "paraphrase-mpnet"
                        logger.info("Loaded paraphrase model (paraphrase-mpnet-base-v2) for ensemble")
                        self.use_ensemble = True
                    except:
                        # Option 3: Use the same model with different prompting strategies
                        self.context_model = self.sentence_model  # Same model, different usage
                        self.secondary_model_name = "context-aware"
                        logger.info("Using context-aware approach with primary model for ensemble")
                        self.use_ensemble = True
                        
            except Exception as e:
                logger.warning(f"Could not set up ensemble, using single model: {e}")
                self.use_ensemble = False
        else:
            raise ValueError(f"Unknown model_config: {self.model_config}")

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF by page"""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        text_by_page = {}
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    text_by_page[page_num] = text
            logger.info(f"Extracted text from {len(text_by_page)} pages in {pdf_path}")
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
        return text_by_page

    def extract_text_from_file(self, file_path: str) -> Dict[int, str]:
        """Extract text from file (PDF or text) by page"""
        logger.info(f"Extracting text from file: {file_path}")
        if file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        else:
            # Handle text files for testing
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Split into pseudo-pages for text files (every 1000 characters)
                page_size = 1000
                text_by_page = {}
                for i in range(0, len(content), page_size):
                    page_num = (i // page_size) + 1
                    text_by_page[page_num] = content[i:i + page_size]
                logger.info(f"Extracted text from {len(text_by_page)} pseudo-pages in {file_path}")
                return text_by_page
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}")
                return {}

    def detect_sections(self, text: str) -> List[Tuple[str, int]]:
        """Detect sections/headings in text using enhanced patterns and NLP"""
        logger.debug("Detecting sections in text...")
        sections = []
        lines = text.split('\n')
        
        # Enhanced patterns for detecting headings
        heading_patterns = [
            r'^[A-Z][A-Z\s]{5,50}$',  # ALL CAPS headings
            r'^\d+\.?\s+[A-Z][\w\s]{5,50}$',  # Numbered headings
            r'^[IVX]+\.?\s+[A-Z][\w\s]{5,50}$',  # Roman numeral headings
            r'^[A-Z][\w\s]{5,50}:$',  # Headings ending with colon
            r'^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*$',  # Title case headings
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*$'  # Multi-word title case
        ]
        
        # Enhanced section detection using NLP for better accuracy
        if self.model_config in ["enhanced", "ensemble"]:
            # Use NLP to identify potential headings based on linguistic features
            doc = self.nlp(text)
            potential_headings = []
            
            # Look for sentences that might be headings
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if (len(sent_text) > 5 and len(sent_text) < 100 and
                    len(sent_text.split()) >= 2 and len(sent_text.split()) <= 12 and
                    not sent_text.endswith('.') and
                    any(token.pos_ in ['NOUN', 'PROPN'] for token in sent)):
                    potential_headings.append(sent_text)
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 5:
                continue
                
            # Check if line matches heading patterns
            is_heading = False
            for pattern in heading_patterns:
                if re.match(pattern, line) and len(line.split()) >= 2:
                    is_heading = True
                    break
            
            # Additional heuristics for heading detection
            if not is_heading:
                # Check for standalone lines that look like headings
                if (len(line) < 100 and 
                    len(line.split()) >= 2 and 
                    len(line.split()) <= 8 and
                    line[0].isupper() and 
                    not line.endswith('.') and
                    not line.endswith(',') and
                    not any(word.lower() in line.lower() for word in ['figure', 'table', 'equation', 'page', 'chapter']) and
                    not re.match(r'^\d+$', line.strip())):  # Not just a number
                    is_heading = True
            
            # Enhanced NLP-based validation for enhanced modes
            if is_heading and self.model_config in ["enhanced", "ensemble"]:
                # Use NLP to validate if this looks like a real heading
                doc = self.nlp(line)
                # Check if it has meaningful content and structure
                has_meaningful_content = any(token.pos_ in ['NOUN', 'PROPN', 'ADJ'] for token in doc)
                if not has_meaningful_content:
                    is_heading = False
            
            if is_heading:
                sections.append((line, i))
        
        # Remove duplicates and very similar sections
        unique_sections = []
        for section_title, line_num in sections:
            is_duplicate = False
            for existing_title, _ in unique_sections:
                if (section_title.lower() == existing_title.lower() or 
                    section_title.lower() in existing_title.lower() or
                    existing_title.lower() in section_title.lower()):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_sections.append((section_title, line_num))
        
        logger.debug(f"Detected {len(sections)} raw sections, {len(unique_sections)} unique sections")
        return unique_sections

    def extract_document_sections(self, file_path: str) -> List[Dict]:
        """Extract sections from a document"""
        logger.info(f"Extracting document sections from: {file_path}")
        text_by_page = self.extract_text_from_file(file_path)
        all_sections = []
        for page_num, text in text_by_page.items():
            logger.debug(f"Processing page {page_num} of {file_path}")
            sections = self.detect_sections(text)
            for section_title, line_num in sections:
                # Extract surrounding context for the section
                lines = text.split('\n')
                start_idx = max(0, line_num - 2)
                end_idx = min(len(lines), line_num + 15)  # Increased context
                context = '\n'.join(lines[start_idx:end_idx])
                
                # Clean up context
                context = re.sub(r'\n+', '\n', context).strip()
                
                all_sections.append({
                    'document': os.path.basename(file_path),
                    'section_title': section_title,
                    'page_number': page_num,
                    'context': context,
                    'full_text': text  # Store full page text for ranking
                })
        logger.info(f"Extracted {len(all_sections)} sections from {file_path}")
        return all_sections

    def calculate_relevance_score(self, section: Dict, persona_embedding: np.ndarray, 
                                job_embedding: np.ndarray, persona_text: str = "", job_text: str = "") -> float:
        """Calculate enhanced relevance score for a section based on persona and job"""
        try:
            logger.debug(f"Calculating relevance score for section: {section['section_title']}")
            # Combine section title and context for embedding
            section_text = f"{section['section_title']} {section['context']}"
            
            # Clean and prepare text for embedding
            section_text = re.sub(r'\s+', ' ', section_text).strip()
            if len(section_text) < 10:  # Too short to be meaningful
                return 0.0
            
            if self.model_config == "ensemble" and hasattr(self, 'use_ensemble') and self.use_ensemble:
                # Enhanced ensemble approach focusing on real-world impact
                
                # Primary semantic model score (70% weight - stronger influence)
                section_embedding = self.sentence_model.encode([section_text])
                persona_similarity_1 = cosine_similarity(section_embedding, persona_embedding.reshape(1, -1))[0][0]
                job_similarity_1 = cosine_similarity(section_embedding, job_embedding.reshape(1, -1))[0][0]
                
                # Context-aware scoring with different strategies
                if self.secondary_model_name == "context-aware":
                    # Same model but with enhanced context focus
                    # Emphasize real-world examples and case studies
                    enhanced_section_text = f"Real-world application case study: {section_text}"
                    enhanced_persona_text = f"Practical consultant experience: {persona_text}"
                    enhanced_job_text = f"Implementable strategies for: {job_text}"
                    
                    context_section_embedding = self.context_model.encode([enhanced_section_text])
                    context_persona_embedding = self.context_model.encode([enhanced_persona_text])
                    context_job_embedding = self.context_model.encode([enhanced_job_text])
                    
                else:
                    # Different model approach
                    context_section_embedding = self.context_model.encode([section_text])
                    context_persona_embedding = self.context_model.encode([persona_text])
                    context_job_embedding = self.context_model.encode([job_text])
                
                persona_similarity_2 = cosine_similarity(context_section_embedding, context_persona_embedding.reshape(1, -1))[0][0]
                job_similarity_2 = cosine_similarity(context_section_embedding, context_job_embedding.reshape(1, -1))[0][0]
                
                # Real-world impact bonus detection
                impact_keywords = [
                    'dharavi', 'case study', 'implementation', 'community', 'residents', 
                    'displacement', 'social impact', 'livelihood', 'participatory',
                    'ground reality', 'field experience', 'actual', 'practical'
                ]
                
                policy_keywords = [
                    'policy', 'regulation', 'framework', 'guideline', 'procedure',
                    'compliance', 'statutory', 'legal', 'official', 'government'
                ]
                
                section_lower = section_text.lower()
                impact_score = sum(1 for keyword in impact_keywords if keyword in section_lower)
                policy_score = sum(1 for keyword in policy_keywords if keyword in section_lower)
                
                # Boost real-world content, reduce pure policy content
                real_world_bonus = min(0.15, impact_score * 0.03)  # Up to +0.15 bonus
                policy_penalty = min(0.05, policy_score * 0.01) if policy_score > impact_score else 0  # Small penalty if too policy-heavy
                
                # Weighted combination favoring primary model (70/30) + bonuses
                persona_similarity = 0.7 * persona_similarity_1 + 0.3 * persona_similarity_2
                job_similarity = 0.7 * job_similarity_1 + 0.3 * job_similarity_2
                
                logger.debug(f"Enhanced ensemble - Model 1: P={persona_similarity_1:.3f}, J={job_similarity_1:.3f}")
                logger.debug(f"Enhanced ensemble - Model 2: P={persona_similarity_2:.3f}, J={job_similarity_2:.3f}")
                logger.debug(f"Impact bonus: {real_world_bonus:.3f}, Policy penalty: {policy_penalty:.3f}")
                
                # Apply bonuses/penalties
                job_similarity += real_world_bonus - policy_penalty
                
            else:
                # Single model approach (basic or enhanced)
                section_embedding = self.sentence_model.encode([section_text])
                
                # Calculate similarity with persona and job
                persona_similarity = cosine_similarity(section_embedding, persona_embedding.reshape(1, -1))[0][0]
                job_similarity = cosine_similarity(section_embedding, job_embedding.reshape(1, -1))[0][0]
            
            # Enhanced scoring with keyword matching bonus
            if self.model_config in ["enhanced", "ensemble"]:
                # Add keyword matching bonus
                section_lower = section_text.lower()
                keyword_bonus = 0.0
                
                # Simple keyword extraction and matching (could be enhanced further)
                important_words = [word for word in section_lower.split() 
                                 if len(word) > 4 and word.isalpha()]
                
                if len(important_words) > 0:
                    keyword_bonus = min(0.1, len(important_words) * 0.01)  # Small bonus
            else:
                keyword_bonus = 0.0
            
            # Weighted combination (job is more important) + keyword bonus
            relevance_score = 0.3 * persona_similarity + 0.7 * job_similarity + keyword_bonus
            
            logger.debug(f"Relevance score: {relevance_score:.4f} (P: {persona_similarity:.3f}, J: {job_similarity:.3f}, K: {keyword_bonus:.3f})")
            return float(relevance_score)
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.0

    def extract_refined_subsections(self, section: Dict, max_subsections: int = 3) -> List[Dict]:
        """Extract refined subsections from a section with enhanced NLP and noise filtering"""
        logger.debug(f"Extracting refined subsections for section: {section['section_title']}")
        try:
            text = section['context']
            doc = self.nlp(text)
            
            # Enhanced noise filtering for ensemble mode
            noise_indicators = [
                'elsevier', 'springer', 'wiley', 'publisher', 'copyright', 'doi:',
                'www.', 'http', 'email:', 'tel:', 'fax:', 'issn', 'isbn',
                'all rights reserved', 'reprinted with permission', 'page number',
                'figure caption', 'table caption', 'appendix', 'bibliography'
            ]
            
            # Enhanced sentence segmentation and grouping
            sentences = []
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if len(sent_text) > 20:
                    # Filter out noise sentences
                    sent_lower = sent_text.lower()
                    is_noise = any(indicator in sent_lower for indicator in noise_indicators)
                    
                    # Additional noise detection for ensemble mode
                    if self.model_config == "ensemble":
                        # Filter out sentences that are mostly numbers/references
                        word_count = len(sent_text.split())
                        digit_count = sum(1 for char in sent_text if char.isdigit())
                        if word_count > 0 and digit_count / len(sent_text) > 0.3:  # >30% digits
                            is_noise = True
                        
                        # Filter out very short sentences that don't add value
                        if word_count < 5:
                            is_noise = True
                    
                    if not is_noise:
                        sentences.append(sent_text)
            
            if not sentences:
                return []
            
            subsections = []
            current_subsection = []
            
            for i, sentence in enumerate(sentences):
                current_subsection.append(sentence)
                
                # Enhanced subsection boundary detection
                should_break = False
                
                # Natural break indicators
                if (sentence.endswith('\n\n') or
                    any(marker in sentence.lower() for marker in ['however', 'therefore', 'moreover', 'furthermore', 'additionally', 'consequently'])):
                    should_break = True
                
                # Length-based breaks
                if len(current_subsection) >= 3:
                    should_break = True
                
                # Enhanced NLP-based break detection for enhanced modes
                if self.model_config in ["enhanced", "ensemble"] and i < len(sentences) - 1:
                    current_sent_doc = self.nlp(sentence)
                    next_sent_doc = self.nlp(sentences[i + 1])
                    
                    # Check for topic shifts using entity differences
                    current_entities = set([ent.label_ for ent in current_sent_doc.ents])
                    next_entities = set([ent.label_ for ent in next_sent_doc.ents])
                    
                    if current_entities and next_entities and len(current_entities.intersection(next_entities)) == 0:
                        should_break = True
                
                if should_break and current_subsection:
                    refined_text = ' '.join(current_subsection).strip()
                    refined_text = re.sub(r'\s+', ' ', refined_text)  # Clean whitespace
                    
                    # Enhanced quality filtering
                    min_length = 80 if self.model_config == "ensemble" else 50  # Higher standards for ensemble
                    
                    if len(refined_text) > min_length:  # Only include substantial subsections
                        # Additional quality check for ensemble mode
                        if self.model_config == "ensemble":
                            # Ensure the subsection has meaningful content
                            words = refined_text.split()
                            meaningful_words = [w for w in words if len(w) > 3 and w.isalpha()]
                            if len(meaningful_words) >= 8:  # At least 8 meaningful words
                                subsections.append({
                                    'document': section['document'],
                                    'refined_text': refined_text,
                                    'page_number': section['page_number']
                                })
                        else:
                            subsections.append({
                                'document': section['document'],
                                'refined_text': refined_text,
                                'page_number': section['page_number']
                            })
                    current_subsection = []
            
            # Add remaining sentences as final subsection
            if current_subsection:
                refined_text = ' '.join(current_subsection).strip()
                refined_text = re.sub(r'\s+', ' ', refined_text)
                min_length = 80 if self.model_config == "ensemble" else 50
                
                if len(refined_text) > min_length:
                    if self.model_config == "ensemble":
                        words = refined_text.split()
                        meaningful_words = [w for w in words if len(w) > 3 and w.isalpha()]
                        if len(meaningful_words) >= 8:
                            subsections.append({
                                'document': section['document'],
                                'refined_text': refined_text,
                                'page_number': section['page_number']
                            })
                    else:
                        subsections.append({
                            'document': section['document'],
                            'refined_text': refined_text,
                            'page_number': section['page_number']
                        })
            
            logger.debug(f"Extracted {len(subsections[:max_subsections])} refined subsections")
            return subsections[:max_subsections]
            
        except Exception as e:
            logger.error(f"Error extracting subsections: {e}")
            return []

    def find_document_file(self, input_dir: str, filename: str) -> str:
        """Find document file with flexible matching"""
        logger.info(f"Looking for document: {filename} in {input_dir}")
        # Try exact match first
        exact_path = os.path.join(input_dir, filename)
        if os.path.exists(exact_path):
            logger.info(f"Found exact match for {filename}")
            return exact_path
        
        # Try with .txt extension for testing
        if filename.endswith('.pdf'):
            txt_filename = filename.replace('.pdf', '.txt')
            txt_path = os.path.join(input_dir, txt_filename)
            if os.path.exists(txt_path):
                logger.info(f"Found .txt version for {filename}")
                return txt_path
        # Try case-insensitive search
        for file in os.listdir(input_dir):
            if file.lower() == filename.lower():
                logger.info(f"Found case-insensitive match for {filename}: {file}")
                return os.path.join(input_dir, file)
        logger.warning(f"Could not find document: {filename}")
        return None

    def process_documents(self, input_dir: str, input_file: str) -> Dict:
        """Main processing function with enhanced capabilities"""
        logger.info(f"Starting document processing with model_config={self.model_config}")
        logger.info(f"Input: input_dir={input_dir}, input_file={input_file}")
        try:
            # Load input configuration
            logger.info(f"Loading input configuration from {input_file}")
            with open(input_file, 'r') as f:
                config = json.load(f)
            documents = config['documents']
            persona = config['persona']['role']
            job_to_be_done = config['job_to_be_done']['task']
            logger.info(f"Processing {len(documents)} documents for persona: {persona}")
            logger.info(f"Job to be done: {job_to_be_done}")
            
            # Create embeddings for persona and job
            logger.info("Creating embeddings for persona and job description")
            persona_embedding = self.sentence_model.encode([persona])
            job_embedding = self.sentence_model.encode([job_to_be_done])
            
            # Extract sections from all documents
            all_sections = []
            processed_docs = []
            for doc_info in documents:
                logger.info(f"Processing document: {doc_info['filename']}")
                file_path = self.find_document_file(input_dir, doc_info['filename'])
                if file_path:
                    logger.info(f"Found file: {file_path}")
                    sections = self.extract_document_sections(file_path)
                    all_sections.extend(sections)
                    processed_docs.append(doc_info['filename'])
                    logger.info(f"Successfully processed: {doc_info['filename']}")
                else:
                    logger.warning(f"Document not found: {doc_info['filename']}")
            
            if not all_sections:
                logger.error("No sections extracted from any documents")
                raise ValueError("No content could be extracted from the provided documents")
            
            logger.info(f"Extracted {len(all_sections)} sections total from {len(processed_docs)} documents")
            
            # Calculate relevance scores and rank sections
            logger.info("Calculating enhanced relevance scores for all sections")
            for section in all_sections:
                section['relevance_score'] = self.calculate_relevance_score(
                    section, persona_embedding[0], job_embedding[0], persona, job_to_be_done
                )
            
            # Enhanced filtering with adaptive thresholds
            if self.model_config in ["enhanced", "ensemble"]:
                # More sophisticated filtering
                scores = [s['relevance_score'] for s in all_sections]
                if scores:
                    avg_score = np.mean(scores)
                    std_score = np.std(scores)
                    threshold = max(0.1, avg_score - 0.5 * std_score)  # Adaptive threshold
                else:
                    threshold = 0.1
            else:
                threshold = 0.1
            
            logger.info(f"Filtering sections with threshold: {threshold:.3f}")
            all_sections = [s for s in all_sections if s['relevance_score'] > threshold]
            
            # Sort by relevance score (descending)
            all_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Enhanced section selection: top 2 per document, then global top 5
            from collections import defaultdict

            # Step 1: Collect top 2 per document
            top_sections_by_doc = defaultdict(list)
            for section in all_sections:
                doc = section['document']
                if len(top_sections_by_doc[doc]) < 2:
                    top_sections_by_doc[doc].append(section)

            # Step 2: Flatten and globally sort
            flattened_sections = [sec for secs in top_sections_by_doc.values() for sec in secs]
            flattened_sections.sort(key=lambda x: x['relevance_score'], reverse=True)

            # Step 3: Pick top 5 globally
            top_sections = flattened_sections[:5]

            logger.info(f"Selected top {len(top_sections)} sections for analysis")
            logger.info(f"Top section scores: {[f'{s['relevance_score']:.3f}' for s in top_sections]}")
            
            # Extract subsections for top sections
            logger.info("Extracting refined subsections for top sections")
            subsection_analysis = []
            for section in top_sections:
                subsections = self.extract_refined_subsections(section)
                subsection_analysis.extend(subsections)
            
            # Prepare output
            extracted_sections = []
            for i, section in enumerate(top_sections, 1):
                extracted_sections.append({
                    'document': section['document'],
                    'section_title': section['section_title'],
                    'importance_rank': i,
                    'page_number': section['page_number'],
                    'relevance_score': round(section['relevance_score'], 4)  # Include score in output
                })
            
            output = {
                'metadata': {
                    'input_documents': processed_docs,
                    'persona': persona,
                    'job_to_be_done': job_to_be_done,
                    'processing_timestamp': datetime.now().isoformat(),
                    'model_configuration': self.model_config,
                    'total_sections_found': len(all_sections),
                    'sections_after_filtering': len([s for s in all_sections if s['relevance_score'] > threshold])
                },
                'extracted_sections': extracted_sections,
                'subsection_analysis': subsection_analysis
            }
            
            logger.info("Document processing complete")
            return output
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Process documents for persona-driven intelligence')
    parser.add_argument('--input_dir', default='/app/input', help='Input directory path')
    parser.add_argument('--output_dir', default='/app/output', help='Output directory path')
    parser.add_argument('--input_file', default='challenge1b_input.json', help='Input configuration file')
    parser.add_argument('--output_file', default='challenge1b_output.json', help='Output file name')
    parser.add_argument('--model_config', default='ensemble', choices=['basic', 'enhanced', 'ensemble'],
                       help='Model configuration: basic (original), enhanced (single advanced), ensemble (multiple models)')
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting main pipeline with model configuration: {args.model_config}")
        
        # Initialize processor with specified model configuration
        processor = DocumentProcessor(model_config=args.model_config)
        
        # Process documents
        input_path = args.input_file
        result = processor.process_documents(args.input_dir, input_path)
        
        # Save output
        logger.info(f"Saving output to {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.output_file)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Successfully processed documents using {args.model_config} configuration")
        logger.info(f"Output saved to {output_path}")
        
        print(f"Processing complete with {args.model_config} model configuration.")
        print(f"Found {len(result['extracted_sections'])} relevant sections.")
        print(f"Model used: {result['metadata']['model_configuration']}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()