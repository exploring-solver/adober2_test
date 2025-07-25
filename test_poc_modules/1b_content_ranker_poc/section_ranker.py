# poc-modules/1b_content_ranker_poc/section_ranker.py

import json
import re
from typing import Dict, List, Tuple
from collections import Counter
import math

class ContentRanker:
    def __init__(self):
        self.stopwords = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        ])
    
    def rank_sections(self, documents: List[Dict], persona_analysis: Dict, job_analysis: Dict) -> List[Dict]:
        """Rank document sections based on persona and job relevance"""
        all_sections = []
        
        # Extract sections from all documents
        for doc_idx, document in enumerate(documents):
            sections = self._extract_document_sections(document, doc_idx)
            all_sections.extend(sections)
        
        # Score each section
        scored_sections = []
        for section in all_sections:
            score = self._calculate_relevance_score(section, persona_analysis, job_analysis)
            section['relevance_score'] = score
            scored_sections.append(section)
        
        # Sort by relevance score
        scored_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Add importance rank
        for i, section in enumerate(scored_sections):
            section['importance_rank'] = i + 1
        
        return scored_sections
    
    def _extract_document_sections(self, document: Dict, doc_idx: int) -> List[Dict]:
        """Extract sections from a document based on its outline"""
        sections = []
        
        # Simulate document content (in real implementation, extract from PDF)
        doc_title = document.get('title', f'Document_{doc_idx}')
        outline = document.get('outline', [])
        
        # Create sections from outline
        for item in outline:
            # Simulate section content based on heading
            content = self._simulate_section_content(item['text'])
            
            section = {
                'document': doc_title,
                'document_index': doc_idx,
                'section_title': item['text'],
                'page_number': item['page'],
                'level': item['level'],
                'content': content,
                'word_count': len(content.split())
            }
            sections.append(section)
        
        return sections
    
    def _simulate_section_content(self, heading: str) -> str:
        """Simulate section content based on heading (for PoC purposes)"""
        # In real implementation, this would extract actual text from PDF
        content_templates = {
            'introduction': "This section provides an overview and background information. It covers the main concepts and establishes the foundation for understanding the topic.",
            'methodology': "This section describes the research methods, experimental design, data collection procedures, and analytical techniques used in the study.",
            'results': "This section presents the findings, data analysis results, statistical outcomes, and key observations from the research.",
            'conclusion': "This section summarizes the main findings, discusses implications, and provides recommendations for future work.",
            'background': "This section reviews relevant literature, previous research, and provides context for the current study.",
            'analysis': "This section contains detailed examination of data, trends, patterns, and statistical analysis with interpretations."
        }
        
        heading_lower = heading.lower()
        
        # Match heading to content template
        for key, template in content_templates.items():
            if key in heading_lower:
                return f"{heading}. {template}"
        
        # Default content
        return f"{heading}. This section contains detailed information about {heading.lower()}. It includes relevant concepts, explanations, and supporting details that are important for understanding this topic."
    
    def _calculate_relevance_score(self, section: Dict, persona_analysis: Dict, job_analysis: Dict) -> float:
        """Calculate relevance score for a section"""
        score = 0.0
        content = section['content'].lower()
        title = section['section_title'].lower()
        
        # 1. Persona keyword matching (40% weight)
        persona_score = self._score_persona_match(content + " " + title, persona_analysis)
        score += persona_score * 0.4
        
        # 2. Job keyword matching (40% weight)
        job_score = self._score_job_match(content + " " + title, job_analysis)
        score += job_score * 0.4
        
        # 3. Section importance (10% weight)
        importance_score = self._score_section_importance(section)
        score += importance_score * 0.1
        
        # 4. Content quality (10% weight)
        quality_score = self._score_content_quality(section)
        score += quality_score * 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _score_persona_match(self, text: str, persona_analysis: Dict) -> float:
        """Score how well text matches persona interests"""
        score = 0.0
        
        # Focus keywords
        focus_keywords = persona_analysis.get('focus_keywords', [])
        for keyword in focus_keywords:
            if keyword in text:
                score += 0.2
        
        # Expertise areas
        expertise_areas = persona_analysis.get('expertise_areas', [])
        for area in expertise_areas:
            if area.lower() in text:
                score += 0.3
        
        # Persona type keywords
        persona_type = persona_analysis.get('primary_type', '')
        if persona_type == 'researcher':
            research_terms = ['methodology', 'analysis', 'study', 'research', 'findings']
            score += sum(0.1 for term in research_terms if term in text)
        elif persona_type == 'student':
            student_terms = ['concept', 'definition', 'example', 'understand']
            score += sum(0.1 for term in student_terms if term in text)
        elif persona_type == 'analyst':
            analyst_terms = ['trends', 'performance', 'metrics', 'analysis']
            score += sum(0.1 for term in analyst_terms if term in text)
        
        return min(score, 1.0)
    
    def _score_job_match(self, text: str, job_analysis: Dict) -> float:
        """Score how well text matches job requirements"""
        score = 0.0
        
        # Action keywords
        action_keywords = job_analysis.get('action_keywords', [])
        for keyword in action_keywords:
            if keyword in text:
                score += 0.2
        
        # Target topics
        target_topics = job_analysis.get('target_topics', [])
        for topic in target_topics:
            if topic.lower() in text:
                score += 0.3
        
        # Job type specific scoring
        primary_job = job_analysis.get('primary_job', '')
        if 'literature review' in primary_job:
            review_terms = ['previous', 'background', 'related work', 'survey']
            score += sum(0.15 for term in review_terms if term in text)
        elif 'analysis' in primary_job:
            analysis_terms = ['data', 'results', 'findings', 'trends']
            score += sum(0.15 for term in analysis_terms if term in text)
        elif 'exam preparation' in primary_job:
            exam_terms = ['key', 'important', 'definition', 'concept']
            score += sum(0.15 for term in exam_terms if term in text)
        
        return min(score, 1.0)
    
    def _score_section_importance(self, section: Dict) -> float:
        """Score section based on structural importance"""
        level = section['level']
        title = section['section_title'].lower()
        
        # Level-based scoring
        level_scores = {'H1': 0.8, 'H2': 0.6, 'H3': 0.4}
        score = level_scores.get(level, 0.3)
        
        # Important section names
        important_sections = [
            'introduction', 'conclusion', 'summary', 'overview',
            'methodology', 'results', 'findings', 'analysis'
        ]
        
        if any(important in title for important in important_sections):
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_content_quality(self, section: Dict) -> float:
        """Score based on content quality indicators"""
        word_count = section['word_count']
        
        # Optimal length scoring (not too short, not too long)
        if 50 <= word_count <= 500:
            length_score = 1.0
        elif word_count < 20:
            length_score = 0.3
        elif word_count > 1000:
            length_score = 0.7
        else:
            length_score = 0.8
        
        return length_score
    
    def extract_sub_sections(self, top_sections: List[Dict], persona_analysis: Dict, job_analysis: Dict) -> List[Dict]:
        """Extract and rank sub-sections from top sections"""
        sub_sections = []
        
        for section in top_sections[:5]:  # Top 5 sections only
            # Simulate sub-section extraction
            sub_content_parts = self._split_into_subsections(section['content'])
            
            for i, sub_content in enumerate(sub_content_parts):
                if len(sub_content.strip()) > 50:  # Only meaningful subsections
                    sub_section = {
                        'document': section['document'],
                        'page_number': section['page_number'],
                        'parent_section': section['section_title'],
                        'subsection_index': i + 1,
                        'refined_text': sub_content.strip(),
                        'relevance_score': self._calculate_relevance_score({
                            'content': sub_content,
                            'section_title': section['section_title'],
                            'level': 'H3',
                            'word_count': len(sub_content.split())
                        }, persona_analysis, job_analysis)
                    }
                    sub_sections.append(sub_section)
        
        # Sort by relevance
        sub_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        return sub_sections[:10]  # Top 10 sub-sections
    
    def _split_into_subsections(self, content: str) -> List[str]:
        """Split content into meaningful subsections"""
        # Simple sentence-based splitting for PoC
        sentences = re.split(r'[.!?]+', content)
        
        # Group sentences into subsections (2-3 sentences each)
        subsections = []
        current_subsection = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                current_subsection.append(sentence)
                if len(current_subsection) >= 2:  # 2-3 sentences per subsection
                    subsections.append('. '.join(current_subsection) + '.')
                    current_subsection = []
        
        # Add remaining sentences
        if current_subsection:
            subsections.append('. '.join(current_subsection) + '.')
        
        return subsections

# Test the ranker
if __name__ == "__main__":
    print("Round 1B Content Ranker PoC")
    print("=" * 50)
    
    ranker = ContentRanker()
    
    # Sample documents (simulated from Round 1A output)
    sample_documents = [
        {
            "title": "Graph Neural Networks for Drug Discovery",
            "outline": [
                {"level": "H1", "text": "1. Introduction", "page": 1},
                {"level": "H2", "text": "1.1 Background", "page": 2},
                {"level": "H1", "text": "2. Methodology", "page": 3},
                {"level": "H2", "text": "2.1 Data Collection", "page": 4},
                {"level": "H1", "text": "3. Results", "page": 5}
            ]
        },
        {
            "title": "Machine Learning in Computational Biology",
            "outline": [
                {"level": "H1", "text": "1. Overview", "page": 1},
                {"level": "H2", "text": "1.1 Previous Work", "page": 2},
                {"level": "H1", "text": "2. Analysis", "page": 3},
                {"level": "H1", "text": "3. Performance Benchmarks", "page": 4}
            ]
        }
    ]
    
    # Sample persona and job analysis
    persona_analysis = {
        "primary_type": "researcher",
        "expertise_areas": ["computational biology", "machine learning"],
        "focus_keywords": ["methodology", "analysis", "research"]
    }
    
    job_analysis = {
        "primary_job": "literature review",
        "action_keywords": ["review", "analyze", "compare"],
        "target_topics": ["methodologies", "datasets", "benchmarks"]
    }
    
    # Test section ranking
    print("Testing Section Ranking...")
    ranked_sections = ranker.rank_sections(sample_documents, persona_analysis, job_analysis)
    
    print(f"\nTop 5 Ranked Sections:")
    print("-" * 40)
    for i, section in enumerate(ranked_sections[:5]):
        print(f"{i+1}. {section['section_title']} (Score: {section['relevance_score']:.3f})")
        print(f"   Document: {section['document']}")
        print(f"   Page: {section['page_number']}")
        print()
    
    # Test sub-section extraction
    print("Testing Sub-section Extraction...")
    sub_sections = ranker.extract_sub_sections(ranked_sections, persona_analysis, job_analysis)
    
    print(f"\nTop 3 Sub-sections:")
    print("-" * 40)
    for i, sub in enumerate(sub_sections[:3]):
        print(f"{i+1}. From: {sub['parent_section']}")
        print(f"   Text: {sub['refined_text'][:100]}...")
        print(f"   Score: {sub['relevance_score']:.3f}")
        print()