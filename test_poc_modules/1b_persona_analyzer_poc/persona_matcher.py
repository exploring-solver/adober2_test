# poc-modules/1b_persona_analyzer_poc/persona_matcher.py

import json
import re
from typing import Dict, List, Set
from collections import defaultdict

class PersonaAnalyzer:
    def __init__(self):
        # Predefined persona knowledge base
        self.persona_keywords = {
            "researcher": ["methodology", "literature", "analysis", "study", "research", "findings", "data", "hypothesis", "experiment"],
            "student": ["study", "exam", "learn", "understand", "concept", "definition", "example", "practice", "tutorial"],
            "analyst": ["trends", "performance", "metrics", "revenue", "market", "competition", "strategy", "growth", "financial"],
            "manager": ["strategy", "team", "leadership", "decision", "planning", "objectives", "resources", "budget"],
            "developer": ["implementation", "code", "system", "architecture", "technical", "framework", "api", "development"],
            "salesperson": ["customer", "product", "features", "benefits", "pricing", "market", "competition", "value proposition"]
        }
        
        self.job_keywords = {
            "literature review": ["background", "previous work", "related work", "survey", "overview", "comparison"],
            "exam preparation": ["key concepts", "important", "summary", "definition", "formula", "example"],
            "financial analysis": ["revenue", "profit", "costs", "financial", "budget", "performance", "metrics"],
            "market analysis": ["market", "competition", "trends", "customers", "strategy", "positioning"],
            "technical implementation": ["implementation", "design", "architecture", "system", "technical", "solution"]
        }
    
    def analyze_persona(self, persona_text: str) -> Dict:
        """Extract persona characteristics and focus areas"""
        persona_text = persona_text.lower()
        
        # Detect persona type
        detected_types = []
        for persona_type, keywords in self.persona_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in persona_text)
            if matches > 0:
                detected_types.append((persona_type, matches))
        
        # Sort by relevance
        detected_types.sort(key=lambda x: x[1], reverse=True)
        primary_persona = detected_types[0][0] if detected_types else "general"
        
        # Extract expertise areas
        expertise_areas = self._extract_expertise_areas(persona_text)
        
        # Extract focus keywords
        focus_keywords = self._extract_focus_keywords(persona_text, primary_persona)
        
        return {
            "primary_type": primary_persona,
            "all_types": [t[0] for t in detected_types[:3]],  # Top 3
            "expertise_areas": expertise_areas,
            "focus_keywords": focus_keywords,
            "raw_text": persona_text
        }
    
    def analyze_job_to_be_done(self, job_text: str) -> Dict:
        """Analyze the specific job/task to be accomplished"""
        job_text = job_text.lower()
        
        # Detect job type
        detected_jobs = []
        for job_type, keywords in self.job_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in job_text)
            if matches > 0:
                detected_jobs.append((job_type, matches))
        
        detected_jobs.sort(key=lambda x: x[1], reverse=True)
        primary_job = detected_jobs[0][0] if detected_jobs else "general task"
        
        # Extract action keywords
        action_keywords = self._extract_action_keywords(job_text)
        
        # Extract target topics
        target_topics = self._extract_topics(job_text)
        
        return {
            "primary_job": primary_job,
            "all_jobs": [j[0] for j in detected_jobs[:2]],
            "action_keywords": action_keywords,
            "target_topics": target_topics,
            "raw_text": job_text
        }
    
    def _extract_expertise_areas(self, text: str) -> List[str]:
        """Extract domain expertise from persona description"""
        # Look for common expertise patterns
        expertise_patterns = [
            r'in (\w+(?:\s+\w+){0,2})',  # "in machine learning"
            r'of (\w+(?:\s+\w+){0,2})',  # "of computer science"
            r'(\w+) expert',              # "AI expert"
            r'specialist in (\w+(?:\s+\w+){0,2})'  # "specialist in data science"
        ]
        
        areas = []
        for pattern in expertise_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            areas.extend(matches)
        
        # Clean and filter
        areas = [area.strip() for area in areas if len(area.strip()) > 2]
        return list(set(areas))[:5]  # Top 5 unique areas
    
    def _extract_focus_keywords(self, text: str, persona_type: str) -> List[str]:
        """Extract focus keywords based on persona type"""
        # Get base keywords for persona type
        base_keywords = self.persona_keywords.get(persona_type, [])
        
        # Add words from text that match domain vocabulary
        text_words = set(re.findall(r'\b\w{4,}\b', text))  # Words 4+ chars
        focus_words = text_words.intersection(set(base_keywords))
        
        return list(focus_words)[:10]  # Top 10
    
    def _extract_action_keywords(self, text: str) -> List[str]:
        """Extract action verbs and task-related keywords"""
        action_words = [
            "analyze", "review", "study", "examine", "compare", "evaluate",
            "summarize", "identify", "extract", "find", "prepare", "create",
            "understand", "learn", "research", "investigate", "assess"
        ]
        
        found_actions = []
        for word in action_words:
            if word in text:
                found_actions.append(word)
        
        return found_actions
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract specific topics mentioned in job description"""
        # Look for quoted topics or capitalized terms
        topics = []
        
        # Quoted phrases
        quoted = re.findall(r'"([^"]+)"', text)
        topics.extend(quoted)
        
        # Capitalized terms (potential topics)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        topics.extend(capitalized)
        
        # Technical terms (words ending in common suffixes)
        technical = re.findall(r'\b\w+(?:tion|ness|ment|ity|ology|graphy)\b', text)
        topics.extend(technical)
        
        # Clean and deduplicate
        topics = [topic.strip() for topic in topics if len(topic.strip()) > 3]
        return list(set(topics))[:8]  # Top 8 unique topics

# Test the analyzer
if __name__ == "__main__":
    print("Round 1B Persona Analyzer PoC")
    print("=" * 50)
    
    analyzer = PersonaAnalyzer()
    
    # Test cases
    test_cases = [
        {
            "persona": "PhD Researcher in Computational Biology specializing in machine learning applications",
            "job": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
        },
        {
            "persona": "Investment Analyst with expertise in technology sector evaluation",
            "job": "Analyze revenue trends, R&D investments, and market positioning strategies"
        },
        {
            "persona": "Undergraduate Chemistry Student preparing for final exams",
            "job": "Identify key concepts and mechanisms for exam preparation on reaction kinetics"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print("-" * 30)
        print(f"Persona: {test_case['persona']}")
        print(f"Job: {test_case['job']}")
        
        persona_analysis = analyzer.analyze_persona(test_case['persona'])
        job_analysis = analyzer.analyze_job_to_be_done(test_case['job'])
        
        print(f"\nPersona Analysis:")
        print(f"  Type: {persona_analysis['primary_type']}")
        print(f"  Expertise: {persona_analysis['expertise_areas']}")
        print(f"  Focus Keywords: {persona_analysis['focus_keywords']}")
        
        print(f"\nJob Analysis:")
        print(f"  Primary Job: {job_analysis['primary_job']}")
        print(f"  Actions: {job_analysis['action_keywords']}")
        print(f"  Topics: {job_analysis['target_topics']}")
        
        print("\n" + "="*50)