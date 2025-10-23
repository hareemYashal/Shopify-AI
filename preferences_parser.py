#!/usr/bin/env python3
"""
Preferences Parser - Converts natural language business rules into structured search configuration
Supports both structured key=value format and natural language rules
"""

import os
import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path


class PreferencesParser:
    """Parse and manage search preferences from plain text files"""
    
    def __init__(self, preferences_path: str = "data/preferences.txt"):
        self.preferences_path = preferences_path
        self.last_modified = None
        self.raw_text = ""
        self.config = {}
        self.load()
    
    def load(self) -> bool:
        """Load and parse preferences file"""
        try:
            file_path = Path(self.preferences_path)
            if not file_path.exists():
                print(f"[WARN] Preferences file not found: {self.preferences_path}")
                return False
            
            # Check if file has been modified
            current_modified = file_path.stat().st_mtime
            if self.last_modified and current_modified == self.last_modified:
                return False  # No changes
            
            # Load file content
            with open(file_path, 'r', encoding='utf-8') as f:
                self.raw_text = f.read()
            
            self.last_modified = current_modified
            self.config = self._parse_preferences()
            print(f"[OK] Preferences loaded from {self.preferences_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading preferences: {e}")
            return False
    
    def _parse_preferences(self) -> Dict[str, Any]:
        """Parse preferences text into structured configuration"""
        config = {
            # Search behavior
            "k": 8,
            "max_results": 24,
            "min_similarity_score": 0.7,
            
            # Filters
            "default_filters": {
                "in_stock": True,
                "max_price": None,
                "min_price": None,
                "categories_exclude": [],
                "categories_include": []
            },
            
            # Boosting
            "boost_rules": {
                "tags": {},
                "categories": {},
                "attributes": {}
            },
            
            # Ranking signals
            "ranking_signals": {
                "in_stock_boost": 1.2,
                "new_arrivals_boost": 1.15,
                "high_rating_boost": 1.1,
                "exact_match_boost": 1.3,
                "color_match_boost": 1.25,
                "size_match_boost": 1.2
            },
            
            # Personalization rules
            "personalization": {},
            
            # Response formatting
            "response": {
                "tone": "friendly",
                "style": "bullets",
                "include_price": True,
                "include_availability": True,
                "include_reason": True,
                "max_items": 5
            },
            
            # Chat settings
            "chat": {
                "temperature": 0.3,
                "max_tokens": 300,
                "grounding_strict": True,
                "include_citations": True
            },
            
            # Business rules
            "business_rules": [],
            
            # Promotions
            "promotions": []
        }
        
        # Parse structured key=value pairs
        config = self._parse_structured_format(config)
        
        # Parse natural language rules
        config = self._parse_natural_language_rules(config)
        
        return config
    
    def _parse_structured_format(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse traditional key=value format"""
        lines = self.raw_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Parse specific keys
                if key == 'k':
                    config['k'] = int(value)
                
                elif key == 'max_results':
                    config['max_results'] = int(value)
                
                elif key == 'min_similarity_score':
                    config['min_similarity_score'] = float(value)
                
                elif key.startswith('filters.required'):
                    # Parse filters like: filters.required=in_stock:true
                    filter_parts = value.split(':')
                    if len(filter_parts) == 2:
                        filter_key, filter_val = filter_parts
                        if filter_key == 'in_stock':
                            config['default_filters']['in_stock'] = filter_val.lower() == 'true'
                
                elif key.startswith('filters.optional'):
                    # Parse optional filters like: price<=100
                    if '<=' in value:
                        filter_key, filter_val = value.split('<=')
                        if filter_key == 'price':
                            config['default_filters']['max_price'] = float(filter_val)
                
                elif key.startswith('boost.tags'):
                    # Parse boost rules like: quick-dry:1.2,women:1.1
                    boosts = value.split(',')
                    for boost in boosts:
                        if ':' in boost:
                            tag, factor = boost.split(':')
                            config['boost_rules']['tags'][tag.strip()] = float(factor)
                
                elif key == 'tone':
                    config['response']['tone'] = value
                
                elif key == 'style':
                    config['response']['style'] = value
                
                elif key == 'max_items':
                    config['response']['max_items'] = int(value)
                
                elif key == 'chat_temperature':
                    config['chat']['temperature'] = float(value)
                
                elif key == 'chat_max_tokens':
                    config['chat']['max_tokens'] = int(value)
        
        return config
    
    def _parse_natural_language_rules(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse natural language business rules"""
        text = self.raw_text.lower()
        
        # Extract number of results per page
        results_match = re.search(r'show\s+(\d+)\s+results?\s+per\s+page', text)
        if results_match:
            config['max_results'] = int(results_match.group(1))
        
        # Extract price guardrails
        price_match = re.search(r'hide\s+items?\s+over\s+\$?([\d,]+)', text)
        if price_match:
            price_str = price_match.group(1).replace(',', '')
            config['default_filters']['max_price'] = float(price_str)
        
        # Extract stock preference
        if 'hide out-of-stock' in text or 'in stock only' in text:
            config['default_filters']['in_stock'] = True
        
        if 'include sold out' in text or 'show out-of-stock' in text:
            config['default_filters']['in_stock'] = False
        
        # Extract boost rules for specific keywords
        boost_patterns = [
            (r'strong boost.*?match.*?color', 'color_match_boost', 1.3),
            (r'strong boost.*?match.*?size', 'size_match_boost', 1.25),
            (r'moderate boost.*?in-stock', 'in_stock_boost', 1.2),
            (r'moderate boost.*?new arrivals?', 'new_arrivals_boost', 1.2),
            (r'moderate boost.*?bestsellers?', 'bestseller_boost', 1.15),
            (r'moderate boost.*?4★\+.*?reviews?', 'high_rating_boost', 1.15),
        ]
        
        for pattern, key, default_boost in boost_patterns:
            if re.search(pattern, text):
                # Try to extract specific boost value
                boost_value_match = re.search(pattern + r'.*?(\d+\.?\d*)%?', text)
                if boost_value_match:
                    boost_val = float(boost_value_match.group(1))
                    if boost_val > 2:  # If it's a percentage like 40%
                        boost_val = 1 + (boost_val / 100)
                    config['ranking_signals'][key] = boost_val
                else:
                    config['ranking_signals'][key] = default_boost
        
        # Extract personalization rules
        personalization_patterns = [
            (r'"for mom"', 'for_mom', {
                'boost_attributes': ['armrests', 'high-back', 'padded'],
                'boost_tags': ['comfortable', 'supportive'],
                'min_seat_height': 17,
                'fabric_preference': ['velvet', 'fabric']
            }),
            (r'"for small apartments?"', 'for_small_apartment', {
                'max_width': 72,
                'boost_tags': ['compact', 'space-saving']
            }),
            (r'"for pets?"', 'for_pets', {
                'boost_tags': ['stain-resistant', 'washable', 'performance-fabric']
            }),
        ]
        
        for pattern, key, rules in personalization_patterns:
            if re.search(pattern, text):
                config['personalization'][key] = rules
        
        # Extract tone preferences
        tone_patterns = [
            (r'tone:?\s*friendly', 'friendly'),
            (r'tone:?\s*concise', 'concise'),
            (r'tone:?\s*matter-of-fact', 'matter-of-fact'),
            (r'tone:?\s*professional', 'professional'),
        ]
        
        for pattern, tone in tone_patterns:
            if re.search(pattern, text):
                config['response']['tone'] = tone
                break
        
        # Extract business rules as text snippets for LLM context
        business_rules = []
        
        # Look for numbered sections or bullet points
        rule_sections = re.findall(r'(?:^|\n)(?:\d+\)|•)\s*(.+?)(?=\n(?:\d+\)|•)|\n\n|$)', text, re.MULTILINE)
        if rule_sections:
            business_rules.extend([rule.strip() for rule in rule_sections if len(rule.strip()) > 20])
        
        config['business_rules'] = business_rules[:10]  # Keep top 10 most relevant
        
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config
    
    def should_boost_tag(self, tag: str) -> float:
        """Get boost factor for a tag (1.0 = no boost)"""
        return self.config['boost_rules']['tags'].get(tag, 1.0)
    
    def should_boost_category(self, category: str) -> float:
        """Get boost factor for a category (1.0 = no boost)"""
        return self.config['boost_rules']['categories'].get(category, 1.0)
    
    def get_default_filters(self) -> Dict[str, Any]:
        """Get default filters to apply"""
        return self.config['default_filters'].copy()
    
    def get_ranking_signals(self) -> Dict[str, float]:
        """Get ranking signal boost factors"""
        return self.config['ranking_signals'].copy()
    
    def get_personalization_rules(self, query: str) -> Dict[str, Any]:
        """Get personalization rules based on query context"""
        query_lower = query.lower()
        
        # Check for personalization triggers
        for trigger, rules in self.config['personalization'].items():
            trigger_words = trigger.replace('for_', '').replace('_', ' ')
            if trigger_words in query_lower:
                return rules
        
        return {}
    
    def get_response_config(self) -> Dict[str, Any]:
        """Get response formatting configuration"""
        return self.config['response'].copy()
    
    def get_chat_config(self) -> Dict[str, Any]:
        """Get chat/LLM configuration"""
        return self.config['chat'].copy()
    
    def get_business_context(self) -> str:
        """Get business rules as context for LLM"""
        rules = self.config.get('business_rules', [])
        if rules:
            return "Business Rules:\n" + "\n".join(f"- {rule}" for rule in rules[:5])
        return ""
    
    def apply_query_boost(self, query: str, product: Dict[str, Any], base_score: float) -> float:
        """Apply boost factors based on query and product attributes"""
        boosted_score = base_score
        boost_reasons = []
        
        query_lower = query.lower()
        
        # Tag boosting
        if 'tags' in product:
            for tag in product['tags']:
                tag_boost = self.should_boost_tag(tag)
                if tag_boost > 1.0:
                    boosted_score *= tag_boost
                    boost_reasons.append(f"tag:{tag}")
        
        # In-stock boosting
        if product.get('in_stock', False):
            in_stock_boost = self.config['ranking_signals'].get('in_stock_boost', 1.0)
            boosted_score *= in_stock_boost
            boost_reasons.append("in-stock")
        
        # Color match boosting
        colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 'gray', 'brown']
        for color in colors:
            if color in query_lower and color in product.get('title', '').lower():
                color_boost = self.config['ranking_signals'].get('color_match_boost', 1.0)
                boosted_score *= color_boost
                boost_reasons.append(f"color:{color}")
                break
        
        # Price-based boosting (if within query constraints)
        price_match = re.search(r'under\s+\$?(\d+)', query_lower)
        if price_match and 'price' in product:
            max_price = float(price_match.group(1))
            if product['price'] <= max_price:
                boosted_score *= 1.1
                boost_reasons.append("budget-match")
        
        return boosted_score, boost_reasons
    
    def format_business_rules_for_llm(self) -> str:
        """Format business rules as context for LLM prompts"""
        config = self.config
        
        context_parts = []
        
        # Add tone guidance
        tone = config['response']['tone']
        context_parts.append(f"Response Tone: {tone}")
        
        # Add key business rules
        if config.get('business_rules'):
            context_parts.append("\nKey Business Rules:")
            for rule in config['business_rules'][:5]:
                context_parts.append(f"- {rule}")
        
        # Add must-include information
        if config['response']['include_price']:
            context_parts.append("- Always mention price")
        if config['response']['include_availability']:
            context_parts.append("- Always mention stock availability")
        
        return "\n".join(context_parts)


# Global instance
_preferences_instance = None

def get_preferences() -> PreferencesParser:
    """Get or create global preferences instance"""
    global _preferences_instance
    if _preferences_instance is None:
        _preferences_instance = PreferencesParser()
    else:
        # Try to reload if file changed
        _preferences_instance.load()
    return _preferences_instance


def reload_preferences():
    """Force reload preferences from file"""
    global _preferences_instance
    _preferences_instance = PreferencesParser()
    return _preferences_instance

