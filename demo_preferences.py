#!/usr/bin/env python3
"""
Simple demo script to show preferences integration
Run this to see how preferences affect search results
"""

from preferences_parser import PreferencesParser
import json

print("=" * 80)
print("PREFERENCES SYSTEM DEMO")
print("=" * 80)
print()

# Load preferences
print("Loading preferences from data/preferences.txt...")
prefs = PreferencesParser("data/preferences.txt")
config = prefs.get_config()

print("SUCCESS - Preferences loaded!")
print()

# Show configuration
print("-" * 80)
print("CURRENT CONFIGURATION:")
print("-" * 80)
print(f"Results per search (k): {config['k']}")
print(f"Max results per page: {config['max_results']}")
print(f"Min similarity score: {config['min_similarity_score']}")
print()

print("Default Filters:")
for key, value in config['default_filters'].items():
    print(f"  - {key}: {value}")
print()

print("Tag Boost Rules:")
if config['boost_rules']['tags']:
    for tag, boost in config['boost_rules']['tags'].items():
        print(f"  - {tag}: {boost}x")
else:
    print("  (none configured)")
print()

print("Ranking Signals:")
for signal, boost in config['ranking_signals'].items():
    print(f"  - {signal}: {boost}x")
print()

print("Response Configuration:")
for key, value in config['response'].items():
    print(f"  - {key}: {value}")
print()

print("Chat Configuration:")
for key, value in config['chat'].items():
    print(f"  - {key}: {value}")
print()

# Test query boost calculation
print("-" * 80)
print("TESTING QUERY BOOST CALCULATION:")
print("-" * 80)

test_product = {
    "product_id": "12345",
    "title": "Black Quick-Dry Running Shorts",
    "price": 45.0,
    "in_stock": True,
    "tags": ["quick-dry", "running", "black", "women"]
}

test_query = "black running shorts under $50"
base_score = 0.85

print(f"Query: '{test_query}'")
print(f"Product: {test_product['title']}")
print(f"Base Score: {base_score}")
print()

boosted_score, reasons = prefs.apply_query_boost(test_query, test_product, base_score)

print(f"Boosted Score: {boosted_score:.4f}")
print(f"Boost Reasons: {', '.join(reasons)}")
print(f"Total Boost: {boosted_score/base_score:.2f}x")
print()

# Test personalization
print("-" * 80)
print("TESTING PERSONALIZATION:")
print("-" * 80)

personalization_queries = [
    "comfortable chair for mom",
    "furniture for small apartments",
    "sofa for pets"
]

for query in personalization_queries:
    rules = prefs.get_personalization_rules(query)
    if rules:
        print(f"Query: '{query}'")
        print(f"  Personalization rules applied:")
        print(f"  {json.dumps(rules, indent=4)}")
    else:
        print(f"Query: '{query}' - No special personalization")
    print()

# Show business context for LLM
print("-" * 80)
print("BUSINESS CONTEXT FOR LLM:")
print("-" * 80)
context = prefs.format_business_rules_for_llm()
print(context)
print()

print("=" * 80)
print("DEMO COMPLETE")
print("=" * 80)
print()
print("To test with live API:")
print("1. Start API: python search_api.py")
print("2. In another terminal: python test_preferences_integration.py")
print()
print("To edit preferences:")
print("1. Edit: data/preferences.txt")
print("2. Save file")
print("3. Run this demo again to see changes")
print("4. OR call POST http://localhost:8000/admin/reload-preferences")

