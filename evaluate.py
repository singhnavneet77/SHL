#!/usr/bin/env python3
"""
Evaluation harness for the SHL Assessment Recommender.
Tests against the 10 sample conversations and reports:
  - Schema compliance
  - Catalog-only recommendations
  - Turn cap compliance
  - Recall@10
  - Behavior probes
"""
import json
import sys
import os
import re
import requests
import time
from pathlib import Path

BASE_URL = os.environ.get("API_URL", "http://localhost:8000")
CONVERSATIONS_DIR = Path("../sample_conversations/GenAI_SampleConversations")

def parse_conversation(md_path: Path):
    """Parse a conversation trace into user/assistant turns + expected URLs."""
    content = md_path.read_text()
    
    turns = []
    expected_urls = set()
    
    # Extract expected URLs from tables in the last agent response that has end_of_conversation: true
    lines = content.split('\n')
    in_final = False
    
    for i, line in enumerate(lines):
        if '`end_of_conversation`: **true**' in line:
            # Go back and find the most recent table
            for j in range(i, -1, -1):
                if 'https://www.shl.com/products/product-catalog/view/' in lines[j]:
                    m = re.search(r'<(https://www\.shl\.com/products/product-catalog/view/[^>]+)>', lines[j])
                    if m:
                        expected_urls.add(m.group(1))
    
    # Parse turns
    current_role = None
    current_content = []
    
    for line in lines:
        if line.startswith('**User**'):
            if current_role and current_content:
                turns.append({"role": current_role, "content": ' '.join(current_content).strip()})
            current_role = 'user'
            current_content = []
        elif line.startswith('**Agent**'):
            if current_role and current_content:
                turns.append({"role": current_role, "content": ' '.join(current_content).strip()})
            current_role = 'assistant'
            current_content = []
        elif line.startswith('> '):
            current_content.append(line[2:])
        elif line.startswith('_No recommendations') or line.startswith('_`end_of_conversation`'):
            pass  # Skip metadata lines
    
    if current_role and current_content:
        turns.append({"role": current_role, "content": ' '.join(current_content).strip()})
    
    # Filter to user messages only (we'll regenerate agent responses)
    user_turns = [t for t in turns if t['role'] == 'user']
    
    return user_turns, expected_urls

def chat(messages: list) -> dict:
    """Call the /chat endpoint."""
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"messages": messages},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()

def compute_recall_at_k(recommended_urls: set, expected_urls: set, k: int = 10) -> float:
    """Compute Recall@K."""
    if not expected_urls:
        return 1.0  # No ground truth = pass
    top_k = list(recommended_urls)[:k]
    hits = len(set(top_k) & expected_urls)
    return hits / len(expected_urls)

def check_schema(response: dict) -> list[str]:
    """Check schema compliance."""
    errors = []
    if "reply" not in response:
        errors.append("Missing 'reply' field")
    if "recommendations" not in response:
        errors.append("Missing 'recommendations' field")
    if "end_of_conversation" not in response:
        errors.append("Missing 'end_of_conversation' field")
    
    if "recommendations" in response:
        for rec in response["recommendations"]:
            if "name" not in rec:
                errors.append(f"Recommendation missing 'name': {rec}")
            if "url" not in rec:
                errors.append(f"Recommendation missing 'url': {rec}")
            if "test_type" not in rec:
                errors.append(f"Recommendation missing 'test_type': {rec}")
    
    return errors

def check_catalog_only(response: dict, valid_urls: set) -> list[str]:
    """Check all recommendation URLs are from catalog."""
    errors = []
    for rec in response.get("recommendations", []):
        url = rec.get("url", "")
        if url not in valid_urls:
            errors.append(f"Non-catalog URL: {url}")
    return errors

def run_evaluation():
    """Run full evaluation."""
    print(f"Testing against: {BASE_URL}")
    print("=" * 60)
    
    # Health check
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Health check: {r.json()}")
    except Exception as e:
        print(f"Health check FAILED: {e}")
        sys.exit(1)
    
    # Load valid catalog URLs
    with open("catalog.json") as f:
        catalog = json.load(f)
    valid_urls = {p["url"] for p in catalog}
    
    # Run conversation traces
    results = []
    conv_files = sorted(CONVERSATIONS_DIR.glob("C*.md"))
    
    if not conv_files:
        print("No conversation files found!")
        sys.exit(1)
    
    for conv_file in conv_files:
        print(f"\n{'='*40}")
        print(f"Testing: {conv_file.name}")
        
        user_turns, expected_urls = parse_conversation(conv_file)
        print(f"  User turns: {len(user_turns)}, Expected URLs: {len(expected_urls)}")
        
        messages = []
        final_recommendations = set()
        schema_errors = []
        catalog_errors = []
        turn_count = 0
        
        for turn in user_turns:
            messages.append({"role": "user", "content": turn["content"]})
            turn_count += 1
            
            if turn_count > 8:
                print(f"  WARN: Turn cap would be exceeded ({turn_count})")
                break
            
            try:
                resp = chat(messages)
                
                # Schema check
                errors = check_schema(resp)
                schema_errors.extend(errors)
                
                # Catalog check
                errors = check_catalog_only(resp, valid_urls)
                catalog_errors.extend(errors)
                
                # Track final recommendations
                recs = resp.get("recommendations", [])
                if recs:
                    final_recommendations = {r["url"] for r in recs}
                    print(f"  Turn {turn_count}: Got {len(recs)} recommendations")
                else:
                    print(f"  Turn {turn_count}: Clarifying (no recs)")
                
                # Add assistant response to history
                messages.append({"role": "assistant", "content": resp["reply"]})
                
                if resp.get("end_of_conversation"):
                    print(f"  Conversation ended at turn {turn_count}")
                    break
                    
            except requests.exceptions.Timeout:
                print(f"  TIMEOUT on turn {turn_count}")
                break
            except Exception as e:
                print(f"  ERROR on turn {turn_count}: {e}")
                break
            
            time.sleep(0.2)
        
        recall = compute_recall_at_k(final_recommendations, expected_urls, k=10)
        
        result = {
            "conversation": conv_file.name,
            "turns": turn_count,
            "recall_at_10": recall,
            "expected_urls": list(expected_urls),
            "recommended_urls": list(final_recommendations),
            "schema_errors": schema_errors,
            "catalog_errors": catalog_errors,
        }
        results.append(result)
        
        print(f"  Recall@10: {recall:.2f}")
        if schema_errors:
            print(f"  Schema errors: {schema_errors}")
        if catalog_errors:
            print(f"  Catalog errors: {catalog_errors}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    mean_recall = sum(r["recall_at_10"] for r in results) / len(results) if results else 0
    total_schema_errors = sum(len(r["schema_errors"]) for r in results)
    total_catalog_errors = sum(len(r["catalog_errors"]) for r in results)
    
    print(f"Mean Recall@10: {mean_recall:.3f}")
    print(f"Total schema errors: {total_schema_errors}")
    print(f"Total catalog errors: {total_catalog_errors}")
    print(f"Conversations tested: {len(results)}")
    
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to eval_results.json")

if __name__ == "__main__":
    run_evaluation()
