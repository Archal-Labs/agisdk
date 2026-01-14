#!/usr/bin/env python3
"""
Extract prompts from JSON task files and create a CSV with columns:
prompt, possible, and websites.
"""

import json
import csv
import os
from pathlib import Path

def extract_tasks_to_csv(tasks_dir, output_csv):
    """
    Extract task data from JSON files and write to CSV.
    
    Args:
        tasks_dir: Directory containing JSON task files
        output_csv: Output CSV file path
    """
    tasks_dir = Path(tasks_dir)
    assert tasks_dir.is_dir(), f"Directory {tasks_dir} does not exist"
    
    rows = []
    
    # Read all JSON files in the directory
    json_files = sorted(tasks_dir.glob("*.json"))
    assert len(json_files) > 0, f"No JSON files found in {tasks_dir}"
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract prompt (goal field)
        prompt = data.get('goal', '')
        
        # Extract possible (boolean)
        possible = data.get('possible', False)
        
        # Extract websites and format as comma-separated names
        websites = data.get('websites', [])
        website_names = ', '.join([site.get('name', '') for site in websites])
        
        rows.append({
            'prompt': prompt,
            'possible': str(possible),
            'websites': website_names
        })
    
    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['prompt', 'possible', 'websites']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Extracted {len(rows)} tasks to {output_csv}")

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    tasks_dir = script_dir / 'multi-real' / 'tasks'
    output_csv = script_dir / 'multi-real' / 'tasks.csv'
    
    extract_tasks_to_csv(tasks_dir, output_csv)

