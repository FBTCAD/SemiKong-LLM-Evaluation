#!/usr/bin/env python3
import json
import sys
from typing import Any

def contains_keyword(item: Any, keyword: str) -> bool:
    """
    Recursively checks if `item` (which may be a dict, list, string, or scalar) 
    contains the keyword. Returns True if found, False otherwise.
    """
    if isinstance(item, dict):
        # Check all values in the dictionary
        return any(contains_keyword(val, keyword) for val in item.values())
    elif isinstance(item, list):
        # Check each element in the list
        return any(contains_keyword(elem, keyword) for elem in item)
    elif isinstance(item, str):
        # Check if keyword is in string (case-insensitive)
        return keyword.lower() in item.lower()
    else:
        # For numbers, bool, None, etc., skip or convert to string if desired
        return False

def main():
    if len(sys.argv) < 4:
        print("Usage: python search_and_extract_with_index.py <path_to_input_json> <keyword> <path_to_output_json>")
        sys.exit(1)

    input_json_file = sys.argv[1]
    keyword = sys.argv[2]
    output_json_file = sys.argv[3]

    # Load the JSON file
    try:
        with open(input_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_json_file}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{input_json_file}'. Make sure it is valid JSON.")
        sys.exit(1)

    # Ensure the JSON root is a list/array
    if not isinstance(data, list):
        print("The root of the JSON does not appear to be a list. Exiting.")
        sys.exit(1)

    # Collect matching entries, preserving original index
    matching_entries = []
    for idx, item in enumerate(data):
        if contains_keyword(item, keyword):
            matching_entries.append({
                "original_index": idx,
                "data": item
            })

    # Write matches to output file
    with open(output_json_file, "w", encoding="utf-8") as out_f:
        json.dump(matching_entries, out_f, ensure_ascii=False, indent=2)

    # Print results
    print(f"Found {len(matching_entries)} matching entries for keyword '{keyword}'.")
    print(f"Saved results to '{output_json_file}'.")

if __name__ == "__main__":
    main()
