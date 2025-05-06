import json
import anthropic
from typing import Dict, Any, List
from AnalyserAgent import *
from SuggesterAgent import *


def parse_input_file(file_path: str) -> List[Dict]:
    """Parse the input JSON file and return a list of prompt dictionaries"""
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def generate_refined_input_file(
    input_data: List[Dict],
    output_file_path: str,
    analyzer: AnalyzerAgent,
    suggester: SuggesterAgent,
):
    """Generate a new input file with refined prompts"""
    with open(output_file_path, "w") as out_file:
        for item in input_data:
            original_prompt = item["prompt"]
            analysis = analyzer.analyze_prompt(original_prompt)
            refined_prompt = suggester.refine_prompt(original_prompt, analysis)

            # Ensure the refined prompt maintains the original formatting
            if original_prompt.isupper():
                refined_prompt = refined_prompt.upper()
            elif original_prompt.islower():
                refined_prompt = refined_prompt.lower()

            # Check for specific formatting requirements
            if "wrapped in double angular brackets" in original_prompt:
                if "<<" not in refined_prompt:
                    refined_prompt = f"<<{refined_prompt.split()[0]}>> {' '.join(refined_prompt.split()[1:])}"

            # Create a new item with the refined prompt
            new_item = item.copy()
            new_item["prompt"] = refined_prompt

            # Write the new item to the output file
            out_file.write(json.dumps(new_item) + "\n")


def main():
    input_file_path = "data/input_data3.jsonl"
    output_file_path = "data/refined_prompt_3.jsonl"

    analyzer = AnalyzerAgent()

    suggester = SuggesterAgent()

    input_data = parse_input_file(input_file_path)
    generate_refined_input_file(input_data, output_file_path, analyzer, suggester)

    print(f"Processing complete. Refined input file written to {output_file_path}")


if __name__ == "__main__":
    main()
