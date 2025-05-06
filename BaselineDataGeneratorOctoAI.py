import json
from octoai.text_gen import ChatMessage
from octoai.client import OctoAI
import os
from typing import List, Dict
import time


class OctoAIPromptRunner:
    def __init__(self):

        models_list = ["LLAMA_2_70B_CHAT_FP16", "meta-llama-3-70b-instruct"]

        self.api_key = "AddYourKey"
        self.client = OctoAI(api_key=self.api_key)
        self.model = "mixtral-8x7b-instruct"  #  "meta-llama-3.1-70b-instruct"

    def run_prompt(self, prompt: str) -> str:
        """Run a single prompt through the Anthropic model and return the response"""

        response = self.client.text_gen.create_chat_completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"Execute the following prompt. Ensure all instructions are strictly followed: {prompt}",
                }
            ],
            presence_penalty=0,
            temperature=0,
            max_tokens=2000,
        )
        time.sleep(10)
        return response.choices[0].message.content


def parse_input_file(file_path: str) -> List[Dict]:
    """Parse the input JSON file and return a list of prompt dictionaries"""
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def generate_output_file(
    input_data: List[Dict], output_file_path: str, runner: OctoAIPromptRunner
):
    """Generate the output file by running prompts and formatting responses"""
    with open(output_file_path, "w") as out_file:
        for item in input_data:
            prompt = item["prompt"]
            response = runner.run_prompt(prompt)

            # Format the output based on any specific instructions in the prompt
            if "all lowercase letters" in prompt.lower():
                response = response.lower()
            elif "all capital letters" in prompt.lower():
                response = response.upper()

            # Check for specific formatting requirements
            if "wrap the entire response with double quotation marks" in prompt.lower():
                response = f'"{response}"'

            output_line = json.dumps({"prompt": prompt, "response": response})
            out_file.write(output_line + "\n")


def main():
    input_file_path = "data/refined_prompt.jsonl"  # "data/refined_input_data_all.jsonl"  # "data/refined_input_data18.jsonl"
    output_file_path = "data/refined_output_Mixtral.jsonl"  # "data/refined_output_response_data_anthropic18.jsonl"

    runner = OctoAIPromptRunner()
    input_data = parse_input_file(input_file_path)
    generate_output_file(input_data, output_file_path, runner)

    print(f"Processing complete. Output written to {output_file_path}")


if __name__ == "__main__":
    main()
