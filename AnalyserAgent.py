import anthropic
from typing import Dict, Any
import json


class AnalyzerAgent:
    def __init__(self):
        self.ApateKey = "AddYourKey"
        self.client = anthropic.Client(api_key=self.ApateKey)
        self.model = "claude-3-sonnet-20240229"

    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Main function to analyze the prompt using Anthropic's LLM"""
        analysis_prompt = f"""Analyze the following manually crafted prompt. Understand the intent, context, and overall objective. Extract these key elements as a JSON Dictionary:

1. Role/Identity: The persona the AI is asked to emulate. Expected values: Specific role (e.g., "Data Scientist", "Legal Expert"), or "N/A" if not specified.

2. Context: The background information or setting for the prompt. Expected values: Brief description of the context, or "N/A" if not provided.

3. Objective: Clarify the user’s objective (e.g., concise output, informative, adhering to formatting rules, staying within word limits).

3. Prompt/Query: The specific question or task being asked. Expected values: The main task or question from the prompt.

4. Output_Format: Any specified format for the response. Expected values: Description of requested format (e.g., "Report with bullet points"), or "N/A" if not specified.

5. Examples: Any examples provided in the prompt. Expected values: List of examples, or "N/A" if none provided.

6. Tasks: Identify if it's a single task or combined tasks, and list them. Expected values: "Single Task: [task description]" or "Combined Tasks: [list of tasks]".

7. Instructions/Constrains: Clear and Comprehensive Enumerated List of detailed usual and unusual Particular Conditions, Restrictions and Constrains on the output. Expected values: "respond in bullet points" or "highlight at least 3 sections with titles by enclosing them in markdown format" or "N/A" if none provided. Break down and clarify complex instructions while preserving original intent.


8. Linguistic_Analysis: Provide a thorough linguistic analysis including:
   a. Word count
   b. Sentence count
   c. Average words per sentence
   d. Readability score (e.g., Flesch-Kincaid)
   e. Tone analysis (e.g., formal, casual, technical)
   f. Use of jargon or specialized vocabulary
   g. Grammar and syntax observations
   h. Any notable linguistic patterns or structures

9. Technicality_Level: Identify the level of technicality in the prompt. Expected values: 
   - "Low": Minimal technical terms, suitable for general audience
   - "Medium": Some technical terms, requires basic domain knowledge
   - "High": Many technical terms, requires advanced domain knowledge
   Provide a brief explanation for the assigned level.

Ensure all elements are present in the output. If any element is not found in the provided user prompt, use "N/A" as the content of that field.

Prompt to analyze: "{prompt}"

Provide your analysis as a JSON object where all strings should be enclosed by double quotation and no special characters in keys. 
Your response should only contain the JSON output, without any introductions or explanations.
Ensure correct and consistent JSON output formatting.
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": analysis_prompt}],
        )
        print("text analysis: ", response.content[0].text)

        analysis_result = self.parse_json_response(response.content[0].text)
        return analysis_result

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse the JSON response from the LLM"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse LLM response"}


if __name__ == "__main__":
    # Example usage
    analyzer = AnalyzerAgent()
    prompt = """Act as an expert data scientist. Given a dataset of customer transactions from an e-commerce website, analyze the purchasing patterns and provide insights on customer behavior. Present your findings in a report format with bullet points for key insights and include any relevant statistical measures. For example, you might look at frequency of purchases, average order value, or popular product categories."""
    prompt2 = """Act as a data scientist and analyze customer transactions."""
    analysis = analyzer.analyze_prompt(prompt2)
    print(json.dumps(analysis, indent=2))
