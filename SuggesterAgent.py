from AnalyserAgent import *
import anthropic
from typing import Dict, Any, List
import json
import re
from openai import OpenAI


class SuggesterAgent:
    def __init__(self):
        self.api_key = "AddYourKey"
        self.client = anthropic.Client(api_key=self.api_key)
        self.model = "claude-3-sonnet-20240229"

    def refine_prompt(self, original_prompt: str, analysis: Dict[str, Any]) -> str:
        """Main method to refine the prompt, ask questions if needed, and optionally run the refined prompt"""

        user_inputs = {}

        refined_prompt = self._generate_refined_prompt(
            original_prompt, analysis, user_inputs
        )

        print("\nRefined Prompt:")
        print(refined_prompt)

        return refined_prompt

    def _generate_refined_prompt(
        self,
        original_prompt: str,
        analysis: Dict[str, Any],
        user_inputs: Dict[str, str],
    ) -> str:
        """Generate a refined and enhanced prompt based on the original prompt, analysis, and user inputs"""
        refinement_prompt = f"""
                        You are an expert prompt engineer you goal to optimise and enhance the provided manually crafted prompt to achieve improved instruction following.
                Given the original manually crafted prompt: {original_prompt},
                Given the detailed analyses of the original prompt components provided by an expert analyser agent:
                Role/Identity: {analysis.get('Role/Identity', 'N/A')}
                Context: {analysis.get('Context', 'N/A')}
                Objective: {analysis.get('Objective', 'N/A')}
                Prompt/Query: {analysis.get('Prompt/Query', 'N/A')}
                Output_Format: {analysis.get('Output_Format', 'N/A')}
                Examples: {analysis.get('Examples', 'N/A')}
                Tasks: {analysis.get('Tasks', 'N/A')}
                Instructions/Constrains: {analysis.get('Instructions/Constrains', 'N/A')}
                Linguistic_Analysis: {analysis.get('Linguistic_Analysis', 'N/A')}
                Technicality_Level: {analysis.get('Technicality_Level', 'N/A')}
 
                Firstly, understand the main objective of the prompt, the user intent, the context, and detailed prompt instructions/restriction/constraints.
 
                Secondly, identify any language issues, ambiguities, missing details, or structural issues in the manually crafted prompt.
 
                Thirdly, your task is to create an improved clear and concise version of the prompt that:
                1. Improves grammar and sentence’s structure of the original prompt.
                2. Enhances clarity and specificity of the instructions.
                3. Addresses any gaps or ambiguities identified in the provided analysis.
                4. Strictly maintains and effectively communicate the original intent of the prompt.
                5. Includes each and every instruction/restriction/constraint provided by the analyser agent analysis in the Instructions/Constrains field.
                6. Strictly Does not add instructions/conditions/restriction/constraint not present in original prompt. Maintains a straightforward and concise prompt. 
                7. Adheres to the below prompt format and incorporates the below elements in the refined prompt, in the below order, only when confident of their values:
                    a. Role: Defining a clear role or identity for the AI to adopt.
                    b. Context: Providing sufficient context to frame the task.
                    c. Objective: Specifying the user's overall objective and main intent.
                    d. Instruction/constraints: Thoroughly, comprehensively and clearly listing/enumerating, emphasizing all usual and unusual constrains, restrictions, conditions and formatting requests along with their details, ensuring they are stated in an order and in a way that the LLM will strictly adhere to them. 
                8. Makes a clear distinction between prompt sections and strictly do not repeat content/instructions across sections.
                9. Ensures the refined prompt is concise, to the point, is not verbose and does not deviate from main objective.
                10. Preserves the input or any particular data required for a correct output.
 
                Fourthly, structure the prompt to ensure improved instruction following. Strictly provide only the refined prompt without any introduction, explanations, additional text or any preamble such as 'Here is the refined prompt:'. The refined prompt should be ready for immediate use without further editing.
                Do NOT include linguistic analysis or technicality level information in the refined prompt.
                """
        # -"Strictly and consistently follow all word count and word frequency instructions. Ensure this class of instructions is consistently and thoroughly followed."
        # -"For the purpose of word counting, do not confuse a word with a token, A word is defined as any sequence of characters (letters, numbers, or symbols) separated by white space, or punctuation"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            temperature=0,
            messages=[{"role": "user", "content": refinement_prompt}],
        )

        return response.content[0].text.strip()

    def _generate_refined_prompt3(
        self,
        original_prompt: str,
        analysis: Dict[str, Any],
        user_inputs: Dict[str, str],
    ) -> str:
        """Generate a refined and enhanced prompt based on the original prompt, analysis, and user inputs"""
        refinement_prompt = f"""
                You are an expert prompt engineer you goal to optimise and enhance the provided manually crafted prompt to achieve improved instruction following.
        Given the original manually crafted prompt: {original_prompt}
        Given the detailed analyses of the original prompt components provided by an expert analyser agent:
        Role/Identity: {analysis.get('Role/Identity', 'N/A')}
        Context: {analysis.get('Context', 'N/A')}
        Objective: {analysis.get('Objective', 'N/A')}
        Prompt/Query: {analysis.get('Prompt/Query', 'N/A')}
        Output_Format: {analysis.get('Output_Format', 'N/A')}
        Examples: {analysis.get('Examples', 'N/A')}
        Tasks: {analysis.get('Tasks', 'N/A')}
        Instructions/Constrains: {analysis.get('Instructions/Constrains', 'N/A')}
        Linguistic_Analysis: {analysis.get('Linguistic_Analysis', 'N/A')}
        Technicality_Level: {analysis.get('Technicality_Level', 'N/A')}

        Firstly, understand the main objective of the prompt, the user intent, the context, and detailed prompt instructions/restriction/constraints.
        
        Secondly, identify any language issues, ambiguities, missing details, or structural issues in the manually crafted prompt.

        Thirdly, your task is to create an improved clear and concise version of the prompt that:
        1. Improves grammar and sentence’s structure of the original prompt.
        2. Enhances clarity and specificity of the instructions.
        3. Addresses any gaps or ambiguities identified in the provided analysis.
        4. Strictly maintains and effectively communicate the original intent of the prompt.
        5. Preserves and emphasizes ALL constraints identified in the original prompt without excessive verbosity nor adding new requirements not present in the original prompt.
        6. Includes each and every instruction/restriction/constraint provided by the analyser agent analysis in the Instructions/Constrains field.
        7. Adheres to the below prompt format and incorporates the below elements in the refined prompt, in the below order, only when confident of their values:
            a. Role: Defining a clear role or identity for the AI to adopt.
            b. Context: Providing sufficient context to frame the task.
            c. Objective: Specifying the user's overall objective and main intent.
            d. Instruction/constraints: Thoroughly, comprehensively and clearly listing/enumerating, emphasizing all usual and unusual constrains, restrictions, conditions and formatting requests along with their details, ensuring they are stated in an order and in a way that the LLM will strictly adhere to them. 
            e. Output Format: Specifying the desired output format, if applicable. 
        8. Makes a clear distinction between prompt sections and strictly do not repeat content/instructions across sections.
        9. Ensures the refined prompt is concise, to the point, is not verbose and does not deviate from main objective.
        10.Strictly Does not add instructions/conditions/restriction/constraint not present in original prompt. Maintains a straightforward and concise prompt. 

        Fourthly, structure the prompt to ensure improved instruction following. Strictly provide only the refined prompt without any introduction, explanations, additional text or any preamble such as 'Here is the refined prompt:'. The refined prompt should be ready for immediate use without further editing.
        Add the following instruction to the refined prompt:
         - "Strictly Provide only the response without any introduction, explanations, additional text or any preamble such aspreamble such as 'Here is the response'"
         -"Strictly and consistently follow all word count and word frequency instructions. Ensure this class of instructions is consistently and thoroughly followed."
        Write the prompt in the second person singular.
        Does NOT include linguistic analysis or technicality level information in the refined prompt.
        """
        # -"Strictly and consistently follow all word count and word frequency instructions. Ensure this class of instructions is consistently and thoroughly followed."
        # -"For the purpose of word counting, do not confuse a word with a token, A word is defined as any sequence of characters (letters, numbers, or symbols) separated by white space, or punctuation"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": refinement_prompt}],
        )

        return response.content[0].text.strip()

    def run_prompt_output(self, prompt: str) -> str:
        """Run a single prompt through the Anthropic model and return the response"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": f"Execute the following prompt. Ensure all instructions are strictly followed.: {prompt}",
                },
            ],
        )
        print("############", response.content[0].text.strip())
        return response.content[0].text.strip()

    def run_prompt_GPT(self, prompt: str) -> str:
        """Run a single prompt through the Anthropic model and return the response"""
        api_key = "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"
        client = OpenAI(api_key=api_key)
        model = "gpt-4o-2024-08-06"  # "meta-llama-3.1-70b-instruct"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            presence_penalty=0,
            temperature=0,
            max_tokens=2000,
        )
        print("############", response.choices[0].message.content)
        return response.choices[0].message.content


# Example usage remains the same
if __name__ == "__main__":
    analyzer = AnalyzerAgent()
    suggester = SuggesterAgent()

    original_prompt = """Act as an expert data scientist. Given a dataset of customer transactions from an e-commerce website, analyze the purchasing patterns and provide insights on customer behavior. Present your findings in a report format with bullet points for key insights and include any relevant statistical measures. For example, you might look at frequency of purchases, average order value, or popular product categories."""
    original_prompt2 = """Act as a data scientist and analyze customer transactions."""
    original_prompt3 = """"Write an email to my boss telling him that I am quitting. The email must contain a title wrapped in double angular brackets, i.e. <<title>>.\nFirst repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)"""
    original_prompt4 = """Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*."""
    original_prompt5 = """Write me a resume for Matthias Algiers. Use words with all capital letters to highlight key abilities, but make sure that words with all capital letters appear less than 10 times. Wrap the entire response with double quotation marks."""
    analysis = analyzer.analyze_prompt(original_prompt4)
    refined = suggester.refine_prompt(original_prompt4, analysis)
    output = suggester.run_prompt_output(refined)
