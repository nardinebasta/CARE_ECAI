# CARE Framework: Automated Prompt Refinement

## Overview
The CARE (Comprehensive Analyzer & REfiner) framework is a multi-agent system designed to enhance LLM performance through automated prompt refinement. This project implements a dual-agent architecture that systematically improves prompts by:

- Analyzing and decomposing the original prompt structure
- Identifying potential ambiguities and improvement opportunities
- Generating refined prompts that maintain semantic fidelity while enhancing clarity
- Evaluating instruction-following performance improvements

The framework addresses common challenges in prompt optimization by implementing a structured transformation pipeline with explicit validation constraints.

## Components

### 1. Analyzer Agent (`AnalyserAgent.py`)
The Analyzer Agent performs systematic prompt decomposition by:
- Analyzing manually crafted prompts
- Extracting key elements such as:
  - Role/Identity (persona the AI is asked to emulate)
  - Context (background information)
  - Objective (user's goal)
  - Prompt/Query (specific question or task)
  - Output Format (specified format for response)
  - Examples (provided in the prompt)
  - Tasks (single or combined)
  - Instructions/Constraints (conditions and restrictions)
  - Linguistic Analysis (readability, tone, etc.)
  - Technicality Level (low/medium/high)
- Detecting explicit and implicit constraints
- Analyzing linguistic patterns and technicality level
- Producing a structured JSON representation of the prompt components for consumption by the Suggester Agent

```python
from AnalyserAgent import AnalyzerAgent

analyzer = AnalyzerAgent()
analysis = analyzer.analyze_prompt("Write a resume for John Doe...")
print(json.dumps(analysis, indent=2))
```

### 2. Refiner Agent (`SuggesterAgent.py`) 
The Refiner agent (implemented as SuggesterAgent) transforms the original prompt into an improved version by:
- Takes the original prompt and analyzer output as inputs
- Identifies language issues, ambiguities, missing details, and structural problems
- Improves grammar and sentence structure
- Enhances clarity and specificity of instructions
- Addresses gaps identified in the analysis
- Maintains the original intent while improving formatting
- Ensures all constraints are preserved and emphasized
- Structures the prompt according to a clear format:
  - Role
  - Context
  - Objective
  - Instructions/constraints
  - Output Format
- Provides a refined prompt ready for immediate use

```python
from SuggesterAgent import SuggesterAgent
from AnalyserAgent import AnalyzerAgent

analyzer = AnalyzerAgent()
suggester = SuggesterAgent()

original_prompt = "Write a resume for John Doe..."
analysis = analyzer.analyze_prompt(original_prompt)
refined_prompt = suggester.refine_prompt(original_prompt, analysis)
print(refined_prompt)
```

### 3. Refined Prompt Generator (`RefinedPromptGenerator.py`)
The Refined Prompt Generator orchestrates the entire refinement process by:
- Loading input prompts from a JSONL file
- For each prompt:
  - Calls the Analyzer Agent to analyze the prompt
  - Passes the analysis to the Suggester Agent for refinement
  - Maintains original formatting (uppercase, lowercase, etc.)
  - Preserves special formatting requirements (e.g., double angular brackets)
- Outputs refined prompts to a new JSONL file

```python
# Example usage
python RefinedPromptGenerator.py
```

### 4. Baseline Data Generators
These modules generate model outputs for both original and refined prompts to enable evaluation of instruction following performance:

#### a. Anthropic Claude (`BaselineDataGenerator.py`)
- Processes prompts using Anthropic's Claude model
- Formats responses based on prompt instructions
- Generates output JSONL files with prompt-response pairs

```python
# Example usage
python BaselineDataGenerator.py
```

#### b. DeepSeek (`BaselineDataGeneratorDeepSeek.py`)
- Processes prompts using DeepSeek model
- Formats responses based on prompt instructions
- Generates output JSONL files with prompt-response pairs

```python
# Example usage
python BaselineDataGeneratorDeepSeek.py
```

#### c. GPT (`BaselineDataGeneratorGPT.py`)
- Processes prompts using OpenAI's GPT-4 model
- Formats responses based on prompt instructions
- Generates output JSONL files with prompt-response pairs

```python
# Example usage
python BaselineDataGeneratorGPT.py
```

#### d. OctoAI/Mixtral (`BaselineDataGeneratorOctoAI.py`)
- Processes prompts using Mixtral 8x7B model via OctoAI
- Formats responses based on prompt instructions
- Generates output JSONL files with prompt-response pairs

```python
# Example usage
python BaselineDataGeneratorOctoAI.py
```

## Usage Workflow

1. Prepare input data: Create a JSONL file with prompts you want to refine
2. Run prompt refinement: Execute `RefinedPromptGenerator.py` to analyze and refine prompts
3. Generate outputs: Use the appropriate baseline data generators to create model outputs
4. Evaluate performance: Compare instruction following between original and refined prompts

## Implementation Details

- The framework uses Claude 3 Sonnet for both analysis and refinement
- JSON is used as the interchange format between components
- Refined prompts maintain the original intent while improving structure
- Special handling is implemented for formatting requirements (e.g., case sensitivity, wrapping)

## Setup and Usage

1. Add your API keys to the appropriate files
2. Prepare input data in JSONL format
3. Run `RefinedPromptGenerator.py` to generate refined prompts
4. Run the baseline data generators to process prompts with different models
5. Compare and evaluate the results

## Repository Structure

```
CARE/
├── AnalyserAgent.py        # Implements the Analyzer agent
├── SuggesterAgent.py       # Implements the Refiner agent
├── RefinedPromptGenerator.py    # Orchestrates the prompt refinement process
├── BaselineDataGenerator.py     # Generates outputs for Anthropic models
├── BaselineDataGeneratorDeepSeek.py    # Generates outputs for DeepSeek models
├── BaselineDataGeneratorGPT.py         # Generates outputs for OpenAI models
├── BaselineDataGeneratorOctoAI.py      # Generates outputs for Mixtral models
└── data/
    ├── input_data.jsonl
    ├── refined_prompt.jsonl
    ├── refined_output_anthropic.jsonl
    ├── refined_output_deepSeek.jsonl
    ├── refined_output_GPT.jsonl
    └── refined_output_Mixtral.jsonl
```

## Dependencies

- anthropic
- openai
- octoai
- json
- typing

## Notes

- Remember to replace "AddYourKey" with your actual API keys
- The framework supports multiple LLM providers for comparative evaluation
- The data directory should be created before running the scripts
