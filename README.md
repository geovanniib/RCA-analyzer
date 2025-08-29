# RCA-analyzer
This repository contains an RCA analyzer that leverages an LLM to identify patterns and suggest infrastructure improvements. If no specific model is provided, it defaults to OpenAI’s GPT-4.


# What this prompt does (prompt.txt)

This prompt guides an LLM-powered analysis of Root Cause Analysis (RCA) reports from cloud reliability incidents.
It helps identify patterns, systemic issues, and improvement opportunities across multiple incidents, producing a structured report with actionable insights.

The analysis framework covers:

🔎 Root Cause patterns (common issues, classification, recurrence)

📊 Trends (categories, frequency, temporal patterns, impact)

🛠️ Corrective & Preventive Actions (effectiveness, ownership, follow-up)

📈 Systemic Issues (training, communication, bottlenecks)

🚀 Strategic Recommendations (top fixes, investments, early indicators, quick wins)

# Why it’s useful

Provides a standardized and repeatable RCA review across multiple incidents.

Surfaces cross-cutting reliability issues (instead of one-off fixes).

Helps teams move beyond “human error” labels to deeper system insights.

Delivers executive-ready summaries with actionable recommendations.


# HOW TO RUN

## Setting Up

This steps are are tested in ubuntu desktop

### Create a virtual environment

python3 -m venv venv

### Activate the virtual environment

source venv/bin/activate

### Install project dependencies

pip install -r requirements.txt 


## Basic usage (uses prompt.txt automatically)

export OPENAI_API_KEY=***** (your_api_key_here)

python rca_analyzer.py

## Use custom prompt file
python rca_analyzer.py --prompt security_focused_prompt.txt

## Full customization
python rca_analyzer.py --input my_rcas.txt --prompt custom_prompt.txt --model gpt-4o


---
# Analyze different file types
python rca_analyzer.py my_rcas.pdf
python rca_analyzer.py incidents.txt

# Use custom prompt file
python rca_analyzer.py data.pdf --prompt custom_prompt.txt

# Use GPT-4o model
python rca_analyzer.py --model gpt-4o

# Save to specific output file
python rca_analyzer.py data.pdf --output my_analysis.md

# Don't save file, just display
python rca_analyzer.py --no-save

# Provide API key via command line
python rca_analyzer.py --api-key your-key-here

# Combine multiple options
python rca_analyzer.py incidents.pdf --prompt prompt.txt --model gpt-4o --output analysis.md
---


## Deactivate the environment (optional, when finished)

deactivate