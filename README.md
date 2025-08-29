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

python3 rca_analyzer.py

## Use custom prompt file
python3 rca_analyzer.py --prompt security_focused_prompt.txt

## Full customization
python3 rca_analyzer.py --input my_rcas.txt --prompt custom_prompt.txt --model gpt-4o


## Deactivate the environment (optional, when finished)

deactivate