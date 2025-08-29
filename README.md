# RCA-analyzer
This repository contains a Root Cause Analysis (RCA) analyzer that leverages an LLM to identify patterns and suggest infrastructure improvements. By default, it uses prompt.txt as the analysis prompt. If no specific model is provided, it defaults to OpenAI’s GPT-4 (gpt-4-turbo-preview).


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

python rca_analyzer.py data/Incident_RCA_Summary.pdf --all-formats --output-dir ./reports



## Use custom options
Examples:
  python rca_analyzer.py data/Incident_RCA_Summary.pdf                                              # Basic analysis  
  python rca_analyzer.py data/Incident_RCA_Summary.pdf --prompt prompt.txt -d ./output              # custom prompt
  python rca_analyzer.py data/Incident_RCA_Summary.pdf --pdf                                        # Generate PDF report
  python rca_analyzer.py data/Incident_RCA_Summary.pdf --csv                                        # Generate CSV data
  python rca_analyzer.py data/Incident_RCA_Summary.pdf --all-formats                                # Generate all formats
  python rca_analyzer.py data/Incident_RCA_Summary.pdf --pdf --csv                                  # Generate PDF + CSV
  python rca_analyzer.py data/Incident_RCA_Summary.pdf --model gpt-4o --pdf                         # Use GPT-4o with PDF
  python rca_analyzer.py data/Incident_RCA_Summary.pdf --output-dir ./reports                       # Save to specific directory
  python rca_analyzer.py data/Incident_RCA_Summary.pdf --all-formats -d ./output                    # All formats to directory
  python rca_analyzer.py --api-key=your-key-here                                                    # Use api-key directly
  python rca_analyzer.py data/RCA.pdf --prompt prompt.txt --model gpt-4o --output analysis.md       # Combine multiple options
  python rca_analyzer.py data/data.txt --prompt prompt.txt --model gpt-4o --output analysis.md      # TXT input


## Deactivate the environment (optional, when finished)

deactivate