# RCA-analyzer
This repository contains an RCA analyzer that leverages an LLM to identify patterns and suggest infrastructure improvements. If no specific model is provided, it defaults to OpenAI’s GPT-4.


# Setting Up

This steps are are tested in ubuntu desktop

## Create a virtual environment

python3 -m venv venv

## Activate the virtual environment

source venv/bin/activate

## Install project dependencies

pip install -r requirements.txt 

# Basic usage (uses prompt.txt automatically)

python3 rca_analyzer.py

# Use custom prompt file
python3 rca_analyzer.py --prompt security_focused_prompt.txt

# Full customization
python3 rca_analyzer.py --input my_rcas.txt --prompt custom_prompt.txt --model gpt-4o


# Deactivate the environment (optional, when finished)

deactivate