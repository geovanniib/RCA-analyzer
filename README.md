# RCA-analyzer
This repository contains an analyzer of RCAs that uses an LLM to identify patterns and suggest improvements in the infrastructure.


# Basic usage (uses prompt.txt automatically)


python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt 

python3 rca_analyzer.py

# Use custom prompt file
python3 rca_analyzer.py --prompt security_focused_prompt.txt

# Full customization
python3 rca_analyzer.py --input my_rcas.txt --prompt custom_prompt.txt --model gpt-4o