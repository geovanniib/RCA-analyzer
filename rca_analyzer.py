#!/usr/bin/env python3
"""
Cloud RCA Analysis Tool using OpenAI GPT-4
Analyzes Root Cause Analysis reports to identify patterns and improvement opportunities.
Supports both text files and PDF files.
"""

import openai
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# PDF processing imports
try:
    import PyPDF2
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Load environment variables
load_dotenv()

class RCAAnalyzer:
    def __init__(self, api_key=None, model="gpt-4-turbo-preview", prompt_file="prompt.txt"):
        """Initialize the RCA Analyzer with OpenAI API."""
        # Get API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {e}")
        
        self.model = model
        self.prompt_file = prompt_file
        self.analysis_prompt = self._load_analysis_prompt(prompt_file)
        
    def _load_analysis_prompt(self, prompt_file="prompt.txt"):
        """Load the specialized RCA analysis prompt from file."""
        try:
            with open(prompt_file, 'r', encoding='utf-8') as file:
                prompt_content = file.read().strip()
            
            if not prompt_content:
                raise ValueError("Prompt file is empty")
            
            print(f"✅ Loaded analysis prompt from {prompt_file}")
            return prompt_content
            
        except FileNotFoundError:
            print(f"❌ Error: Prompt file '{prompt_file}' not found.")
            print("Please ensure prompt.txt exists in the current directory.")
            raise
        except Exception as e:
            print(f"❌ Error loading prompt: {e}")
            raise

    def load_rca_data(self, file_path="data.txt"):
        """Load RCA data from file (supports .txt, .pdf)."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"RCA data file '{file_path}' not found. Please ensure the file exists.")
            
            # Determine file type and process accordingly
            if file_path.suffix.lower() == '.pdf':
                content = self._extract_text_from_pdf(file_path)
            else:
                # Handle as text file
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
            
            if not content:
                raise ValueError("File appears to be empty or no text could be extracted")
                
            print(f"✅ Loaded RCA data from {file_path} ({len(content)} characters)")
            return content
            
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise Exception(f"Error loading RCA data: {e}")

    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file using multiple methods for best results."""
        if not PDF_SUPPORT:
            raise Exception("PDF support not available. Please install: pip install PyPDF2 pdfplumber")
        
        print(f"📄 Extracting text from PDF: {pdf_path}")
        
        # Try pdfplumber first (better for complex layouts)
        try:
            text_content = self._extract_with_pdfplumber(pdf_path)
            if text_content and len(text_content.strip()) > 100:
                print("✅ Successfully extracted text using pdfplumber")
                return text_content
        except Exception as e:
            print(f"⚠️  pdfplumber extraction failed: {e}")
        
        # Fallback to PyPDF2
        try:
            text_content = self._extract_with_pypdf2(pdf_path)
            if text_content and len(text_content.strip()) > 100:
                print("✅ Successfully extracted text using PyPDF2")
                return text_content
        except Exception as e:
            print(f"⚠️  PyPDF2 extraction failed: {e}")
        
        raise Exception("Failed to extract readable text from PDF. The PDF might be image-based or corrupted.")
    
    def _extract_with_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber (better for tables and complex layouts)."""
        text_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            print(f"📖 Processing {len(pdf.pages)} pages with pdfplumber...")
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Extract text
                    page_text = page.extract_text()
                    
                    if page_text:
                        text_content.append(f"\n--- Page {page_num} ---\n")
                        text_content.append(page_text)
                    
                    # Also try to extract tables if present
                    tables = page.extract_tables()
                    for table in tables:
                        text_content.append(f"\n[TABLE on Page {page_num}]")
                        for row in table:
                            if row:
                                text_content.append(" | ".join([cell or "" for cell in row]))
                
                except Exception as e:
                    print(f"⚠️  Error processing page {page_num}: {e}")
                    continue
        
        return "\n".join(text_content)
    
    def _extract_with_pypdf2(self, pdf_path):
        """Extract text using PyPDF2 (fallback method)."""
        text_content = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"📖 Processing {len(pdf_reader.pages)} pages with PyPDF2...")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"\n--- Page {page_num} ---\n")
                        text_content.append(page_text)
                except Exception as e:
                    print(f"⚠️  Error processing page {page_num}: {e}")
                    continue
        
        return "\n".join(text_content)

    def analyze_rcas(self, rca_data, max_tokens=4000, temperature=0.1):
        """Send RCA data to OpenAI for analysis."""
        try:
            print("🔍 Analyzing RCAs with OpenAI GPT-4...")
            print("⏳ This may take 30-60 seconds...")
            
            # Combine prompt with RCA data
            full_prompt = f"{self.analysis_prompt}\n\nRCA REPORTS TO ANALYZE:\n{rca_data}"
            
            # Check token length (rough estimate)
            estimated_tokens = len(full_prompt.split()) * 1.3
            if estimated_tokens > 16000:  # Leave room for response
                print("⚠️  Warning: Input is quite large. Consider splitting into smaller batches.")
            
            # Make API call with proper error handling
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a senior cloud reliability engineer specializing in incident analysis and pattern recognition."
                        },
                        {
                            "role": "user", 
                            "content": full_prompt
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9
                )
            except openai.RateLimitError:
                raise Exception("Rate limit exceeded. Please wait a moment and try again, or upgrade your OpenAI plan.")
            except openai.AuthenticationError:
                raise Exception("Invalid API key. Please check your OPENAI_API_KEY.")
            except openai.APIError as e:
                raise Exception(f"OpenAI API error: {e}")
            
            analysis = response.choices[0].message.content
            
            # Log usage stats
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                print(f"📊 Token usage: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total")
            
            return analysis
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                raise Exception(f"Rate limit error: {e}. Please wait and try again.")
            elif "api key" in str(e).lower() or "authentication" in str(e).lower():
                raise Exception(f"Authentication error: {e}. Please check your API key.")
            else:
                raise Exception(f"Analysis failed: {e}")

    def save_analysis(self, analysis, output_file=None):
        """Save analysis results to file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"output/rca_analysis_{timestamp}.md"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(f"# RCA Analysis Report\n")
                file.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                file.write(f"**Model:** {self.model}\n\n")
                file.write(analysis)
            
            print(f"💾 Analysis saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")
            return None

    def run_analysis(self, data_file="data.txt", output_file=None, save_results=True):
        """Run complete RCA analysis workflow."""
        try:
            # Load RCA data
            rca_data = self.load_rca_data(data_file)
            
            # Analyze with OpenAI
            analysis = self.analyze_rcas(rca_data)
            
            # Save results
            if save_results:
                output_path = self.save_analysis(analysis, output_file)
            
            # Display results
            print("\n" + "="*80)
            print("📋 RCA ANALYSIS RESULTS")
            print("="*80)
            print(analysis)
            print("="*80)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return None

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Analyze Root Cause Analysis reports using OpenAI GPT-4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rca_analyzer.py                          # Analyze data.txt with defaults
  python rca_analyzer.py --input rcas.pdf        # Analyze PDF file
  python rca_analyzer.py --input rcas.txt        # Analyze text file  
  python rca_analyzer.py --model gpt-4o           # Use GPT-4o model
  python rca_analyzer.py --prompt custom.txt     # Use custom prompt file
  python rca_analyzer.py --no-save               # Don't save output file
        """
    )
    
    parser.add_argument(
        'input', 
        nargs='?',
        default='data.txt',
        help='Path to RCA file (.txt, .pdf) - default: data.txt'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file for analysis results (default: auto-generated)'
    )
    
    parser.add_argument(
        '--model', '-m',
        default='gpt-4-turbo-preview',
        choices=['gpt-4-turbo-preview', 'gpt-4o', 'gpt-4', 'gpt-3.5-turbo'],
        help='OpenAI model to use (default: gpt-4-turbo-preview)'
    )
    
    parser.add_argument(
        '--prompt', '-p',
        default='prompt.txt',
        help='Prompt file to use for analysis (default: prompt.txt)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Don\'t save analysis to file'
    )
    
    parser.add_argument(
        '--api-key',
        help='OpenAI API key (or set OPENAI_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Validate API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OpenAI API key required.")
        print("Set OPENAI_API_KEY environment variable or use --api-key parameter")
        return 1
    
    # Initialize analyzer
    try:
        analyzer = RCAAnalyzer(
            api_key=api_key, 
            model=args.model,
            prompt_file=args.prompt
        )
        print(f"🚀 Starting RCA analysis with {args.model}")
        print(f"📋 Using prompt file: {args.prompt}")
        
        # Run analysis
        result = analyzer.run_analysis(
            data_file=args.input,
            output_file=args.output,
            save_results=not args.no_save
        )
        
        if result:
            print("✅ Analysis completed successfully!")
            return 0
        else:
            print("❌ Analysis failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())