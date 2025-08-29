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

# PDF generation and visualization imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pandas as pd
    import seaborn as sns
    import io
    import re
    VISUALIZATION_SUPPORT = True
except ImportError:
    VISUALIZATION_SUPPORT = False

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
            prompt_path = Path(prompt_file)
            if not prompt_path.exists():
                # Create default prompt if file doesn't exist
                default_prompt = self._create_default_prompt()
                with open(prompt_file, 'w', encoding='utf-8') as file:
                    file.write(default_prompt)
                print(f"✅ Created default prompt file: {prompt_file}")
                return default_prompt
            
            with open(prompt_file, 'r', encoding='utf-8') as file:
                prompt_content = file.read().strip()
            
            if not prompt_content:
                raise ValueError("Prompt file is empty")
            
            print(f"✅ Loaded analysis prompt from {prompt_file}")
            return prompt_content
            
        except Exception as e:
            print(f"❌ Error loading prompt: {e}")
            # Fallback to default prompt
            print("🔄 Using default prompt instead")
            return self._create_default_prompt()

    def _create_default_prompt(self):
        """Create a default RCA analysis prompt."""
        return """You are a senior cloud reliability engineer specializing in Root Cause Analysis (RCA) and incident pattern recognition.

Analyze the provided RCA reports and provide a comprehensive analysis including:

1. **Executive Summary**
   - Brief overview of the analysis scope
   - Key findings and insights

2. **Root Cause Classification**
   - Categorize incidents by root cause type (Technical, Process, Human Error, Infrastructure, etc.)
   - Provide percentages and incident counts for each category

3. **Pattern Analysis**
   - Identify recurring patterns across incidents
   - Common failure modes and contributing factors
   - Temporal patterns (time of day, day of week, etc.)

4. **Impact Assessment**
   - Severity distribution
   - Service/system impact analysis
   - Customer impact patterns

5. **Improvement Recommendations**
   - Specific actionable recommendations
   - Prioritized by impact and feasibility
   - Include preventive measures

6. **Systemic Issues**
   - Identify underlying systemic problems
   - Process gaps or technical debt
   - Organizational or cultural factors

Provide specific, actionable insights based on the data. Use clear formatting with headers and bullet points."""

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

    def save_analysis(self, analysis, output_file=None, output_dir=None, generate_pdf=False, generate_csv=False):
        """Save analysis results to file with optional PDF and CSV outputs."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"rca_analysis_{timestamp}.md"
        
        # Handle output directory
        if output_dir:
            output_dir = Path(output_dir)
            # Create directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / Path(output_file).name
            print(f"📁 Output directory: {output_dir}")
        
        output_file = Path(output_file)
        base_name = output_file.stem
        base_dir = output_file.parent
        outputs_created = []
        
        try:
            # Save markdown file
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(f"# RCA Analysis Report\n")
                file.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                file.write(f"**Model:** {self.model}\n\n")
                file.write(analysis)
            
            print(f"💾 Analysis saved to: {output_file}")
            outputs_created.append(str(output_file))
            
            # Generate CSV if requested
            if generate_csv:
                csv_file = base_dir / f"{base_name}_data.csv"
                if self._generate_csv_report(analysis, str(csv_file)):
                    outputs_created.append(str(csv_file))
            
            # Generate PDF if requested
            if generate_pdf:
                pdf_file = base_dir / f"{base_name}_report.pdf"
                if self._generate_pdf_report(analysis, str(pdf_file)):
                    outputs_created.append(str(pdf_file))
            
            return outputs_created
            
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")
            return []

    def _generate_csv_report(self, analysis, csv_file):
        """Generate CSV file with extracted data tables."""
        try:
            if not VISUALIZATION_SUPPORT:
                print("⚠️  CSV generation requires pandas. Install with: pip install pandas")
                return False
            
            print(f"📊 Generating CSV report: {csv_file}")
            
            # Extract Root Cause Classification data
            classification_data = self._extract_classification_data(analysis)
            
            if not classification_data:
                print("⚠️  No classification data found to export to CSV")
                return False
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(classification_data)
            df.to_csv(csv_file, index=False)
            
            print(f"✅ CSV report saved: {csv_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating CSV: {e}")
            return False

    def _extract_classification_data(self, analysis):
        """Extract structured data from the analysis text."""
        data = []
        
        # Extract Root Cause Classification section
        classification_pattern = r"Root Cause Classification:?\s*\n((?:[-*]\s*.*?:\s*\d+%.*?\n?)+)"
        match = re.search(classification_pattern, analysis, re.MULTILINE | re.IGNORECASE)
        
        if match:
            classification_text = match.group(1)
            
            # Parse individual classification items
            item_pattern = r"[-*]\s*(.*?):\s*(\d+)%\s*\((\d+)\s*incidents?\)"
            items = re.findall(item_pattern, classification_text)
            
            for category, percentage, count in items:
                data.append({
                    'Category': category.strip(),
                    'Percentage': int(percentage),
                    'Incident_Count': int(count)
                })
        
        # If no structured data found, try alternative patterns
        if not data:
            # Try to extract from any percentage patterns in the text
            percentage_pattern = r"(Technical|Process|Human|Infrastructure|Equipment).*?(\d+)%.*?(\d+)\s*incidents?"
            matches = re.findall(percentage_pattern, analysis, re.IGNORECASE)
            
            for category, percentage, count in matches:
                data.append({
                    'Category': category.strip(),
                    'Percentage': int(percentage),
                    'Incident_Count': int(count)
                })
        
        return data

    def _generate_pdf_report(self, analysis, pdf_file):
        """Generate comprehensive PDF report with embedded graphs."""
        try:
            if not VISUALIZATION_SUPPORT:
                print("⚠️  PDF generation requires reportlab and matplotlib.")
                print("Install with: pip install reportlab matplotlib pandas seaborn")
                return False
            
            print(f"📄 Generating PDF report: {pdf_file}")
            
            # Create PDF document
            doc = SimpleDocTemplate(pdf_file, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
            story = []
            
            # Get styles and create custom ones
            styles = getSampleStyleSheet()
            
            # Enhanced styles for better hierarchy
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=28,
                spaceAfter=40,
                spaceBefore=20,
                alignment=TA_CENTER,
                textColor=colors.darkblue,
                fontName='Helvetica-Bold'
            )
            
            h1_style = ParagraphStyle(
                'CustomH1',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=16,
                spaceBefore=24,
                textColor=colors.darkred,
                fontName='Helvetica-Bold',
                borderWidth=1,
                borderColor=colors.darkred,
                borderPadding=8
            )
            
            h2_style = ParagraphStyle(
                'CustomH2',
                parent=styles['Heading2'], 
                fontSize=16,
                spaceAfter=12,
                spaceBefore=18,
                textColor=colors.darkblue,
                fontName='Helvetica-Bold'
            )
            
            h3_style = ParagraphStyle(
                'CustomH3',
                parent=styles['Heading3'],
                fontSize=14,
                spaceAfter=10,
                spaceBefore=14,
                textColor=colors.darkgreen,
                fontName='Helvetica-Bold'
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=8,
                spaceBefore=0,
                alignment=TA_JUSTIFY,
                fontName='Helvetica'
            )
            
            bullet_style = ParagraphStyle(
                'CustomBullet',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=6,
                spaceBefore=0,
                leftIndent=20,
                bulletIndent=0,
                fontName='Helvetica'
            )
            
            numbered_style = ParagraphStyle(
                'CustomNumbered',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=6,
                spaceBefore=0,
                leftIndent=20,
                bulletIndent=0,
                fontName='Helvetica'
            )
            
            # Title page
            story.append(Paragraph("RCA Analysis Report", title_style))
            story.append(Spacer(1, 30))
            
            # Report metadata
            metadata = f"""
            <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            <b>Model:</b> {self.model}<br/>
            <b>Analysis Type:</b> Cloud Infrastructure RCA Pattern Analysis<br/>
            """
            story.append(Paragraph(metadata, normal_style))
            story.append(Spacer(1, 30))
            
            # Generate and embed Root Cause Classification chart
            chart_data = self._extract_classification_data(analysis)
            if chart_data:
                chart_image = self._create_classification_chart(chart_data)
                if chart_image:
                    story.append(Paragraph("Root Cause Classification", h1_style))
                    story.append(Spacer(1, 12))
                    story.append(chart_image)
                    story.append(Spacer(1, 20))
                    
                    # Add data table
                    story.append(Paragraph("Classification Data Table", h2_style))
                    table_data = [['Category', 'Percentage', 'Incident Count']]
                    for item in chart_data:
                        table_data.append([
                            item['Category'], 
                            f"{item['Percentage']}%", 
                            str(item['Incident_Count'])
                        ])
                    
                    table = Table(table_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 10)
                    ]))
                    story.append(table)
                    story.append(PageBreak())
            
            # Process the analysis content with improved markdown parsing
            story.extend(self._parse_markdown_content(analysis, h1_style, h2_style, h3_style, normal_style, bullet_style, numbered_style))
            
            # Build PDF
            doc.build(story)
            print(f"✅ PDF report saved: {pdf_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            return False

    def _parse_markdown_content(self, analysis, h1_style, h2_style, h3_style, normal_style, bullet_style, numbered_style):
        """Parse markdown content and convert to PDF elements with proper formatting."""
        story = []
        lines = analysis.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Handle headers (# ## ###)
            if line.startswith('#'):
                header_level = len(line) - len(line.lstrip('#'))
                header_text = line.lstrip('#').strip()
                # Format header text (remove ** and make uppercase)
                header_text = self._format_header_text(header_text)
                
                if header_level == 1:
                    story.append(Spacer(1, 20))
                    story.append(Paragraph(header_text, h1_style))
                elif header_level == 2:
                    story.append(Spacer(1, 16))
                    story.append(Paragraph(header_text, h2_style))
                elif header_level >= 3:
                    story.append(Spacer(1, 12))
                    story.append(Paragraph(header_text, h3_style))
                
                i += 1
                continue
            
            # Handle numbered lists (1. 2. etc.)
            numbered_match = re.match(r'^(\d+)\.\s+(.+)', line)
            if numbered_match:
                number, text = numbered_match.groups()
                # Process multi-line numbered items
                full_text = text
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#', '-', '*')) and not re.match(r'^\d+\.', lines[i].strip()) and not re.match(r'^\*\*', lines[i].strip()):
                    full_text += ' ' + lines[i].strip()
                    i += 1
                
                # Format the text BEFORE adding number
                formatted_text = self._format_text_content(full_text)
                story.append(Paragraph(f"{number}. {formatted_text}", numbered_style))
                continue
            
            # Handle bullet points (- or *)
            bullet_match = re.match(r'^[-*]\s+(.+)', line)
            if bullet_match:
                bullet_text = bullet_match.group(1)
                # Process multi-line bullet items
                full_text = bullet_text
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#', '-', '*')) and not re.match(r'^\d+\.', lines[i].strip()) and not re.match(r'^\*\*', lines[i].strip()):
                    full_text += ' ' + lines[i].strip()
                    i += 1
                
                # Format the text BEFORE adding bullet
                formatted_text = self._format_text_content(full_text)
                story.append(Paragraph(f"• {formatted_text}", bullet_style))
                continue
            
            # Handle standalone lines that start with **bold** (like section labels)
            if re.match(r'^\*\*[^*]+?\*\*', line):
                # This is likely a standalone bold statement
                paragraph_text = line
                i += 1
                # Collect continuation lines but stop at new sections
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#', '-', '*')) and not re.match(r'^\d+\.', lines[i].strip()) and not re.match(r'^\*\*', lines[i].strip()):
                    paragraph_text += ' ' + lines[i].strip()
                    i += 1
                
                formatted_text = self._format_text_content(paragraph_text)
                story.append(Paragraph(formatted_text, normal_style))
                continue
            
            # Handle regular paragraphs
            paragraph_text = line
            i += 1
            # Collect multi-line paragraphs
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#', '-', '*')) and not re.match(r'^\d+\.', lines[i].strip()) and not re.match(r'^\*\*', lines[i].strip()):
                paragraph_text += ' ' + lines[i].strip()
                i += 1
            
            if paragraph_text:
                formatted_text = self._format_text_content(paragraph_text)
                story.append(Paragraph(formatted_text, normal_style))
        
        return story
    
    def _format_header_text(self, text):
        """Format header text by removing ** and making it uppercase if it was bold."""
        if not text:
            return ""
        
        # Check if the entire header was meant to be bold
        if text.startswith('**') and text.endswith('**'):
            # Remove ** and make uppercase
            return text[2:-2].upper()
        
        # Handle partial bold in headers
        def bold_and_upper(match):
            content = match.group(1)
            return content.upper()
        
        text = re.sub(r'\*\*(.*?)\*\*', bold_and_upper, text)
        
        return text
    
    def _format_text_content(self, text):
        """Format text content with bold, italic, and other markdown elements."""
        if not text:
            return ""
        
        # Convert **bold** to HTML with uppercase - more precise pattern
        def bold_and_upper(match):
            content = match.group(1)
            return f'<b>{content.upper()}</b>'
        
        # Use a more precise regex that handles the specific case
        text = re.sub(r'\*\*([^*]+?)\*\*', bold_and_upper, text)
        
        # Convert *italic* to HTML (single asterisks only, not part of **)
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', text)
        
        # Convert `code` to HTML
        text = re.sub(r'`([^`]+)`', r'<font name="Courier"><b>\1</b></font>', text)
        
        # Handle percentages and numbers to make them stand out
        text = re.sub(r'(\d+%)', r'<b>\1</b>', text)
        text = re.sub(r'(\d+\s+incidents?)', r'<b>\1</b>', text)
        
        # Handle section labels that might be in quotes and make them bold+uppercase
        text = re.sub(r'"([^"]+?)":', r'<b>\1</b>:', text, flags=re.IGNORECASE)
        
        # Handle common RCA terms and make them bold+uppercase (but not if already processed)
        rca_terms = [
            r'\b(root cause)\b', r'\b(incidents?)\b', r'\b(outages?)\b', 
            r'\b(failures?)\b', r'\b(errors?)\b', r'\b(alerts?)\b',
            r'\b(monitoring)\b', r'\b(services?)\b', r'\b(systems?)\b',
            r'\b(technical)\b', r'\b(process)\b', r'\b(human error)\b',
            r'\b(infrastructure)\b', r'\b(recommendations?)\b', r'\b(action items?)\b',
            r'\b(deployment)\b', r'\b(configuration)\b', r'\b(permissions?)\b',
            r'\b(validation)\b', r'\b(automated?)\b', r'\b(monitoring)\b'
        ]
        
        for term_pattern in rca_terms:
            # Only apply if not already in bold tags
            def term_formatter(match):
                full_match = match.group(0)
                term = match.group(1)
                return full_match.replace(term, f'<b>{term.upper()}</b>')
            
            # Use negative lookbehind and lookahead to avoid double-processing
            text = re.sub(f'(?<!<b>){term_pattern}(?!</b>)', term_formatter, text, flags=re.IGNORECASE)
        
        # Clean up any remaining markdown artifacts
        text = text.replace('###', '').replace('##', '').replace('#', '')
        
        return text

    def _create_classification_chart(self, data):
        """Create a pie chart for root cause classification."""
        try:
            # Set up the plot with a clean style
            plt.style.use('default')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            # Extract data for plotting
            categories = [item['Category'] for item in data]
            percentages = [item['Percentage'] for item in data]
            counts = [item['Incident_Count'] for item in data]
            
            # Define colors
            colors_palette = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
            
            # Create pie chart
            wedges, texts, autotexts = ax1.pie(
                percentages, 
                labels=categories, 
                autopct='%1.1f%%',
                colors=colors_palette[:len(categories)],
                startangle=90,
                explode=[0.05] * len(categories)  # Slightly separate wedges
            )
            
            ax1.set_title('Root Cause Distribution by Percentage', fontsize=14, fontweight='bold', pad=20)
            
            # Create bar chart for incident counts
            bars = ax2.bar(categories, counts, color=colors_palette[:len(categories)], alpha=0.7)
            ax2.set_title('Root Cause Distribution by Incident Count', fontsize=14, fontweight='bold', pad=20)
            ax2.set_ylabel('Number of Incidents', fontweight='bold')
            ax2.set_xlabel('Root Cause Category', fontweight='bold')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-axis labels if needed
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save to memory buffer
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='PNG', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            # Create ReportLab Image object
            from reportlab.platypus import Image
            chart_image = Image(img_buffer, width=6*inch, height=3*inch)
            
            return chart_image
            
        except Exception as e:
            print(f"❌ Error creating chart: {e}")
            return None

    def run_analysis(self, data_file="data.txt", output_file=None, output_dir=None, save_results=True, generate_pdf=False, generate_csv=False):
        """Run complete RCA analysis workflow."""
        try:
            # Load RCA data
            rca_data = self.load_rca_data(data_file)
            
            # Analyze with OpenAI
            analysis = self.analyze_rcas(rca_data)
            
            # Save results
            if save_results:
                outputs = self.save_analysis(analysis, output_file, output_dir, generate_pdf, generate_csv)
                if outputs:
                    print(f"📁 Generated {len(outputs)} output files:")
                    for output in outputs:
                        print(f"   • {output}")
            
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
  python rca_analyzer.py data.pdf                              # Basic analysis  
  python rca_analyzer.py data.pdf --pdf                        # Generate PDF report
  python rca_analyzer.py data.pdf --csv                        # Generate CSV data
  python rca_analyzer.py data.pdf --all-formats                # Generate all formats
  python rca_analyzer.py data.pdf --pdf --csv                  # Generate PDF + CSV
  python rca_analyzer.py --model gpt-4o --pdf                  # Use GPT-4o with PDF
  python rca_analyzer.py data.pdf --output-dir ./reports       # Save to specific directory
  python rca_analyzer.py data.pdf --all-formats -d ./output    # All formats to directory
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
        '--output-dir', '-d',
        help='Output directory for all generated files (default: current directory)'
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
        '--api-key',
        help='OpenAI API key (can also use OPENAI_API_KEY env var)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Don\'t save analysis to file'
    )
    
    parser.add_argument(
        '--pdf',
        action='store_true',
        help='Generate PDF report with embedded graphs'
    )
    
    parser.add_argument(
        '--csv', 
        action='store_true',
        help='Generate CSV file with classification data'
    )
    
    parser.add_argument(
        '--all-formats',
        action='store_true', 
        help='Generate all output formats (markdown, PDF, CSV)'
    )
    
    args = parser.parse_args()
    
    # Handle --all-formats flag
    if args.all_formats:
        args.pdf = True
        args.csv = True
    
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
            output_dir=args.output_dir,
            save_results=not args.no_save,
            generate_pdf=args.pdf,
            generate_csv=args.csv
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