# Academic Paper Summarizer (ScaleDown Powered)

## Overview
This project is a section-aware academic paper summarizer with question-answer capability.

It:
- Extracts text from PDF
- Splits into sections
- Uses ScaleDown API to compress context
- Generates structured summaries
- Supports Q&A mode
- Runs in a web interface using Gradio

## Architecture

PDF → Text Extraction → Section Split → Relevant Section Filtering → 
ScaleDown Compression → Structured Output → Web Interface

## Features

- Multi-level structured summaries
- Question-based answering
- Token optimization (~75–80% reduction)
- Local web app interface

## Setup

1. Clone repository
2. Create virtual environment
3. Install dependencies:
pip install -r requirements.txt

4. Create `.env` file:
SCALEDOWN_API_KEY=your_key_here

5. Run:
python app.py


## Tech Stack

- Python
- PyMuPDF
- ScaleDown API
- GPT-4o (via ScaleDown)
- Gradio