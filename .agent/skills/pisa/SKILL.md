---
name: pisa
description: Mitigate cross-modal hallucinations in Vision-Language Models (VLMs) and run the PISA-1200 benchmark using TACD (Topology-Aware Contrastive Decoding). Triggers on /pisa command.
---

# PISA & TACD Vision-Language Evaluation Skill

Use this skill when the user triggers the `/pisa` slash command, or asks to evaluate multimodal hallucinations, extract scene layout graphs $G = (O, R)$, or run the PISA-1200 benchmark.

## Repository Location
`/Users/harshyadav/Desktop/untitled folder/tacd-pisa1200`

## Core Workflows

### 1. Test Query with TACD Contrastive Decoding
Run a contrastive comparison between baseline VLM and TACD:
```bash
cd "/Users/harshyadav/Desktop/untitled folder/tacd-pisa1200"
python main.py --image <image_path> --query "<question>" --alpha 0.6 --beta 0.1
```

### 2. Connect with Gemini or ChatGPT API
If the user provides an image or asks to test live with Gemini or OpenAI:
```bash
# With Gemini API
export GEMINI_API_KEY="your-key"
python main.py --provider gemini --image <image_path> --query "<question>"

# With ChatGPT API
export OPENAI_API_KEY="your-key"
python main.py --provider openai --image <image_path> --query "<question>"
```

### 3. Run PISA-1200 Benchmark Evaluation
```bash
cd "/Users/harshyadav/Desktop/untitled folder/tacd-pisa1200"
# Run 120 fast samples or 1200 full samples
python benchmarks/pisa1200_eval.py --samples 1200
```

### 4. Extract Structural Scene Topology Graph G = (O, R)
```bash
python main.py --mode extract --image <image_path>
```
