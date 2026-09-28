# MediGuide AI

> A Flask-based health information assistant for organizing medical documents, symptoms, medications and other health-related workflows.

**MediGuide AI is a software prototype and information tool. It is not a medical device, does not provide diagnosis or treatment, and does not replace a qualified healthcare professional.** See the [Medical Disclaimer](#medical-disclaimer).

## Overview

MediGuide AI lets a user keep health-related documents and logs in one local workspace and ask an AI model to help explain, summarize and organize them. A single-page web UI talks to a Flask REST backend, which stores data as local JSON files and calls a configurable third-party LLM provider when one is set up.

## Features

- **Document upload and extraction** – PDF, PNG, JPG/JPEG, WEBP, TXT and DOCX (16 MB limit). Text is extracted with `pypdf`, `python-docx`, or OCR (`pytesseract`) for images.
- **Document Q&A, summaries and analysis** – ask questions about an uploaded document, generate structured summaries, and extract structured details.
- **Report comparison** – compare two uploaded reports.
- **Health timeline** – chronological view of uploaded documents and events, with a longitudinal health-summary feature.
- **Symptom tracking** – log symptoms with severity and notes; a keyword-based red-flag check shows an emergency-care message for certain phrases.
- **Medication organization** – keep a personal medication list.
- **Terminology explanation** – plain-language explanations of medical terms.
- **Doctor-visit summary and specialist prep** – compile records into a summary to bring to an appointment.
- **AI provider abstraction** – OpenRouter, OpenAI, Anthropic, Google Gemini, xAI (Grok), DeepSeek, Mistral, Together, Hugging Face, or a custom OpenAI-compatible endpoint, with a primary and a fallback slot.
- **Health knowledge library** – a small static knowledge base (`data/health_knowledge.json`).
- **Healthcare-services workflow (demo only)** – doctor directory, appointments, calendar tasks, prescription uploads, medicine ordering, vitals and analytics. The doctor directory and medicine catalog are **built-in fictional demo data**; no real booking or ordering takes place.
- **Data controls** – export, import and clear stored data from the app.

## Architecture

```
Browser (single-page UI: templates/index.html)
        ↓  REST / JSON
Flask backend (app.py)
        ↓
Storage layer (local JSON files in data/, uploaded files in uploads/)
        ↓
AI provider APIs (optional)  +  static health knowledge (data/health_knowledge.json)
```

## Tech Stack

- Python, Flask
- HTML, CSS, JavaScript (single template, no build step)
- JSON file storage
- `requests`, `pypdf`, `python-docx`, `Pillow`, `pytesseract`, `python-dotenv`
- Third-party LLM provider REST APIs

## Project Structure

```
mediguide-ai/
├── app.py                        # Flask app: routes, storage helpers, provider layer, safety logic
├── templates/
│   └── index.html                # Single-page frontend
├── data/
│   └── health_knowledge.json     # Static, generic health knowledge (tracked)
├── uploads/                      # Runtime uploads (contents git-ignored)
│   └── prescriptions/
├── sample_blood_test_report.txt  # SYNTHETIC demo document
├── sample_prescription.txt       # SYNTHETIC demo document
├── requirements.txt
├── .env.example
└── .gitignore
```

Other files in `data/` (`documents_store.json`, `timeline_store.json`, `symptoms_store.json`, `medications_store.json`, `provider_store.json`, and the healthcare-services stores) are created at run time and are git-ignored.

## Setup

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

Image OCR additionally requires the [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) binary to be installed on your system. Without it, image uploads will not yield extracted text; other features still work.

Optional environment variables: `PORT` (default `5000`) and `FLASK_DEBUG=1` (local development only; debug mode is off by default).

## Configuration

The app starts and runs **without any API key**. AI-powered features then show a "no provider configured" message.

Two ways to configure a provider locally:

1. **In the app:** open *Settings* and add an API key for a provider. The key is saved in plain text to `data/provider_store.json` on your machine (this file is git-ignored).
2. **Environment variables:** copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`. These seed the primary provider on first start if none is configured.

Never commit `.env` or anything under `data/` or `uploads/`.

## Demo Data

`sample_blood_test_report.txt` and `sample_prescription.txt` contain **synthetic, fictional demonstration data** ("Demo Patient", "Dr. Demo Physician", "DEMO-0001"). They do not describe any real person. The built-in demo doctor directory, medicine catalog and the "prototype demo data" timeline records are likewise fictional.

## Security

- No API keys or credentials are included in this repository.
- Provider credentials must be configured locally.
- Runtime health data (`data/*` except the knowledge base) and uploaded medical documents are excluded via `.gitignore`.
- Do not store real health information in a public repository or use real records for demos.
- The app has no user authentication and is intended to run locally on `127.0.0.1` for a single user. Do not expose it to a network without adding authentication, HTTPS and encryption at rest.
- Stored provider keys and health data are **not encrypted** on disk.

## Medical Disclaimer

MediGuide AI is a software prototype for information and organization only. It does not diagnose, treat or prevent any condition, and its output may be incomplete or wrong. Always consult a qualified healthcare professional for medical advice. In an emergency, contact your local emergency services immediately.

## Limitations

- AI features require a third-party provider and API key (usage may incur cost).
- Healthcare-services features (doctors, appointments, medicine orders) are simulated with fictional demo data.
- Red-flag detection is keyword/phrase based and will miss many phrasings.
- Data is stored in local JSON files without encryption, authentication or multi-user support.
- Not clinically validated, certified or approved by any regulator.
- OCR quality depends on the image and on a local Tesseract installation.

## Future Improvements

- Authentication and encryption at rest for stored data
- Automated tests and CI
- Replace JSON files with a proper database
- Split the frontend into components and modules
- Broader, more robust safety/red-flag detection
