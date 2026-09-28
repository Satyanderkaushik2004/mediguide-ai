"""
app.py

MediGuide AI - Single-file Flask backend with expanded features:
  - Document Upload, Secure Storage, and OCR/Text Extraction Pipeline
  - Document AI Q&A, Structured Document Summaries, and Report Comparison
  - Personal Health Timeline, Symptom Tracker, and Medication Organizer
  - Medical Terminology Explainer and Doctor Visit Summary Builder
  - Multi-Provider AI Adapter System & Keyword Safety Layer
"""

import json
import os
import re
import sys
import time
import uuid
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List

import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Directories & Storage File Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

PROVIDER_STORE_PATH = os.path.join(DATA_DIR, "provider_store.json")
DOCUMENTS_STORE_PATH = os.path.join(DATA_DIR, "documents_store.json")
TIMELINE_STORE_PATH = os.path.join(DATA_DIR, "timeline_store.json")
SYMPTOMS_STORE_PATH = os.path.join(DATA_DIR, "symptoms_store.json")
MEDICATIONS_STORE_PATH = os.path.join(DATA_DIR, "medications_store.json")

# Healthcare Services stores
APPOINTMENTS_STORE_PATH = os.path.join(DATA_DIR, "appointments_store.json")
CALENDAR_TASKS_STORE_PATH = os.path.join(DATA_DIR, "calendar_tasks_store.json")
PRESCRIPTIONS_STORE_PATH = os.path.join(DATA_DIR, "prescriptions_store.json")
MEDICINE_ORDERS_STORE_PATH = os.path.join(DATA_DIR, "medicine_orders_store.json")
VITALS_STORE_PATH = os.path.join(DATA_DIR, "vitals_store.json")
HEALTH_PREFS_STORE_PATH = os.path.join(DATA_DIR, "health_prefs_store.json")
PRESCRIPTION_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "prescriptions")
os.makedirs(PRESCRIPTION_UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Storage Helpers
# ---------------------------------------------------------------------------

def _read_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _write_json_file(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Provider store
DEFAULT_PROVIDER_STORE = {"primary": None, "fallback": None}

def load_store() -> Dict[str, Any]:
    return _read_json_file(PROVIDER_STORE_PATH, DEFAULT_PROVIDER_STORE)

def save_store(store: Dict[str, Any]) -> None:
    _write_json_file(PROVIDER_STORE_PATH, store)

def get_slot(slot: str) -> Optional[Dict[str, Any]]:
    return load_store().get(slot)

def set_slot(slot: str, config: Optional[Dict[str, Any]]) -> None:
    store = load_store()
    store[slot] = config
    save_store(store)

def clear_all() -> None:
    save_store(dict(DEFAULT_PROVIDER_STORE))
    _write_json_file(DOCUMENTS_STORE_PATH, [])
    _write_json_file(TIMELINE_STORE_PATH, [])
    _write_json_file(SYMPTOMS_STORE_PATH, [])
    _write_json_file(MEDICATIONS_STORE_PATH, [])


# Documents store
def load_documents() -> List[Dict[str, Any]]:
    return _read_json_file(DOCUMENTS_STORE_PATH, [])

def save_documents(docs: List[Dict[str, Any]]) -> None:
    _write_json_file(DOCUMENTS_STORE_PATH, docs)


# Timeline store
def load_timeline() -> List[Dict[str, Any]]:
    return _read_json_file(TIMELINE_STORE_PATH, [])

def save_timeline(entries: List[Dict[str, Any]]) -> None:
    _write_json_file(TIMELINE_STORE_PATH, entries)


# Symptoms log store
def load_symptoms_log() -> List[Dict[str, Any]]:
    return _read_json_file(SYMPTOMS_STORE_PATH, [])

def save_symptoms_log(logs: List[Dict[str, Any]]) -> None:
    _write_json_file(SYMPTOMS_STORE_PATH, logs)


# Medications store
def load_medications() -> List[Dict[str, Any]]:
    return _read_json_file(MEDICATIONS_STORE_PATH, [])

def save_medications(meds: List[Dict[str, Any]]) -> None:
    _write_json_file(MEDICATIONS_STORE_PATH, meds)


# Appointments store
def load_appointments() -> List[Dict[str, Any]]:
    return _read_json_file(APPOINTMENTS_STORE_PATH, [])

def save_appointments(items: List[Dict[str, Any]]) -> None:
    _write_json_file(APPOINTMENTS_STORE_PATH, items)


# Calendar tasks store (lab tests, vaccinations, follow-ups, general health tasks)
def load_calendar_tasks() -> List[Dict[str, Any]]:
    return _read_json_file(CALENDAR_TASKS_STORE_PATH, [])

def save_calendar_tasks(items: List[Dict[str, Any]]) -> None:
    _write_json_file(CALENDAR_TASKS_STORE_PATH, items)


# Prescriptions store
def load_prescriptions() -> List[Dict[str, Any]]:
    return _read_json_file(PRESCRIPTIONS_STORE_PATH, [])

def save_prescriptions(items: List[Dict[str, Any]]) -> None:
    _write_json_file(PRESCRIPTIONS_STORE_PATH, items)


# Medicine orders store
def load_medicine_orders() -> List[Dict[str, Any]]:
    return _read_json_file(MEDICINE_ORDERS_STORE_PATH, [])

def save_medicine_orders(items: List[Dict[str, Any]]) -> None:
    _write_json_file(MEDICINE_ORDERS_STORE_PATH, items)


# Vitals store (user-entered measurements only — never fabricated)
def load_vitals() -> List[Dict[str, Any]]:
    return _read_json_file(VITALS_STORE_PATH, [])

def save_vitals(items: List[Dict[str, Any]]) -> None:
    _write_json_file(VITALS_STORE_PATH, items)


# Health service preferences
DEFAULT_HEALTH_PREFS = {
    "default_location": "",
    "preferred_consultation_type": "video",
    "preferred_specialties": [],
    "reminder_defaults": ["one_day_before"],
    "reminders_enabled": True,
    "notifications_enabled": True,
    "default_pharmacy": "",
    "document_sharing_consent": False,
}

def load_health_prefs() -> Dict[str, Any]:
    prefs = dict(DEFAULT_HEALTH_PREFS)
    prefs.update(_read_json_file(HEALTH_PREFS_STORE_PATH, {}))
    return prefs

def save_health_prefs(prefs: Dict[str, Any]) -> None:
    _write_json_file(HEALTH_PREFS_STORE_PATH, prefs)


def add_timeline_event(event_type: str, title: str, description: str = "", related_id: str = None) -> None:
    """Internal helper — used by healthcare features to auto-log timeline events."""
    timeline = load_timeline()
    timeline.insert(0, {
        "id": "tl_" + uuid.uuid4().hex[:10],
        "date": datetime.now().isoformat(),
        "type": event_type,
        "title": title,
        "description": description,
        "related_doc_id": related_id,
    })
    save_timeline(timeline)


# ---------------------------------------------------------------------------
# Healthcare Services: Demo Data (clearly labeled sample/demo data — no real
# provider is connected. This architecture is provider-adapter ready for a
# future real doctor/pharmacy integration; see get_doctor_provider() /
# get_pharmacy_provider() below.)
# ---------------------------------------------------------------------------

DEMO_SPECIALTIES = [
    "General Physician", "Cardiologist", "Dermatologist", "Orthopedic Specialist",
    "Gynecologist", "Pediatrician", "Dentist", "Eye Specialist", "ENT Specialist",
    "Neurologist", "Psychiatrist", "Diabetologist", "Gastroenterologist", "Other Specialist",
]

DEMO_DOCTORS = [
    {
        "id": "doc_001", "name": "Dr. Anjali Mehra", "specialty": "General Physician",
        "qualifications": "MBBS, MD (Internal Medicine)", "experience_years": 12,
        "clinic": "Sunrise Family Clinic", "location": "Demo City, Demo State", "distance_km": 1.8,
        "fee": 400, "rating": 4.7, "review_count": 128, "online": True, "in_person": True,
        "languages": ["English", "Hindi", "Punjabi"],
        "about": "Focuses on preventive care, chronic condition management, and general wellness for the whole family.",
        "services": ["General consultation", "Health checkups", "Chronic disease management"],
        "avatar": "🩺",
    },
    {
        "id": "doc_002", "name": "Dr. Rohan Kapoor", "specialty": "Cardiologist",
        "qualifications": "MBBS, DM (Cardiology)", "experience_years": 15,
        "clinic": "Heartcare Specialty Centre", "location": "Demo City, Demo State", "distance_km": 4.2,
        "fee": 900, "rating": 4.8, "review_count": 96, "online": True, "in_person": True,
        "languages": ["English", "Hindi"],
        "about": "Specializes in hypertension, arrhythmia, and preventive cardiology.",
        "services": ["ECG review", "Cardiac risk assessment", "Follow-up consultations"],
        "avatar": "❤️",
    },
    {
        "id": "doc_003", "name": "Dr. Simran Kaur", "specialty": "Dermatologist",
        "qualifications": "MBBS, MD (Dermatology)", "experience_years": 8,
        "clinic": "Glow Skin & Hair Clinic", "location": "Nearby Town, Demo State", "distance_km": 22.5,
        "fee": 600, "rating": 4.6, "review_count": 74, "online": True, "in_person": True,
        "languages": ["English", "Hindi", "Punjabi"],
        "about": "Treats acne, skin allergies, and hair-fall concerns with an evidence-based approach.",
        "services": ["Skin consultation", "Acne treatment", "Hair-fall evaluation"],
        "avatar": "🧴",
    },
    {
        "id": "doc_004", "name": "Dr. Vikram Chaudhary", "specialty": "Orthopedic Specialist",
        "qualifications": "MBBS, MS (Orthopedics)", "experience_years": 18,
        "clinic": "Bone & Joint Institute", "location": "Demo City, Demo State", "distance_km": 3.1,
        "fee": 700, "rating": 4.5, "review_count": 152, "online": False, "in_person": True,
        "languages": ["English", "Hindi"],
        "about": "Manages joint pain, sports injuries, and post-fracture rehabilitation.",
        "services": ["Joint pain evaluation", "Fracture follow-up", "Physiotherapy referral"],
        "avatar": "🦴",
    },
    {
        "id": "doc_005", "name": "Dr. Neha Sharma", "specialty": "Gynecologist",
        "qualifications": "MBBS, MS (Obstetrics & Gynecology)", "experience_years": 10,
        "clinic": "Women's Wellness Clinic", "location": "Demo City, Demo State", "distance_km": 2.4,
        "fee": 650, "rating": 4.9, "review_count": 201, "online": True, "in_person": True,
        "languages": ["English", "Hindi", "Punjabi"],
        "about": "Provides comprehensive women's health, prenatal, and reproductive health care.",
        "services": ["Prenatal checkups", "General gynecology", "Family planning consultation"],
        "avatar": "👩‍⚕️",
    },
    {
        "id": "doc_006", "name": "Dr. Aarav Singh", "specialty": "Pediatrician",
        "qualifications": "MBBS, MD (Pediatrics)", "experience_years": 9,
        "clinic": "Little Stars Children's Clinic", "location": "Demo City, Demo State", "distance_km": 2.9,
        "fee": 450, "rating": 4.7, "review_count": 183, "online": True, "in_person": True,
        "languages": ["English", "Hindi", "Punjabi"],
        "about": "Cares for infants, children, and adolescents, including growth and vaccination tracking.",
        "services": ["Well-child visits", "Vaccination guidance", "Growth monitoring"],
        "avatar": "🧸",
    },
    {
        "id": "doc_007", "name": "Dr. Priya Malhotra", "specialty": "Psychiatrist",
        "qualifications": "MBBS, MD (Psychiatry)", "experience_years": 11,
        "clinic": "Mindful Wellness Centre", "location": "Demo City, Demo State", "distance_km": 5.6,
        "fee": 800, "rating": 4.8, "review_count": 67, "online": True, "in_person": False,
        "languages": ["English", "Hindi"],
        "about": "Supports patients with anxiety, mood concerns, and stress-related conditions.",
        "services": ["Mental health consultation", "Stress management", "Follow-up therapy sessions"],
        "avatar": "🧠",
    },
    {
        "id": "doc_008", "name": "Dr. Karan Bhatia", "specialty": "Diabetologist",
        "qualifications": "MBBS, MD (Endocrinology)", "experience_years": 14,
        "clinic": "MetaHealth Diabetes Centre", "location": "Demo City, Demo State", "distance_km": 3.7,
        "fee": 550, "rating": 4.6, "review_count": 110, "online": True, "in_person": True,
        "languages": ["English", "Hindi"],
        "about": "Focuses on diabetes management, thyroid disorders, and metabolic health.",
        "services": ["Diabetes management", "Thyroid consultation", "Diet & lifestyle planning"],
        "avatar": "💉",
    },
]

DEMO_MEDICINES = [
    {"id": "med_p_001", "name": "Paracetamol", "strength": "500mg", "form": "Tablet", "pack_size": "15 tablets", "prescription_required": False, "price": 32, "available": True, "category": "Pain Relief"},
    {"id": "med_p_002", "name": "Cetirizine", "strength": "10mg", "form": "Tablet", "pack_size": "10 tablets", "prescription_required": False, "price": 28, "available": True, "category": "Allergy"},
    {"id": "med_p_003", "name": "Amoxicillin", "strength": "500mg", "form": "Capsule", "pack_size": "10 capsules", "prescription_required": True, "price": 85, "available": True, "category": "Antibiotic"},
    {"id": "med_p_004", "name": "Metformin", "strength": "500mg", "form": "Tablet", "pack_size": "30 tablets", "prescription_required": True, "price": 60, "available": True, "category": "Diabetes"},
    {"id": "med_p_005", "name": "Amlodipine", "strength": "5mg", "form": "Tablet", "pack_size": "30 tablets", "prescription_required": True, "price": 75, "available": True, "category": "Blood Pressure"},
    {"id": "med_p_006", "name": "ORS Solution", "strength": "21.8g/L", "form": "Sachet", "pack_size": "5 sachets", "prescription_required": False, "price": 45, "available": True, "category": "Hydration"},
    {"id": "med_p_007", "name": "Cough Syrup", "strength": "100ml", "form": "Syrup", "pack_size": "1 bottle", "prescription_required": False, "price": 95, "available": True, "category": "Cough & Cold"},
    {"id": "med_p_008", "name": "Vitamin D3", "strength": "60000 IU", "form": "Sachet", "pack_size": "4 sachets", "prescription_required": False, "price": 110, "available": True, "category": "Supplements"},
    {"id": "med_p_009", "name": "Omeprazole", "strength": "20mg", "form": "Capsule", "pack_size": "10 capsules", "prescription_required": True, "price": 55, "available": False, "category": "Gastric Care"},
    {"id": "med_p_010", "name": "Antiseptic Cream", "strength": "20g", "form": "Cream", "pack_size": "1 tube", "prescription_required": False, "price": 65, "available": True, "category": "First Aid"},
]


# ---------------------------------------------------------------------------
# Longitudinal Health Intelligence: Taxonomy (specialties, topics, keyword maps)
# ---------------------------------------------------------------------------

HEALTH_TOPICS = [
    "Heart / Cardiac Health", "Blood Pressure", "Cholesterol", "Diabetes", "Thyroid",
    "Kidney", "Liver", "Respiratory", "Bone & Joint", "Skin", "Digestive Health",
    "Mental Health", "Eye Health", "ENT", "Reproductive Health", "General Wellness",
    "Infection", "Vaccination", "Nutrition",
]

DOCUMENT_TYPE_LABELS = {
    "lab_report": "Lab Report", "prescription": "Prescription", "scan": "Imaging / Scan",
    "doctor_notes": "Doctor Notes", "discharge_summary": "Discharge Summary", "other": "Medical Document",
}

# Keyword -> Specialty relevance map. Used for confidence-scored, non-fabricated
# tagging when no AI provider is connected, and to cross-check AI output.
SPECIALTY_KEYWORDS = {
    "Cardiology": ["cardiac", "cardio", "heart", "ecg", "ekg", "echocardiogram", "echo report",
                   "lipid", "cholesterol", "ldl", "hdl", "triglyceride", "coronary", "angio",
                   "hypertension", "blood pressure", "bp reading", "pulse", "arrhythmia", "chest pain"],
    "Endocrinology": ["thyroid", "tsh", "t3", "t4", "diabet", "hba1c", "glucose", "insulin",
                       "endocrin", "blood sugar"],
    "Dermatology": ["skin", "derma", "rash", "acne", "eczema", "psoriasis"],
    "Orthopedics": ["bone", "joint", "fracture", "orthop", "spine", "knee", "arthritis", "x-ray of"],
    "Neurology": ["neuro", "brain", "seizure", "migraine", "nerve", "mri brain", "headache"],
    "Gastroenterology": ["liver", "gastro", "stomach", "abdomen", "digestive", "colonoscopy",
                          "endoscopy", "hepat"],
    "ENT": ["ent ", "ear", "nose", "throat", "sinus", "tonsil"],
    "Ophthalmology": ["eye", "vision", "retina", "ophthalm", "cataract"],
    "Gynecology": ["gynec", "obstetric", "pregnan", "menstrual", "uterus", "ovarian"],
    "Pediatrics": ["pediatric", "paediatric", "child", "infant", "vaccination", "immunization"],
    "Psychiatry": ["psychiat", "mental health", "anxiety", "depression", "counsel"],
    "Nephrology": ["kidney", "renal", "creatinine", "nephro", "dialysis"],
    "Pulmonology": ["lung", "respiratory", "asthma", "pulmon", "spirometry", "chest x-ray"],
    "Dentistry": ["dental", "tooth", "teeth", "gum", "oral"],
    "General Medicine": ["general physician", "general checkup", "fever", "cbc", "complete blood count"],
}

TOPIC_KEYWORDS = {
    "Heart / Cardiac Health": ["cardiac", "heart", "ecg", "ekg", "echocardiogram", "coronary", "arrhythmia"],
    "Blood Pressure": ["blood pressure", "hypertension", "bp reading", "systolic", "diastolic"],
    "Cholesterol": ["cholesterol", "ldl", "hdl", "lipid", "triglyceride"],
    "Diabetes": ["diabet", "glucose", "hba1c", "insulin", "blood sugar"],
    "Thyroid": ["thyroid", "tsh", "t3", "t4"],
    "Kidney": ["kidney", "renal", "creatinine", "nephro"],
    "Liver": ["liver", "hepat", "sgot", "sgpt", "bilirubin"],
    "Respiratory": ["lung", "respiratory", "asthma", "cough", "breathless"],
    "Bone & Joint": ["bone", "joint", "fracture", "arthritis", "spine"],
    "Skin": ["skin", "rash", "acne", "eczema"],
    "Digestive Health": ["stomach", "digestive", "abdomen", "gastro", "colonoscopy"],
    "Mental Health": ["anxiety", "depression", "mental health", "stress", "counsel"],
    "Eye Health": ["eye", "vision", "retina", "cataract"],
    "ENT": ["ear", "nose", "throat", "sinus", "tonsil"],
    "Reproductive Health": ["pregnan", "menstrual", "gynec", "obstetric"],
    "Infection": ["infection", "fever", "bacterial", "viral"],
    "Vaccination": ["vaccin", "immuniz"],
    "Nutrition": ["vitamin", "deficiency", "nutrition", "supplement"],
}

DOC_DATE_PATTERNS = [
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b"),
    re.compile(r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b", re.IGNORECASE),
]


def _hash_text(text: str) -> str:
    import hashlib
    return hashlib.sha256((text or "").strip().lower().encode("utf-8", errors="ignore")).hexdigest()


def _keyword_matches(text: str, keyword_map: Dict[str, List[str]]) -> List[str]:
    if not text:
        return []
    lowered = text.lower()
    hits = []
    for label, keywords in keyword_map.items():
        if any(kw in lowered for kw in keywords):
            hits.append(label)
    return hits


def _guess_document_date(text: str, fallback_iso: str) -> Dict[str, str]:
    if text:
        for pattern in DOC_DATE_PATTERNS:
            m = pattern.search(text)
            if m:
                return {"value": m.group(1), "confidence": "medium"}
    return {"value": fallback_iso[:10] if fallback_iso else "Unknown", "confidence": "low"}


def build_keyword_intelligence(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Non-AI fallback extraction. Never fabricates — only flags what keyword
    matching in the actual extracted text/filename supports. Used when no AI
    provider is connected, or as a cross-check floor under AI extraction."""
    text = (doc.get("extracted_text") or "")
    haystack = f"{doc.get('display_name', '')} {doc.get('original_filename', '')} {text}"
    specialties = _keyword_matches(haystack, SPECIALTY_KEYWORDS)
    topics = _keyword_matches(haystack, TOPIC_KEYWORDS)
    date_guess = _guess_document_date(text, doc.get("uploaded_at", ""))
    return {
        "document_date": date_guess["value"],
        "document_type": DOCUMENT_TYPE_LABELS.get(doc.get("category"), "Medical Document"),
        "hospital_or_clinic": "Not identified",
        "doctor_name": "Not identified",
        "doctor_specialty": specialties[0] if specialties else "Not identified",
        "tests_performed": [],
        "diagnoses_mentioned": [],
        "symptoms_mentioned": [],
        "medications_mentioned": [],
        "lab_values": [],
        "imaging_findings": [],
        "procedures": [],
        "follow_up_recommendation": "Not identified",
        "specialties": specialties or ["General Medicine"],
        "topics": topics,
        "confidence": {
            "document_date": date_guess["confidence"],
            "specialty": "medium" if specialties else "low",
            "doctor": "low",
        },
        "extraction_method": "keyword",
    }


def get_doctor_provider():
    """Adapter hook for a future real doctor-search/booking provider.
    Returns None while no external provider is configured (demo mode)."""
    return None


def get_pharmacy_provider():
    """Adapter hook for a future real pharmacy/medicine-ordering provider.
    Returns None while no external provider is configured (demo mode)."""
    return None


def _generate_ics(appointment: Dict[str, Any]) -> str:
    """Build a minimal RFC5545 .ics file for an appointment."""
    def _fmt(dt_str, time_str):
        try:
            dt = datetime.strptime(f"{dt_str} {time_str or '09:00'}", "%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            dt = datetime.now()
        return dt.strftime("%Y%m%dT%H%M%S")

    start = _fmt(appointment.get("date", ""), appointment.get("time", ""))
    location = appointment.get("clinic") or ("Online Consultation" if appointment.get("consultation_type") != "in_person" else "")
    desc = (appointment.get("notes") or "").replace("\n", "\\n")
    uid = appointment.get("id", uuid.uuid4().hex) + "@mediguide-ai.demo"

    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//MediGuide AI//Health Calendar//EN",
        "BEGIN:VEVENT", f"UID:{uid}", f"DTSTAMP:{datetime.now().strftime('%Y%m%dT%H%M%S')}",
        f"DTSTART:{start}", f"SUMMARY:Appointment with {appointment.get('doctor_name', 'Doctor')}",
        f"DESCRIPTION:{appointment.get('specialty', '')} consultation. {desc}",
        f"LOCATION:{location}", "STATUS:TENTATIVE", "END:VEVENT", "END:VCALENDAR",
    ]
    return "\r\n".join(lines)


# ---------------------------------------------------------------------------
# Safety Subsystem
# ---------------------------------------------------------------------------

RED_FLAG_PATTERNS = [
    (r"\b(can'?t breathe|cannot breathe|difficulty breathing|severe (shortness of )?breath|saans (lene mein )?(dikkat|takleef)|dum ghut)", "Severe difficulty breathing"),
    (r"\b(severe chest pain|chest pain and pressure|crushing chest pain|seene mein (bahut )?dard|seene mein jakadan)", "Severe chest pain or pressure"),
    (r"\b(one side.*(weak|numb)|face drooping|sudden weakness|ek taraf (kamzori|sunn))", "Sudden one-sided weakness"),
    (r"\b(sudden confusion|can'?t think straight|disoriented suddenly|achanak confusion|achanak behosh)", "Sudden confusion"),
    (r"\b(loss of consciousness|passed out|fainted|unconscious|behosh ho gaya|behosh ho gayi)", "Loss of consciousness"),
    (r"\b(severe bleeding|won'?t stop bleeding|heavy bleeding|bahut khoon beh raha)", "Severe bleeding"),
    (r"\b(seizure|convulsion|fits\b|daura pada)", "Seizure"),
    (r"\b(anaphylaxis|severe allergic reaction|throat (is )?closing|swelling of (the )?throat|face swelling and (can'?t breathe|difficulty breathing))", "Signs of severe allergic reaction"),
    (r"\b(stabbed|gunshot|severe injury|major accident|deep wound|bahut badi chot)", "Serious injury"),
    (r"\b(suicid|kill myself|end my life|don'?t want to live|khudkushi|marna chahta|marna chahti)", "Suicidal thoughts"),
    (r"\b(in immediate danger|someone is hurting me|emergency right now)", "Immediate danger"),
]

COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), label) for p, label in RED_FLAG_PATTERNS]


def detect_red_flags(text: str) -> List[str]:
    if not text:
        return []
    found = []
    for pattern, label in COMPILED_PATTERNS:
        if pattern.search(text):
            found.append(label)
    return found


EMERGENCY_MESSAGE = (
    "This may require immediate medical attention.\n\n"
    "If you are experiencing severe difficulty breathing, severe chest pain, "
    "loss of consciousness, severe bleeding, or another serious emergency, "
    "contact your local emergency service or go to the nearest emergency "
    "department right away. Do not rely on this chatbot for emergency care.\n\n"
    "If you are having thoughts of suicide or self-harm, please reach out to "
    "a crisis line or emergency service in your area right now, or go to "
    "your nearest emergency department. You deserve immediate support from "
    "a real person."
)


# ---------------------------------------------------------------------------
# Text Extraction Pipeline (PDF, DOCX, TXT, OCR Images)
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "webp", "txt", "docx"}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(filepath: str, ext: str) -> tuple[str, str]:
    """
    Returns (extracted_text, extraction_status)
    extraction_status: 'completed', 'unavailable_ocr', or 'failed'
    """
    ext = ext.lower()
    text = ""

    if ext == "txt":
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            return text, "completed"
        except Exception:
            return "", "failed"

    if ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            pages_text = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
            text = "\n\n".join(pages_text).strip()
            if text:
                return text, "completed"
        except Exception:
            pass

        # Fallback pdf regex string extractor if pypdf extracts nothing
        try:
            with open(filepath, "rb") as f:
                content = f.read().decode("latin1", errors="ignore")
                matches = re.findall(r"\((.*?)\)\s*TJ", content)
                if matches:
                    text = " ".join(matches).strip()
                    if text:
                        return text, "completed"
        except Exception:
            pass

        return "", "unavailable_ocr"

    if ext == "docx":
        try:
            import docx
            doc = docx.Document(filepath)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs).strip()
            return text, "completed" if text else "completed"
        except Exception:
            return "", "failed"

    if ext in ("png", "jpg", "jpeg", "webp"):
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(filepath)
            text = pytesseract.image_to_string(img).strip()
            if text:
                return text, "completed"
        except Exception:
            pass
        return "", "unavailable_ocr"

    return "", "unavailable_ocr"


# ---------------------------------------------------------------------------
# System Prompts for AI Features
# ---------------------------------------------------------------------------

RESPONSE_JSON_SHAPE = """{
  "intent": "symptom_assessment",
  "language": "English | Hindi | Hinglish",
  "reply": "the conversational message to show the user",
  "symptoms": [
    {"name": "string", "duration": "string or empty", "severity": "mild|moderate|severe|unknown"}
  ],
  "follow_up_questions": ["string", "..."],
  "possible_causes": [
    {"name": "string", "explanation": "string", "likelihood_label": "Possible"}
  ],
  "urgency_level": "self_care | routine | consult_soon | urgent | emergency | unknown",
  "recommendation": "string",
  "red_flags": ["string", "..."],
  "assessment_complete": false,
  "disclaimer": "This is general information, not a medical diagnosis."
}"""

SYSTEM_PROMPT = f"""You are MediGuide AI, a careful health symptom assessment and triage
assistant embedded in a student/college project web application. You are
NOT a doctor and you do not provide medical diagnoses.

## Your role
- Help the user describe and understand their symptoms.
- Ask relevant, specific follow-up questions based on what they've already told you.
- Keep track of everything said earlier in the conversation.
- After you have enough information, provide a small number of possible general explanations, a general urgency level, and safe next-step guidance.
- Support English, Hindi, and Hinglish. Detect and respond in whichever language/style the user is using.

## Strict safety rules — never break these
1. Never claim to provide a diagnosis.
2. Never prescribe medications, dosages, or drug timings.
3. Never tell the user to stop or change a prescribed medication.
4. Always escalate potential emergencies ("urgency_level": "emergency").
5. Always include a short disclaimer.

## Output format
Respond with ONLY a single valid JSON object matching exactly this shape:
{RESPONSE_JSON_SHAPE}
"""

DOCUMENT_SUMMARY_PROMPT = """You are MediGuide AI's document analysis module. Analyze the extracted text from a patient's medical document and provide a structured summary.

STRICT SAFETY RULES:
- Do NOT provide a definitive medical diagnosis.
- Do NOT prescribe medications or alter dosage instructions.
- Do NOT label abnormal values as confirmed diseases. State clearly that reference ranges vary by lab, age, and context.
- For scans or images with no text, state that an image must be evaluated by a certified radiologist.

Return ONLY a single valid JSON object matching this structure:
{
  "document_type": "Lab Report | Prescription | Doctor Note | Discharge Summary | Scan/Imaging | General Document",
  "report_date": "Extracted date or Unknown",
  "main_findings": ["Finding 1", "Finding 2"],
  "lab_values": [
    {
      "test_name": "Test Name",
      "value": "Reported Value",
      "reference_range": "Stated Range or Unknown",
      "is_out_of_range": true|false,
      "note": "Plain language explanation"
    }
  ],
  "prescription_medications": [
    {
      "medicine": "Medicine Name",
      "strength": "Strength if written",
      "instructions": "Exact written instructions"
    }
  ],
  "medical_terms_explained": [
    {
      "term": "Medical Term",
      "meaning": "Simple plain language explanation"
    }
  ],
  "questions_for_doctor": ["Question 1", "Question 2"],
  "disclaimer": "This summary is generated by MediGuide AI for educational and organizational purposes only. It is not a medical diagnosis."
}
"""

DOCUMENT_QA_PROMPT = """You are MediGuide AI, an assistant helping a patient understand their medical document.

Rules:
- Answer the user's question using ONLY the provided document text as context.
- Explain medical terms simply.
- Never diagnose the user or declare medical certainty.
- Never prescribe medication or suggest changing prescribed dosages.
- If extracted text is incomplete, clearly state that the answer may be partial.
- Always remind the user to review the document with their doctor.
"""

REPORT_COMPARISON_PROMPT = """You are MediGuide AI's health analytics assistant. Compare Report 1 (older) with Report 2 (newer) using their extracted text.

Rules:
- Identify matching lab test names and compare old vs new values.
- Use objective, neutral language (e.g. "Value increased from X to Y").
- Do NOT say "Your condition is worsening/improving" or "Your treatment is working/failing".
- Remind the user that lab reference ranges and values must be evaluated by a healthcare professional.

Return ONLY a single valid JSON object matching this structure:
{
  "comparison_summary": "Overall neutral summary of changes noted between reports",
  "matching_tests": [
    {
      "test_name": "Test Name",
      "report1_date": "Date 1",
      "report1_value": "Value 1",
      "report2_date": "Date 2",
      "report2_value": "Value 2",
      "change_type": "increased | decreased | unchanged",
      "reference_range": "Reference range if available",
      "note": "Neutral educational observation"
    }
  ],
  "key_differences": ["Difference 1", "Difference 2"],
  "questions_for_doctor": ["Question 1", "Question 2"],
  "disclaimer": "This report comparison is for educational tracking only and does not constitute medical advice or diagnosis."
}
"""

TERM_EXPLAINER_PROMPT = """You are MediGuide AI. Explain a medical term in plain, simple language for a patient.

Return ONLY a single valid JSON object matching this structure:
{
  "term": "Term Name",
  "plain_meaning": "Simple plain-English explanation",
  "why_it_appears": "Common context in medical reports",
  "important_context": "What factors influence this",
  "questions_for_doctor": ["Question to ask a physician"]
}
"""

DOCUMENT_INTELLIGENCE_PROMPT = f"""You are MediGuide AI's document intelligence module. Read the extracted text of a
single uploaded medical document and extract ONLY what is explicitly present. This is a document-organization
task, NOT a diagnosis task.

STRICT RULES:
- Never invent a date, doctor, hospital, value, or medication that is not present in the text.
- If something is not present, use the exact string "Not identified" (or an empty list for list fields).
- Do not state or imply the patient has any condition. If a condition/topic is referenced, phrase it as
  "Referenced in document", never as a fact about the patient.
- Specialties/topics should only be included when the document content actually supports them. Do not over-tag.
- A document may reasonably map to zero, one, or multiple specialties/topics.

Valid specialty values (use only ones supported by the text; you may also propose a close match from this list):
{", ".join(SPECIALTY_KEYWORDS.keys())}

Valid topic values (same rule):
{", ".join(HEALTH_TOPICS)}

Return ONLY a single valid JSON object matching exactly this structure:
{{
  "document_date": "YYYY-MM-DD or best available date string, or 'Not identified'",
  "document_type": "Lab Report | Prescription | Doctor Note | Discharge Summary | Imaging / Scan | Vaccination Record | Health Checkup | Medical Document",
  "hospital_or_clinic": "Name or 'Not identified'",
  "doctor_name": "Name or 'Not identified'",
  "doctor_specialty": "Specialty or 'Not identified'",
  "tests_performed": ["Test name", "..."],
  "diagnoses_mentioned": ["Referenced condition/diagnosis text as written", "..."],
  "symptoms_mentioned": ["Symptom as written", "..."],
  "medications_mentioned": ["Medication (+ strength/dose if stated)", "..."],
  "lab_values": [ {{"test_name": "...", "value": "...", "reference_range": "Not identified or range", "date": "YYYY-MM-DD or 'Not identified'"}} ],
  "imaging_findings": ["Finding as written", "..."],
  "procedures": ["Procedure name", "..."],
  "follow_up_recommendation": "As written, or 'Not identified'",
  "specialties": ["Specialty label(s) from the allowed list this document genuinely supports"],
  "topics": ["Topic label(s) from the allowed list this document genuinely supports"],
  "confidence": {{"document_date": "high|medium|low", "specialty": "high|medium|low", "doctor": "high|medium|low"}}
}}
"""

HEALTH_QUERY_PROMPT = """You are MediGuide AI's longitudinal health history assistant. You will be given a user's
natural-language question about their own health history, plus a compact set of ALREADY-FILTERED health events and
extracted facts drawn from their uploaded documents (each fact carries a source_document_id and date).

STRICT RULES:
- Use ONLY the provided events/facts. Never invent a date, value, doctor, or document.
- Never diagnose. Never state the user "has" a condition — say a topic/condition was "referenced in documents".
- Every specific claim (a value, a date, a finding) must be traceable to a source_document_id given to you. If you
  cannot trace it, leave it out.
- If the provided data has gaps (e.g. no records in some sub-range), mention this plainly rather than staying silent.
- Do not fabricate additional documents/events beyond what is provided.

Return ONLY a single valid JSON object matching exactly this structure:
{
  "overview": "2-4 sentence neutral overview of what the provided records show for this query",
  "timeline": [ {"date": "YYYY-MM-DD", "title": "...", "detail": "...", "source_document_id": "doc_..."} ],
  "major_changes": ["Neutral, objective statement of a documented change, e.g. 'LDL increased from 120 to 142 mg/dL between May 2023 and May 2024.'"],
  "tests_and_investigations": [ {"test_name": "...", "date": "...", "value": "Not identified or value", "source_document_id": "doc_..."} ],
  "specialist_consultations": [ {"specialty": "...", "date": "...", "doctor": "Not identified or name", "source_document_id": "doc_..."} ],
  "medications_mentioned": [ {"medication": "...", "date": "...", "source_document_id": "doc_..."} ],
  "recurring_topics": ["Topic label", "..."],
  "missing_periods": ["Plain statement such as 'No records were found between 2020 and 2022.'"],
  "questions_for_doctor": ["Optional neutral question the patient may want to ask, grounded only in documented gaps or changes"],
  "disclaimer": "This is an organizational summary of uploaded records, not a medical diagnosis."
}
"""

SPECIALIST_PREP_PROMPT = """You are MediGuide AI's specialist-visit preparation module. You are given a specialty
and a set of ALREADY-FILTERED events/facts relevant to that specialty, drawn from the user's uploaded documents.
Compile them into a briefing the patient can bring to their appointment.

STRICT RULES:
- Use ONLY the provided data. Never invent values, dates, doctors, or findings.
- NEVER state or imply a diagnosis. Organize the existing information; do not interpret it medically.
- Every claim must be traceable to a source_document_id.
- Generate "questions_for_doctor" only from genuinely documented gaps, changes, or unclear points in the data
  (e.g. a value trending in one direction, a follow-up recommendation that was never revisited).

Return ONLY a single valid JSON object matching exactly this structure (same shape used across MediGuide's
longitudinal history features, so keep the field names exactly as given):
{
  "specialty": "...",
  "overview": "2-4 sentence neutral overview of this specialty's documented history",
  "timeline": [ {"date": "YYYY-MM-DD", "title": "...", "detail": "...", "source_document_id": "doc_..."} ],
  "major_changes": ["Neutral objective statement of a documented change over time"],
  "tests_and_investigations": [ {"test_name": "...", "value": "...", "date": "...", "source_document_id": "doc_..."} ],
  "specialist_consultations": [ {"specialty": "...", "date": "...", "doctor": "Not identified or name", "source_document_id": "doc_..."} ],
  "medications_mentioned": [ {"medication": "...", "date": "...", "source_document_id": "doc_..."} ],
  "recurring_topics": ["Topic label", "..."],
  "missing_periods": ["Plain statement such as 'No records were found between 2020 and 2022.'"],
  "questions_for_doctor": ["Question grounded only in the documented gaps or changes"],
  "disclaimer": "This briefing organizes existing records for discussion with your specialist. It is not a diagnosis."
}
"""


# ---------------------------------------------------------------------------
# Multi-Provider AI Adapter Subsystem
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 30
ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class ProviderConfig:
    name: str
    api_key: str = ""
    model: str = ""
    endpoint: str = ""
    org_id: str = ""
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)


class BaseProvider:
    display_name = "Base Provider"

    def __init__(self, config: ProviderConfig):
        self.config = config

    def _fail(self, message: str, raw_status: Optional[int] = None) -> Dict[str, Any]:
        return {"ok": False, "text": "", "error": message, "raw_status": raw_status}

    def _ok(self, text: str, raw_status: Optional[int] = None) -> Dict[str, Any]:
        return {"ok": True, "text": text, "error": None, "raw_status": raw_status}

    def chat(self, messages: List[Dict[str, str]], system_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    def test_connection(self) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def mask_key(key: str) -> str:
        if not key:
            return ""
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:3]}{'*' * (len(key) - 6)}{key[-3:]}"

    @staticmethod
    def extract_error_detail(resp) -> Optional[str]:
        try:
            data = resp.json()
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        msg = None
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("code")
        elif isinstance(err, str):
            msg = err
        if not msg:
            msg = data.get("message")
        if not msg:
            return None
        return str(msg).strip()[:220]

    def _detail_suffix(self, resp) -> str:
        detail = self.extract_error_detail(resp)
        return f" Details: {detail}" if detail else ""


class GenericOpenAIProvider(BaseProvider):
    display_name = "OpenAI-Compatible"
    default_base_url = "https://api.openai.com/v1"

    def _base_url(self) -> str:
        return (self.config.endpoint or self.default_base_url).rstrip("/")

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        if self.config.org_id:
            headers["OpenAI-Organization"] = self.config.org_id
        headers.update(self.config.extra_headers or {})
        return headers

    def _build_messages(self, messages: List[Dict[str, str]], system_prompt: str):
        chat_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            role = m.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            chat_messages.append({"role": role, "content": m.get("content", "")})
        return chat_messages

    def chat(self, messages: List[Dict[str, str]], system_prompt: str) -> Dict[str, Any]:
        if not self.config.api_key:
            return self._fail("No API key configured for this provider.")
        if not self.config.model:
            return self._fail("No model configured for this provider.")

        url = f"{self._base_url()}/chat/completions"
        body = {
            "model": self.config.model,
            "messages": self._build_messages(messages, system_prompt),
            "temperature": 0.4,
        }
        body.update(self.config.extra_params or {})

        try:
            resp = requests.post(url, headers=self._headers(), json=body, timeout=DEFAULT_TIMEOUT)
        except requests.exceptions.Timeout:
            return self._fail("The AI provider timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            return self._fail("Could not connect to the AI provider. Check the endpoint URL.")
        except requests.exceptions.RequestException:
            return self._fail("A network error occurred while contacting the AI provider.")

        if resp.status_code == 401:
            detail = self._detail_suffix(resp)
            return self._fail(f"Authentication failed. Please check your API key.{detail}", resp.status_code)
        if resp.status_code == 400:
            detail = self._detail_suffix(resp)
            return self._fail(f"The provider rejected the request.{detail or ' Check your model name and configuration.'}", resp.status_code)
        if resp.status_code == 404:
            detail = self._detail_suffix(resp)
            return self._fail(f"The model or endpoint was not found. Check your model name.{detail}", resp.status_code)
        if resp.status_code == 429:
            detail = self._detail_suffix(resp)
            return self._fail(f"Rate limit reached for this provider. Please wait and try again.{detail}", resp.status_code)
        if resp.status_code == 402:
            detail = self._detail_suffix(resp)
            return self._fail(f"The provider reported a billing/credits issue on this account.{detail}", resp.status_code)
        if resp.status_code >= 500:
            return self._fail("The AI provider's servers are currently unavailable.", resp.status_code)
        if resp.status_code != 200:
            detail = self._detail_suffix(resp)
            return self._fail(f"The AI provider rejected the request.{detail or ' Check your configuration.'}", resp.status_code)

        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            return self._fail("The AI provider returned an unexpected response format.")

        return self._ok(text, resp.status_code)

    def test_connection(self) -> Dict[str, Any]:
        if not self.config.api_key:
            return {"ok": False, "message": "No API key provided."}
        if not self.config.model:
            return {"ok": False, "message": "No model specified."}

        result = self.chat(
            [{"role": "user", "content": "Reply with the single word: OK"}],
            "You are a connection test. Reply with only the word OK.",
        )
        if result["ok"]:
            return {"ok": True, "message": f"Connected successfully.\n\nModel: {self.config.model}"}
        return {"ok": False, "message": result["error"] or "Connection failed."}


class OpenAIProvider(GenericOpenAIProvider):
    display_name = "OpenAI"
    default_base_url = "https://api.openai.com/v1"


class OpenRouterProvider(GenericOpenAIProvider):
    display_name = "OpenRouter"
    default_base_url = "https://openrouter.ai/api/v1"

    def _headers(self) -> Dict[str, str]:
        headers = super()._headers()
        headers.setdefault("HTTP-Referer", "http://localhost:5000")
        headers.setdefault("X-Title", "MediGuide AI")
        return headers


class GrokProvider(GenericOpenAIProvider):
    display_name = "Grok / xAI"
    default_base_url = "https://api.x.ai/v1"


class DeepSeekProvider(GenericOpenAIProvider):
    display_name = "DeepSeek"
    default_base_url = "https://api.deepseek.com/v1"


class MistralProvider(GenericOpenAIProvider):
    display_name = "Mistral"
    default_base_url = "https://api.mistral.ai/v1"


class TogetherProvider(GenericOpenAIProvider):
    display_name = "Together AI"
    default_base_url = "https://api.together.xyz/v1"


class CustomOpenAICompatibleProvider(GenericOpenAIProvider):
    display_name = "Custom OpenAI-Compatible API"
    default_base_url = ""


class AnthropicProvider(BaseProvider):
    display_name = "Anthropic"
    default_base_url = "https://api.anthropic.com/v1"

    def _base_url(self) -> str:
        return (self.config.endpoint or self.default_base_url).rstrip("/")

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        headers.update(self.config.extra_headers or {})
        return headers

    def chat(self, messages: List[Dict[str, str]], system_prompt: str) -> Dict[str, Any]:
        if not self.config.api_key:
            return self._fail("No API key configured for this provider.")
        if not self.config.model:
            return self._fail("No model configured for this provider.")

        url = f"{self._base_url()}/messages"
        chat_messages = []
        for m in messages:
            role = m.get("role", "user")
            if role not in ("user", "assistant"):
                role = "user"
            chat_messages.append({"role": role, "content": m.get("content", "")})

        body = {
            "model": self.config.model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": chat_messages,
        }
        body.update(self.config.extra_params or {})

        try:
            resp = requests.post(url, headers=self._headers(), json=body, timeout=DEFAULT_TIMEOUT)
        except requests.exceptions.Timeout:
            return self._fail("The AI provider timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            return self._fail("Could not connect to the AI provider. Check the endpoint URL.")
        except requests.exceptions.RequestException:
            return self._fail("A network error occurred while contacting the AI provider.")

        if resp.status_code == 401:
            detail = self._detail_suffix(resp)
            return self._fail(f"Authentication failed. Please check your API key.{detail}", resp.status_code)
        if resp.status_code == 400:
            detail = self._detail_suffix(resp)
            return self._fail(f"The provider rejected the request.{detail or ' Check your model name and configuration.'}", resp.status_code)
        if resp.status_code == 404:
            detail = self._detail_suffix(resp)
            return self._fail(f"The model was not found. Check your model name.{detail}", resp.status_code)
        if resp.status_code == 429:
            detail = self._detail_suffix(resp)
            return self._fail(f"Rate limit reached for this provider. Please wait and try again.{detail}", resp.status_code)
        if resp.status_code >= 500:
            return self._fail("The AI provider's servers are currently unavailable.", resp.status_code)
        if resp.status_code != 200:
            detail = self._detail_suffix(resp)
            return self._fail(f"The AI provider rejected the request.{detail or ' Check your configuration.'}", resp.status_code)

        try:
            data = resp.json()
            parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
            text = "\n".join(parts).strip()
            if not text:
                raise KeyError("no text block")
        except (ValueError, KeyError, TypeError):
            return self._fail("The AI provider returned an unexpected response format.")

        return self._ok(text, resp.status_code)

    def test_connection(self) -> Dict[str, Any]:
        if not self.config.api_key:
            return {"ok": False, "message": "No API key provided."}
        if not self.config.model:
            return {"ok": False, "message": "No model specified."}

        result = self.chat(
            [{"role": "user", "content": "Reply with the single word: OK"}],
            "You are a connection test. Reply with only the word OK.",
        )
        if result["ok"]:
            return {"ok": True, "message": f"Connected successfully.\n\nModel: {self.config.model}"}
        return {"ok": False, "message": result["error"] or "Connection failed."}


class GeminiProvider(BaseProvider):
    display_name = "Google Gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _base_url(self) -> str:
        return (self.config.endpoint or self.default_base_url).rstrip("/")

    def chat(self, messages: List[Dict[str, str]], system_prompt: str) -> Dict[str, Any]:
        if not self.config.api_key:
            return self._fail("No API key configured for this provider.")
        if not self.config.model:
            return self._fail("No model configured for this provider.")

        model = self.config.model
        url = f"{self._base_url()}/models/{model}:generateContent"

        contents = []
        for m in messages:
            role = "model" if m.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

        body = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.4},
        }
        body.update(self.config.extra_params or {})

        headers = {"Content-Type": "application/json"}
        headers.update(self.config.extra_headers or {})

        try:
            resp = requests.post(
                url,
                headers=headers,
                params={"key": self.config.api_key},
                json=body,
                timeout=DEFAULT_TIMEOUT,
            )
        except requests.exceptions.Timeout:
            return self._fail("The AI provider timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            return self._fail("Could not connect to the AI provider. Check the endpoint URL.")
        except requests.exceptions.RequestException:
            return self._fail("A network error occurred while contacting the AI provider.")

        if resp.status_code == 400:
            detail = self._detail_suffix(resp)
            return self._fail(f"The request was rejected.{detail or ' Check your API key and model name.'}", resp.status_code)
        if resp.status_code in (401, 403):
            detail = self._detail_suffix(resp)
            return self._fail(f"Authentication failed. Please check your API key.{detail}", resp.status_code)
        if resp.status_code == 404:
            detail = self._detail_suffix(resp)
            return self._fail(f"The model was not found. Check your model name.{detail}", resp.status_code)
        if resp.status_code == 429:
            detail = self._detail_suffix(resp)
            return self._fail(f"Rate limit reached for this provider. Please wait and try again.{detail}", resp.status_code)
        if resp.status_code >= 500:
            return self._fail("The AI provider's servers are currently unavailable.", resp.status_code)
        if resp.status_code != 200:
            detail = self._detail_suffix(resp)
            return self._fail(f"The AI provider rejected the request.{detail or ' Check your configuration.'}", resp.status_code)

        try:
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (ValueError, KeyError, IndexError, TypeError):
            return self._fail("The AI provider returned an unexpected response format.")

        return self._ok(text, resp.status_code)

    def test_connection(self) -> Dict[str, Any]:
        if not self.config.api_key:
            return {"ok": False, "message": "No API key provided."}
        if not self.config.model:
            return {"ok": False, "message": "No model specified."}

        result = self.chat(
            [{"role": "user", "content": "Reply with the single word: OK"}],
            "You are a connection test. Reply with only the word OK.",
        )
        if result["ok"]:
            return {"ok": True, "message": f"Connected successfully.\n\nModel: {self.config.model}"}
        return {"ok": False, "message": result["error"] or "Connection failed."}


class HuggingFaceProvider(BaseProvider):
    display_name = "Hugging Face"
    default_base_url = "https://router.huggingface.co/v1"

    def _base_url(self) -> str:
        return (self.config.endpoint or self.default_base_url).rstrip("/")

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        headers.update(self.config.extra_headers or {})
        return headers

    def chat(self, messages: List[Dict[str, str]], system_prompt: str) -> Dict[str, Any]:
        if not self.config.api_key:
            return self._fail("No API key configured for this provider.")
        if not self.config.model:
            return self._fail("No model ID configured for this provider.")

        url = f"{self._base_url()}/chat/completions"
        chat_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            role = m.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            chat_messages.append({"role": role, "content": m.get("content", "")})

        body = {
            "model": self.config.model,
            "messages": chat_messages,
            "temperature": 0.4,
        }
        body.update(self.config.extra_params or {})

        try:
            resp = requests.post(url, headers=self._headers(), json=body, timeout=DEFAULT_TIMEOUT)
        except requests.exceptions.Timeout:
            return self._fail("The AI provider timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            return self._fail("Could not connect to the AI provider. Check the endpoint URL.")
        except requests.exceptions.RequestException:
            return self._fail("A network error occurred while contacting the AI provider.")

        if resp.status_code == 401:
            detail = self._detail_suffix(resp)
            return self._fail(f"Authentication failed. Please check your API key.{detail}", resp.status_code)
        if resp.status_code == 404:
            detail = self._detail_suffix(resp)
            return self._fail(f"The model was not found or is not deployed. Check the model ID.{detail}", resp.status_code)
        if resp.status_code == 429:
            detail = self._detail_suffix(resp)
            return self._fail(f"Rate limit reached for this provider. Please wait and try again.{detail}", resp.status_code)
        if resp.status_code >= 500:
            return self._fail("The AI provider's servers are currently unavailable.", resp.status_code)
        if resp.status_code != 200:
            detail = self._detail_suffix(resp)
            return self._fail(f"The AI provider rejected the request.{detail or ' Check your configuration.'}", resp.status_code)

        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            return self._fail("The AI provider returned an unexpected response format.")

        return self._ok(text, resp.status_code)

    def test_connection(self) -> Dict[str, Any]:
        if not self.config.api_key:
            return {"ok": False, "message": "No API key provided."}
        if not self.config.model:
            return {"ok": False, "message": "No model ID specified."}

        result = self.chat(
            [{"role": "user", "content": "Reply with the single word: OK"}],
            "You are a connection test. Reply with only the word OK.",
        )
        if result["ok"]:
            return {"ok": True, "message": f"Connected successfully.\n\nModel: {self.config.model}"}
        return {"ok": False, "message": result["error"] or "Connection failed."}


PROVIDER_REGISTRY = {
    "openrouter": OpenRouterProvider,
    "openai": OpenAIProvider,
    "grok": GrokProvider,
    "deepseek": DeepSeekProvider,
    "gemini": GeminiProvider,
    "huggingface": HuggingFaceProvider,
    "anthropic": AnthropicProvider,
    "mistral": MistralProvider,
    "together": TogetherProvider,
    "custom": CustomOpenAICompatibleProvider,
}

PROVIDER_META = [
    {"id": "openrouter", "name": "OpenRouter", "needs_endpoint": False, "default_endpoint": "https://openrouter.ai/api/v1"},
    {"id": "openai", "name": "OpenAI", "needs_endpoint": False, "default_endpoint": "https://api.openai.com/v1"},
    {"id": "grok", "name": "Grok / xAI", "needs_endpoint": False, "default_endpoint": "https://api.x.ai/v1"},
    {"id": "deepseek", "name": "DeepSeek", "needs_endpoint": False, "default_endpoint": "https://api.deepseek.com/v1"},
    {"id": "gemini", "name": "Google Gemini", "needs_endpoint": False, "default_endpoint": "https://generativelanguage.googleapis.com/v1beta"},
    {"id": "huggingface", "name": "Hugging Face", "needs_endpoint": False, "default_endpoint": "https://router.huggingface.co/v1"},
    {"id": "anthropic", "name": "Anthropic", "needs_endpoint": False, "default_endpoint": "https://api.anthropic.com/v1"},
    {"id": "mistral", "name": "Mistral", "needs_endpoint": False, "default_endpoint": "https://api.mistral.ai/v1"},
    {"id": "together", "name": "Together AI", "needs_endpoint": False, "default_endpoint": "https://api.together.xyz/v1"},
    {"id": "custom", "name": "Custom OpenAI-Compatible API", "needs_endpoint": True, "default_endpoint": ""},
]

DEFAULT_MODELS = {
    "openrouter": "openrouter/auto",
    "openai": "gpt-5.4-mini",
    "grok": "grok-4.6",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-3.8-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "mistral": "mistral-small-latest",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}


def get_provider(provider_id: str, config_dict: Dict[str, Any]) -> Optional[BaseProvider]:
    cls = PROVIDER_REGISTRY.get(provider_id)
    if not cls:
        return None
    config = ProviderConfig(
        name=provider_id,
        api_key=config_dict.get("api_key", "") or "",
        model=config_dict.get("model", "") or "",
        endpoint=config_dict.get("endpoint", "") or "",
        org_id=config_dict.get("org_id", "") or "",
        extra_headers=config_dict.get("extra_headers") or {},
        extra_params=config_dict.get("extra_params") or {},
    )
    return cls(config)


def list_providers():
    return PROVIDER_META


def detect_provider(api_key: str, model: str = "", endpoint: str = "") -> Dict[str, Any]:
    key = (api_key or "").strip()
    ep = (endpoint or "").strip().lower()
    model_l = (model or "").strip().lower()

    if ep:
        if "openrouter" in ep:
            return {"provider_id": "openrouter", "confident": True}
        if "x.ai" in ep:
            return {"provider_id": "grok", "confident": True}
        if "deepseek" in ep:
            return {"provider_id": "deepseek", "confident": True}
        if "generativelanguage.googleapis" in ep:
            return {"provider_id": "gemini", "confident": True}
        if "anthropic" in ep:
            return {"provider_id": "anthropic", "confident": True}
        if "mistral" in ep:
            return {"provider_id": "mistral", "confident": True}
        if "together" in ep:
            return {"provider_id": "together", "confident": True}
        if "huggingface" in ep or "hf.co" in ep:
            return {"provider_id": "huggingface", "confident": True}
        if "openai.com" in ep:
            return {"provider_id": "openai", "confident": True}
        return {"provider_id": "custom", "confident": False}

    if key.startswith("sk-or-"):
        return {"provider_id": "openrouter", "confident": True}
    if key.startswith("sk-ant-"):
        return {"provider_id": "anthropic", "confident": True}
    if key.startswith("hf_"):
        return {"provider_id": "huggingface", "confident": True}
    if key.startswith("xai-"):
        return {"provider_id": "grok", "confident": True}
    if key.startswith("AIzaSy"):
        return {"provider_id": "gemini", "confident": True}
    if key.startswith("sk-") and "deepseek" in model_l:
        return {"provider_id": "deepseek", "confident": True}
    if key.startswith("sk-") and "mistral" in model_l:
        return {"provider_id": "mistral", "confident": True}
    if key.startswith("sk-"):
        return {"provider_id": "openai", "confident": False}

    return {"provider_id": None, "confident": False}


# ---------------------------------------------------------------------------
# Flask App Initialization
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


def _seed_from_env():
    if get_slot("primary") is not None:
        return
    env_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    env_model = os.getenv("OPENROUTER_MODEL", "").strip()
    if env_key and env_model:
        set_slot("primary", {
            "provider": "openrouter",
            "api_key": env_key,
            "model": env_model,
            "endpoint": "",
            "org_id": "",
            "extra_headers": {},
        })


_seed_from_env()


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------

def _masked_config(config):
    if not config:
        return None
    provider = config.get("provider")
    api_key = config.get("api_key", "")
    has_key = bool(provider and api_key)
    return {
        "provider": provider,
        "model": config.get("model"),
        "endpoint": config.get("endpoint", ""),
        "org_id": config.get("org_id", ""),
        "api_key_masked": BaseProvider.mask_key(api_key) if has_key else "",
        "has_key": has_key,
    }


def _extract_json_object(text):
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def _call_active_provider(messages, system_prompt):
    primary_config = get_slot("primary")
    fallback_config = get_slot("fallback")

    if not primary_config:
        return None, "no_provider", False

    provider = get_provider(primary_config.get("provider"), primary_config)
    if provider:
        res = provider.chat(messages, system_prompt)
        if res.get("ok"):
            return res, None, False

    if fallback_config:
        f_provider = get_provider(fallback_config.get("provider"), fallback_config)
        if f_provider:
            res = f_provider.chat(messages, system_prompt)
            if res.get("ok"):
                return res, None, True

    return None, "provider_error", False


# ---------------------------------------------------------------------------
# Longitudinal Health Intelligence: Extraction, Events, Retrieval
# ---------------------------------------------------------------------------

def analyze_document_intelligence(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Builds (or rebuilds) the structured 'intelligence' block for a document.
    Tries the active AI provider for rich extraction; always falls back to
    keyword-based extraction (never fabricates, never blocks on missing keys)."""
    keyword_result = build_keyword_intelligence(doc)
    extracted = (doc.get("extracted_text") or "").strip()

    if not extracted or not get_slot("primary"):
        return keyword_result

    messages = [{
        "role": "user",
        "content": f"Document Category: {doc.get('category')}\nFilename: {doc.get('original_filename')}\n\n"
                   f"Extracted Text:\n{extracted[:6000]}"
    }]
    result, _err_kind, _fallback = _call_active_provider(messages, DOCUMENT_INTELLIGENCE_PROMPT)
    if result is None:
        return keyword_result

    parsed = _extract_json_object(result.get("text", ""))
    if not isinstance(parsed, dict):
        return keyword_result

    # Merge: trust AI content fields, but keep specialties/topics as the union
    # with the keyword pass so a real signal in the text is never silently dropped.
    merged = dict(keyword_result)
    merged.update({k: v for k, v in parsed.items() if v not in (None, "", [])})
    ai_specialties = parsed.get("specialties") or []
    ai_topics = parsed.get("topics") or []
    merged["specialties"] = sorted(set(
        [s for s in ai_specialties if s in SPECIALTY_KEYWORDS] or keyword_result["specialties"]
    ))
    merged["topics"] = sorted(set([t for t in ai_topics if t in HEALTH_TOPICS] + keyword_result["topics"]))
    merged["extraction_method"] = "ai"
    return merged


def get_or_build_intelligence(doc: Dict[str, Any], docs: List[Dict[str, Any]] = None, persist: bool = True) -> Dict[str, Any]:
    if doc.get("intelligence"):
        return doc["intelligence"]
    intelligence = build_keyword_intelligence(doc)
    doc["intelligence"] = intelligence
    doc["intelligence_generated_at"] = datetime.now().isoformat()
    if persist and docs is not None:
        save_documents(docs)
    return intelligence


def _event_date_sort_key(event: Dict[str, Any]) -> str:
    d = event.get("date") or ""
    return d if re.match(r"^\d{4}-\d{2}-\d{2}", d) else "0000-00-00"


def build_health_events(docs: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Derives one Health Event per document from its (cached) intelligence.
    Always kept in sync with the documents store rather than duplicated state."""
    if docs is None:
        docs = load_documents()
    events = []
    changed = False
    for doc in docs:
        if not doc.get("intelligence"):
            doc["intelligence"] = build_keyword_intelligence(doc)
            doc["intelligence_generated_at"] = doc.get("intelligence_generated_at") or datetime.now().isoformat()
            changed = True
        intel = doc["intelligence"]
        date_val = intel.get("document_date") or doc.get("uploaded_at", "")[:10]
        if date_val == "Not identified":
            date_val = doc.get("uploaded_at", "")[:10] or "Unknown"
        events.append({
            "id": "evt_" + doc["id"],
            "doc_id": doc["id"],
            "date": date_val,
            "date_confidence": (intel.get("confidence") or {}).get("document_date", "low"),
            "type": intel.get("document_type", "Medical Document"),
            "title": doc.get("display_name") or intel.get("document_type", "Medical Document"),
            "hospital_or_clinic": intel.get("hospital_or_clinic", "Not identified"),
            "doctor_name": intel.get("doctor_name", "Not identified"),
            "doctor_specialty": intel.get("doctor_specialty", "Not identified"),
            "specialties": intel.get("specialties", []),
            "topics": intel.get("topics", []),
            "tests_performed": intel.get("tests_performed", []),
            "lab_values": intel.get("lab_values", []),
            "medications_mentioned": intel.get("medications_mentioned", []),
            "follow_up_recommendation": intel.get("follow_up_recommendation", "Not identified"),
            "category": doc.get("category"),
            "is_demo": bool(doc.get("is_demo")),
            "extraction_method": intel.get("extraction_method", "keyword"),
        })
    if changed:
        save_documents(docs)
    events.sort(key=_event_date_sort_key, reverse=True)
    return events


def filter_health_events(events: List[Dict[str, Any]], specialty=None, topic=None, doc_type=None,
                          year=None, start_date=None, end_date=None, search=None) -> List[Dict[str, Any]]:
    out = events
    if specialty and specialty != "all":
        out = [e for e in out if specialty in (e.get("specialties") or [])]
    if topic and topic != "all":
        out = [e for e in out if topic in (e.get("topics") or [])]
    if doc_type and doc_type != "all":
        out = [e for e in out if e.get("type") == doc_type]
    if year and year != "all":
        out = [e for e in out if (e.get("date") or "").startswith(str(year))]
    if start_date:
        out = [e for e in out if _event_date_sort_key(e) >= start_date]
    if end_date:
        out = [e for e in out if _event_date_sort_key(e) <= end_date]
    if search:
        s = search.lower()
        out = [e for e in out if s in json.dumps(e).lower()]
    return out


YEAR_RANGE_RE = re.compile(r"\b(19|20)\d{2}\b")


def parse_query_filters(query: str) -> Dict[str, Any]:
    """Lightweight, deterministic query parsing (no AI dependency) so filtering
    always works even with no provider connected."""
    q = (query or "").lower()
    filters: Dict[str, Any] = {"specialty": None, "topic": None, "doc_type": None, "years": None,
                                "start_date": None, "end_date": None}

    n_years_match = re.search(r"last\s+(\d{1,2})\s*year", q)
    if n_years_match:
        n = int(n_years_match.group(1))
        filters["start_date"] = f"{datetime.now().year - n}-01-01"
    elif "last 5 years" in q or "past 5 years" in q:
        filters["start_date"] = f"{datetime.now().year - 5}-01-01"
    elif "last 10 years" in q or "past 10 years" in q:
        filters["start_date"] = f"{datetime.now().year - 10}-01-01"

    range_match = re.search(r"(20\d{2})\s*(?:to|-|–|through)\s*(20\d{2})", q)
    if range_match:
        filters["start_date"] = f"{range_match.group(1)}-01-01"
        filters["end_date"] = f"{range_match.group(2)}-12-31"

    all_years = re.findall(r"\b(?:19|20)\d{2}\b", q)
    if all_years and not filters["start_date"] and not filters["end_date"]:
        if len(all_years) == 1:
            filters["start_date"] = f"{all_years[0]}-01-01"
            filters["end_date"] = f"{all_years[0]}-12-31"

    for specialty, keywords in SPECIALTY_KEYWORDS.items():
        if any(kw in q for kw in keywords) or specialty.lower() in q:
            filters["specialty"] = specialty
            break

    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in q for kw in keywords) or topic.lower() in q:
            filters["topic"] = topic
            break

    for cat, label in DOCUMENT_TYPE_LABELS.items():
        if label.lower() in q or cat.replace("_", " ") in q:
            filters["doc_type"] = label
            break
    if "ecg" in q or "ekg" in q:
        filters["doc_type"] = None  # test-level match handled via search, not doc_type

    return filters


def _compact_event_for_context(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_document_id": e["doc_id"], "date": e["date"], "type": e["type"],
        "title": e["title"], "specialties": e["specialties"], "topics": e["topics"],
        "doctor_specialty": e["doctor_specialty"], "hospital_or_clinic": e["hospital_or_clinic"],
        "tests_performed": e["tests_performed"], "lab_values": e["lab_values"],
        "medications_mentioned": e["medications_mentioned"],
        "follow_up_recommendation": e["follow_up_recommendation"],
    }


def _deterministic_health_summary(events: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """No-AI-provider fallback: builds a structured, source-traceable summary
    directly from filtered events so every feature still works in demo mode."""
    events_sorted = sorted(events, key=_event_date_sort_key)
    years_present = sorted({e["date"][:4] for e in events_sorted if re.match(r"^\d{4}", e["date"])})
    missing = []
    if len(years_present) >= 2:
        y_int = [int(y) for y in years_present]
        for gap_start, gap_end in zip(y_int, y_int[1:]):
            if gap_end - gap_start > 1:
                missing.append(f"No records were found between {gap_start} and {gap_end}.")
    if not events_sorted:
        missing.append(f"No records were found for {label}.")

    tests, meds, specialists, topics_seen = [], [], [], set()
    for e in events_sorted:
        for lv in e.get("lab_values", []):
            tests.append({**lv, "source_document_id": e["doc_id"]})
        for t in e.get("tests_performed", []):
            tests.append({"test_name": t, "value": "Not identified", "date": e["date"], "source_document_id": e["doc_id"]})
        for m in e.get("medications_mentioned", []):
            meds.append({"medication": m, "date": e["date"], "source_document_id": e["doc_id"]})
        if e.get("doctor_specialty") not in ("Not identified", None):
            specialists.append({"specialty": e["doctor_specialty"], "date": e["date"],
                                 "doctor": e.get("doctor_name", "Not identified"), "source_document_id": e["doc_id"]})
        topics_seen.update(e.get("topics", []))

    overview = (
        f"{len(events_sorted)} document(s) matched \"{label}\", spanning "
        f"{years_present[0] if years_present else 'Unknown'}–{years_present[-1] if years_present else 'Unknown'}."
        if events_sorted else f"No uploaded documents matched \"{label}\"."
    )

    return {
        "overview": overview,
        "timeline": [{"date": e["date"], "title": e["title"], "detail": e["type"], "source_document_id": e["doc_id"]} for e in events_sorted],
        "major_changes": [],
        "tests_and_investigations": tests,
        "specialist_consultations": specialists,
        "medications_mentioned": meds,
        "recurring_topics": sorted(topics_seen),
        "missing_periods": missing,
        "questions_for_doctor": [],
        "disclaimer": "This is an organizational summary of uploaded records, not a medical diagnosis.",
        "generated_by": "DOCUMENTED",
    }


def run_structured_health_ai(prompt: str, context_payload: Dict[str, Any], fallback_events: List[Dict[str, Any]],
                              label: str) -> Dict[str, Any]:
    if not get_slot("primary"):
        result = _deterministic_health_summary(fallback_events, label)
        result["generated_by"] = "DOCUMENTED"
        return result

    messages = [{"role": "user", "content": json.dumps(context_payload)}]
    result, _err, _fb = _call_active_provider(messages, prompt)
    if result is None:
        fallback = _deterministic_health_summary(fallback_events, label)
        fallback["generated_by"] = "DOCUMENTED"
        return fallback
    parsed = _extract_json_object(result.get("text", ""))
    if not isinstance(parsed, dict):
        fallback = _deterministic_health_summary(fallback_events, label)
        fallback["generated_by"] = "DOCUMENTED"
        return fallback
    parsed["generated_by"] = "AI-INTERPRETED"
    return parsed


# ---------------------------------------------------------------------------
# Longitudinal Health Intelligence: Demo Data (clearly labeled prototype data)
# ---------------------------------------------------------------------------

def _demo_doc(doc_id, date, category, display_name, intel):
    return {
        "id": doc_id, "filename": None, "original_filename": display_name,
        "display_name": display_name, "category": category, "mime_type": "text/plain",
        "size": 0, "uploaded_at": f"{date}T09:00:00", "processing_status": "completed",
        "extraction_status": "demo", "extracted_text": "", "notes": "", "tags": [category],
        "is_demo": True, "intelligence": intel, "intelligence_generated_at": f"{date}T09:00:00",
    }


def _demo_intel(date, doc_type, specialties, topics, **kw):
    base = {
        "document_date": date, "document_type": doc_type, "hospital_or_clinic": "Not identified",
        "doctor_name": "Not identified", "doctor_specialty": specialties[0] if specialties else "Not identified",
        "tests_performed": [], "diagnoses_mentioned": [], "symptoms_mentioned": [],
        "medications_mentioned": [], "lab_values": [], "imaging_findings": [], "procedures": [],
        "follow_up_recommendation": "Not identified", "specialties": specialties, "topics": topics,
        "confidence": {"document_date": "high", "specialty": "high", "doctor": "medium"},
        "extraction_method": "demo",
    }
    base.update(kw)
    return base


def build_demo_dataset() -> List[Dict[str, Any]]:
    return [
        _demo_doc("demo_doc_001", "2019-03-14", "lab_report", "General Blood Report",
                  _demo_intel("2019-03-14", "Lab Report", ["General Medicine"], [],
                              tests_performed=["Complete Blood Count"])),
        _demo_doc("demo_doc_002", "2020-01-19", "lab_report", "Lipid Profile",
                  _demo_intel("2020-01-19", "Lab Report", ["Cardiology"], ["Cholesterol"],
                              lab_values=[{"test_name": "LDL", "value": "128 mg/dL", "reference_range": "<130 mg/dL", "date": "2020-01-19"},
                                          {"test_name": "HDL", "value": "44 mg/dL", "reference_range": ">40 mg/dL", "date": "2020-01-19"}])),
        _demo_doc("demo_doc_003", "2021-11-02", "doctor_notes", "General Physician Consultation",
                  _demo_intel("2021-11-02", "Doctor Note", ["General Medicine"], [],
                              doctor_specialty="General Medicine", follow_up_recommendation="Routine annual checkup recommended")),
        _demo_doc("demo_doc_004", "2022-08-27", "doctor_notes", "Cardiology Consultation",
                  _demo_intel("2022-08-27", "Doctor Note", ["Cardiology"], ["Heart / Cardiac Health", "Blood Pressure"],
                              symptoms_mentioned=["Occasional palpitations"], doctor_specialty="Cardiology",
                              follow_up_recommendation="ECG recommended at next visit")),
        _demo_doc("demo_doc_005", "2023-02-11", "scan", "ECG Report",
                  _demo_intel("2023-02-11", "Imaging / Scan", ["Cardiology"], ["Heart / Cardiac Health"],
                              imaging_findings=["Normal sinus rhythm"], procedures=["Electrocardiogram"])),
        _demo_doc("demo_doc_006", "2023-02-15", "doctor_notes", "Cardiology Follow-up",
                  _demo_intel("2023-02-15", "Doctor Note", ["Cardiology"], ["Heart / Cardiac Health"],
                              doctor_specialty="Cardiology", follow_up_recommendation="Repeat lipid profile in 12 months")),
        _demo_doc("demo_doc_007", "2023-09-08", "lab_report", "Lipid Profile",
                  _demo_intel("2023-09-08", "Lab Report", ["Cardiology"], ["Cholesterol"],
                              lab_values=[{"test_name": "LDL", "value": "142 mg/dL", "reference_range": "<130 mg/dL", "date": "2023-09-08"},
                                          {"test_name": "HDL", "value": "41 mg/dL", "reference_range": ">40 mg/dL", "date": "2023-09-08"}])),
        _demo_doc("demo_doc_008", "2024-06-17", "scan", "Echocardiogram",
                  _demo_intel("2024-06-17", "Imaging / Scan", ["Cardiology"], ["Heart / Cardiac Health"],
                              imaging_findings=["Normal left ventricular function referenced in report"],
                              procedures=["Echocardiogram"])),
        _demo_doc("demo_doc_009", "2024-09-30", "lab_report", "Thyroid Panel",
                  _demo_intel("2024-09-30", "Lab Report", ["Endocrinology"], ["Thyroid"],
                              lab_values=[{"test_name": "TSH", "value": "3.1 uIU/mL", "reference_range": "0.4-4.0 uIU/mL", "date": "2024-09-30"}])),
        _demo_doc("demo_doc_010", "2025-01-12", "doctor_notes", "Cardiology Consultation",
                  _demo_intel("2025-01-12", "Doctor Note", ["Cardiology"], ["Heart / Cardiac Health", "Cholesterol"],
                              doctor_specialty="Cardiology", medications_mentioned=["Atorvastatin 10mg (referenced)"],
                              follow_up_recommendation="Recheck lipid profile in 6 months")),
        _demo_doc("demo_doc_011", "2026-02-28", "lab_report", "Lipid Profile",
                  _demo_intel("2026-02-28", "Lab Report", ["Cardiology"], ["Cholesterol"],
                              lab_values=[{"test_name": "LDL", "value": "119 mg/dL", "reference_range": "<130 mg/dL", "date": "2026-02-28"},
                                          {"test_name": "HDL", "value": "47 mg/dL", "reference_range": ">40 mg/dL", "date": "2026-02-28"}])),
    ]


# ---------------------------------------------------------------------------
# Page Routes
# ---------------------------------------------------------------------------

@app.route("/")
@app.route("/settings")
@app.route("/documents")
@app.route("/timeline")
@app.route("/symptoms")
@app.route("/medications")
@app.route("/healthcare")
@app.route("/doctors")
@app.route("/calendar")
@app.route("/consultations")
@app.route("/medicines")
@app.route("/analytics")
def home():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API: System & Provider Settings
# ---------------------------------------------------------------------------

@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "service": "MediGuide AI"})


@app.route("/api/health-library")
def api_health_library():
    path = os.path.join(DATA_DIR, "health_knowledge.json")
    items = _read_json_file(path, [])
    return jsonify({"success": True, "items": items})


@app.route("/api/providers")
def api_providers():
    primary = get_slot("primary")
    fallback = get_slot("fallback")
    return jsonify({
        "success": True,
        "providers": list_providers(),
        "primary": _masked_config(primary),
        "fallback": _masked_config(fallback),
    })


@app.route("/api/quick-connect", methods=["POST"])
def api_quick_connect():
    data = request.get_json(silent=True) or {}
    api_key = (data.get("api_key") or "").strip()
    slot = data.get("slot", "primary")
    if slot not in ("primary", "fallback"):
        slot = "primary"

    if not api_key:
        return jsonify({"success": False, "message": "Please paste an API key."}), 400

    detection = detect_provider(api_key)
    provider_id = detection.get("provider_id")

    if not provider_id:
        return jsonify({
            "success": False,
            "connected": False,
            "needs_manual": True,
            "message": "We could not recognize this API key format. Please check the key and try again.",
        })

    model = DEFAULT_MODELS.get(provider_id, "")
    config = {
        "provider": provider_id,
        "api_key": api_key,
        "model": model,
        "endpoint": "",
        "org_id": "",
        "extra_headers": {},
    }
    provider = get_provider(provider_id, config)
    try:
        result = provider.test_connection()
    except Exception:
        return jsonify({"success": False, "connected": False, "message": "An unexpected error occurred while connecting."}), 500

    if result.get("ok"):
        set_slot(slot, config)
        return jsonify({
            "success": True,
            "connected": True,
            "provider": provider_id,
            "model": model,
            "confident_detection": detection.get("confident", False),
        })

    return jsonify({
        "success": False,
        "connected": False,
        "message": result.get("message", "Connection failed. Please check your API key."),
    })


@app.route("/api/delete-api-key", methods=["POST"])
def api_delete_api_key():
    data = request.get_json(silent=True) or {}
    slot = data.get("slot", "primary")
    if slot not in ("primary", "fallback"):
        return jsonify({"success": False, "error": "Invalid slot."}), 400
    set_slot(slot, None)
    return jsonify({"success": True})


@app.route("/api/clear-user-data", methods=["POST"])
def api_clear_user_data():
    clear_all()
    # Delete uploaded files
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
            except OSError:
                pass
    return jsonify({"success": True})


@app.route("/api/export-data", methods=["GET"])
def api_export_data():
    primary = _masked_config(get_slot("primary"))
    fallback = _masked_config(get_slot("fallback"))
    if primary:
        primary.pop("api_key_masked", None)
    if fallback:
        fallback.pop("api_key_masked", None)
    return jsonify({
        "success": True,
        "primary": primary,
        "fallback": fallback,
        "documents": load_documents(),
        "timeline": load_timeline(),
        "symptoms": load_symptoms_log(),
        "medications": load_medications(),
    })


@app.route("/api/import-data", methods=["POST"])
def api_import_data():
    data = request.get_json(silent=True) or {}
    if "documents" in data and isinstance(data["documents"], list):
        save_documents(data["documents"])
    if "timeline" in data and isinstance(data["timeline"], list):
        save_timeline(data["timeline"])
    if "symptoms" in data and isinstance(data["symptoms"], list):
        save_symptoms_log(data["symptoms"])
    if "medications" in data and isinstance(data["medications"], list):
        save_medications(data["medications"])
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# API: Chat & General Assessment
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    history = data.get("history") or []
    language = data.get("language", "English")
    profile_context = data.get("profile_context") or {}

    if not message:
        return jsonify({"success": False, "error": "Message cannot be empty."}), 400
    if len(message) > 4000:
        return jsonify({"success": False, "error": "Message is too long. Please shorten it."}), 400

    red_flags = detect_red_flags(message)

    primary_config = get_slot("primary")
    if not primary_config:
        response = {
            "success": True,
            "reply": "No AI provider is configured yet. Please open Settings to add an API key.",
            "symptoms": [],
            "follow_up_questions": [],
            "possible_causes": [],
            "urgency_level": "emergency" if red_flags else "unknown",
            "recommendation": "",
            "red_flags": red_flags,
            "assessment_complete": False,
            "disclaimer": "This is general information, not a medical diagnosis.",
            "provider_configured": False,
        }
        if red_flags:
            response["reply"] = EMERGENCY_MESSAGE
        return jsonify(response)

    provider_messages = []
    for turn in history[-20:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            provider_messages.append({"role": role, "content": content})
    provider_messages.append({"role": "user", "content": message})

    lang_note = f"\n\nThe user's selected language preference is: {language}."
    context_note = ""
    if profile_context:
        safe_bits = [f"{k}: {v}" for k, v in profile_context.items() if v]
        if safe_bits:
            context_note = "\n\nOptional user context: " + "; ".join(safe_bits) + "."

    full_prompt = SYSTEM_PROMPT + lang_note + context_note
    if red_flags:
        full_prompt += f"\n\nIMPORTANT: Safety check flagged red flags: {', '.join(red_flags)}. Escalated to emergency."

    result, err_kind, used_fallback = _call_active_provider(provider_messages, full_prompt)

    if result is None:
        if red_flags:
            return jsonify({
                "success": True,
                "reply": EMERGENCY_MESSAGE,
                "urgency_level": "emergency",
                "red_flags": red_flags,
                "disclaimer": "This is general information, not a medical diagnosis.",
                "provider_configured": True,
            })
        return jsonify({"success": False, "error": "I'm having trouble connecting to the AI provider."}), 502

    parsed = _extract_json_object(result.get("text", ""))
    if parsed is None:
        return jsonify({
            "success": True,
            "reply": result.get("text", "").strip(),
            "symptoms": [],
            "follow_up_questions": [],
            "possible_causes": [],
            "urgency_level": "emergency" if red_flags else "unknown",
            "recommendation": "",
            "red_flags": red_flags,
            "assessment_complete": False,
            "disclaimer": "This is general information, not a medical diagnosis.",
            "provider_configured": True,
            "used_fallback": used_fallback,
        })

    model_red_flags = parsed.get("red_flags") or []
    merged_red_flags = list(dict.fromkeys([*model_red_flags, *red_flags]))
    urgency = parsed.get("urgency_level", "unknown")
    reply_text = parsed.get("reply", "").strip()

    if red_flags and urgency != "emergency":
        urgency = "emergency"
    if red_flags and EMERGENCY_MESSAGE not in reply_text:
        reply_text = EMERGENCY_MESSAGE + "\n\n" + reply_text if reply_text else EMERGENCY_MESSAGE

    return jsonify({
        "success": True,
        "reply": reply_text,
        "symptoms": parsed.get("symptoms", []),
        "follow_up_questions": parsed.get("follow_up_questions", []),
        "possible_causes": parsed.get("possible_causes", []),
        "urgency_level": urgency,
        "recommendation": parsed.get("recommendation", ""),
        "red_flags": merged_red_flags,
        "assessment_complete": bool(parsed.get("assessment_complete", False)),
        "disclaimer": parsed.get("disclaimer", "This is general information, not a medical diagnosis."),
        "provider_configured": True,
        "used_fallback": used_fallback,
    })


# ---------------------------------------------------------------------------
# API: Medical Document Upload, Management, and Extraction
# ---------------------------------------------------------------------------

@app.route("/api/documents/upload", methods=["POST"])
def api_upload_document():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file included in upload request."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    original_filename = secure_filename(file.filename)
    if not allowed_file(original_filename):
        return jsonify({
            "success": False,
            "error": "Unsupported file format. Please upload a PDF, PNG, JPG, JPEG, WEBP, TXT, or DOCX document."
        }), 400

    category = request.form.get("category", "other")
    notes = request.form.get("notes", "").strip()
    user_title = request.form.get("title", "").strip()
    force_upload = (request.form.get("force") or "").lower() in ("1", "true", "yes")

    doc_id = "doc_" + uuid.uuid4().hex[:12]
    ext = original_filename.rsplit(".", 1)[1].lower() if "." in original_filename else "bin"
    stored_filename = f"{doc_id}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, stored_filename)

    file.save(filepath)
    file_size = os.path.getsize(filepath)

    # Run Text Extraction Pipeline
    extracted_text, extraction_status = extract_text_from_file(filepath, ext)

    # --- Duplicate Detection -------------------------------------------------
    content_hash = _hash_text(extracted_text) if extracted_text.strip() else None
    docs = load_documents()
    duplicate_of = None
    if not force_upload:
        for existing in docs:
            same_hash = content_hash and _hash_text(existing.get("extracted_text", "")) == content_hash
            same_name_size = (existing.get("original_filename") == original_filename and existing.get("size") == file_size)
            if same_hash or same_name_size:
                duplicate_of = existing
                break
    if duplicate_of:
        # Remove the just-saved physical file — we're not creating a new document yet.
        try:
            os.remove(filepath)
        except OSError:
            pass
        return jsonify({
            "success": False,
            "duplicate": True,
            "message": "This appears to be a duplicate of an existing report.",
            "existing_document": duplicate_of,
        })

    doc_meta = {
        "id": doc_id,
        "filename": stored_filename,
        "original_filename": original_filename,
        "display_name": user_title or original_filename,
        "category": category,
        "mime_type": file.mimetype or f"application/{ext}",
        "size": file_size,
        "uploaded_at": datetime.now().isoformat(),
        "processing_status": "completed",
        "extraction_status": extraction_status,
        "extracted_text": extracted_text,
        "notes": notes,
        "tags": [category],
    }

    # --- Longitudinal Health Intelligence: best-effort extraction on upload --
    try:
        doc_meta["intelligence"] = analyze_document_intelligence(doc_meta)
        doc_meta["intelligence_generated_at"] = datetime.now().isoformat()
    except Exception:
        doc_meta["intelligence"] = build_keyword_intelligence(doc_meta)
        doc_meta["intelligence_generated_at"] = datetime.now().isoformat()

    docs.insert(0, doc_meta)
    save_documents(docs)

    # Automatically log entry to timeline
    timeline = load_timeline()
    timeline.insert(0, {
        "id": "tl_" + uuid.uuid4().hex[:10],
        "date": doc_meta["uploaded_at"],
        "type": "document",
        "title": f"Uploaded Report: {doc_meta['display_name']}",
        "description": f"Category: {category.replace('_', ' ').title()} · Size: {round(file_size/1024, 1)} KB",
        "related_doc_id": doc_id,
    })
    save_timeline(timeline)

    return jsonify({"success": True, "document": doc_meta})


@app.route("/api/documents", methods=["GET"])
def api_list_documents():
    docs = load_documents()
    cat = request.args.get("category")
    search = request.args.get("search", "").strip().lower()

    filtered = docs
    if cat and cat != "all":
        filtered = [d for d in filtered if d.get("category") == cat]
    if search:
        filtered = [d for d in filtered if search in d.get("display_name", "").lower() or search in d.get("original_filename", "").lower() or search in d.get("notes", "").lower()]

    return jsonify({"success": True, "documents": filtered})


@app.route("/api/documents/<doc_id>", methods=["GET"])
def api_get_document(doc_id):
    docs = load_documents()
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if not doc:
        return jsonify({"success": False, "error": "Document not found."}), 404
    return jsonify({"success": True, "document": doc})


@app.route("/api/documents/<doc_id>/file", methods=["GET"])
def api_serve_document_file(doc_id):
    docs = load_documents()
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if not doc:
        return jsonify({"success": False, "error": "Document not found."}), 404
    if not doc.get("filename"):
        return jsonify({"success": False, "error": "This is prototype demo data and has no underlying file to preview."}), 404
    return send_from_directory(UPLOAD_DIR, doc["filename"])


@app.route("/api/documents/<doc_id>/rename", methods=["POST"])
def api_rename_document(doc_id):
    data = request.get_json(silent=True) or {}
    new_title = (data.get("display_name") or "").strip()
    if not new_title:
        return jsonify({"success": False, "error": "Title cannot be empty."}), 400

    docs = load_documents()
    for d in docs:
        if d["id"] == doc_id:
            d["display_name"] = new_title
            save_documents(docs)
            return jsonify({"success": True, "document": d})

    return jsonify({"success": False, "error": "Document not found."}), 404


@app.route("/api/documents/<doc_id>/notes", methods=["POST"])
def api_save_document_notes(doc_id):
    data = request.get_json(silent=True) or {}
    notes = (data.get("notes") or "").strip()

    docs = load_documents()
    for d in docs:
        if d["id"] == doc_id:
            d["notes"] = notes
            save_documents(docs)
            return jsonify({"success": True, "document": d})

    return jsonify({"success": False, "error": "Document not found."}), 404


@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def api_delete_document(doc_id):
    docs = load_documents()
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if not doc:
        return jsonify({"success": False, "error": "Document not found."}), 404

    # Remove physical file (demo records have no backing file on disk)
    if doc.get("filename"):
        filepath = os.path.join(UPLOAD_DIR, doc["filename"])
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass

    # Update docs store
    updated = [d for d in docs if d["id"] != doc_id]
    save_documents(updated)

    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# API: Document AI Features (Summary, Q&A, Compare)
# ---------------------------------------------------------------------------

@app.route("/api/documents/<doc_id>/summarize", methods=["POST"])
def api_summarize_document(doc_id):
    docs = load_documents()
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if not doc:
        return jsonify({"success": False, "error": "Document not found."}), 404

    extracted = doc.get("extracted_text", "").strip()
    if not extracted:
        return jsonify({
            "success": False,
            "error": "No extracted text is available for this document to generate an AI summary. You may still view the file preview or type your report findings manually into the chat."
        }), 400

    messages = [
        {"role": "user", "content": f"Document Title: {doc['display_name']}\nCategory: {doc['category']}\n\nExtracted Text:\n{extracted[:6000]}"}
    ]
    result, err_kind, used_fallback = _call_active_provider(messages, DOCUMENT_SUMMARY_PROMPT)

    if result is None:
        return jsonify({"success": False, "error": "Could not connect to AI provider to generate summary."}), 502

    parsed = _extract_json_object(result.get("text", ""))
    if parsed is None:
        return jsonify({
            "success": True,
            "raw_summary": result.get("text", ""),
            "disclaimer": "This summary is generated for educational purposes and does not constitute a medical diagnosis.",
        })

    return jsonify({"success": True, "summary": parsed})


@app.route("/api/documents/<doc_id>/ask", methods=["POST"])
def api_ask_document(doc_id):
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"success": False, "error": "Question cannot be empty."}), 400

    docs = load_documents()
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if not doc:
        return jsonify({"success": False, "error": "Document not found."}), 404

    extracted = doc.get("extracted_text", "").strip()
    red_flags = detect_red_flags(question)

    prompt = DOCUMENT_QA_PROMPT + f"\n\nDocument Title: {doc['display_name']}\nCategory: {doc['category']}\n\nExtracted Text:\n{extracted[:6000] if extracted else '[No text extracted]'}"
    if red_flags:
        prompt += f"\n\nIMPORTANT: User question contains safety indicators: {', '.join(red_flags)}."

    messages = [{"role": "user", "content": question}]
    result, err_kind, used_fallback = _call_active_provider(messages, prompt)

    if result is None:
        if red_flags:
            return jsonify({"success": True, "reply": EMERGENCY_MESSAGE, "red_flags": red_flags})
        return jsonify({"success": False, "error": "Could not connect to AI provider."}), 502

    reply = result.get("text", "")
    if red_flags and EMERGENCY_MESSAGE not in reply:
        reply = EMERGENCY_MESSAGE + "\n\n" + reply

    return jsonify({"success": True, "reply": reply, "red_flags": red_flags})


@app.route("/api/documents/compare", methods=["POST"])
def api_compare_documents():
    data = request.get_json(silent=True) or {}
    doc1_id = data.get("doc1_id")
    doc2_id = data.get("doc2_id")

    if not doc1_id or not doc2_id:
        return jsonify({"success": False, "error": "Please select two reports to compare."}), 400

    docs = load_documents()
    doc1 = next((d for d in docs if d["id"] == doc1_id), None)
    doc2 = next((d for d in docs if d["id"] == doc2_id), None)

    if not doc1 or not doc2:
        return jsonify({"success": False, "error": "One or both documents could not be found."}), 404

    text1 = doc1.get("extracted_text", "").strip()
    text2 = doc2.get("extracted_text", "").strip()

    if not text1 or not text2:
        return jsonify({
            "success": False,
            "error": "Both selected documents must have extracted text available to compare them automatically."
        }), 400

    messages = [
        {
            "role": "user",
            "content": f"Report 1 (Date: {doc1.get('uploaded_at')[:10]}, Title: {doc1['display_name']}):\n{text1[:4000]}\n\n"
                       f"Report 2 (Date: {doc2.get('uploaded_at')[:10]}, Title: {doc2['display_name']}):\n{text2[:4000]}"
        }
    ]
    result, err_kind, used_fallback = _call_active_provider(messages, REPORT_COMPARISON_PROMPT)

    if result is None:
        return jsonify({"success": False, "error": "Could not connect to AI provider for comparison."}), 502

    parsed = _extract_json_object(result.get("text", ""))
    if parsed is None:
        return jsonify({"success": True, "raw_comparison": result.get("text", "")})

    return jsonify({"success": True, "comparison": parsed})


@app.route("/api/documents/<doc_id>/analyze", methods=["POST"])
def api_analyze_document(doc_id):
    docs = load_documents()
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if not doc:
        return jsonify({"success": False, "error": "Document not found."}), 404

    try:
        doc["intelligence"] = analyze_document_intelligence(doc)
        doc["intelligence_generated_at"] = datetime.now().isoformat()
        save_documents(docs)
    except Exception:
        return jsonify({"success": False, "error": "Could not analyze this document right now."}), 502

    return jsonify({"success": True, "document": doc})


# ---------------------------------------------------------------------------
# API: Longitudinal Health Intelligence
# ---------------------------------------------------------------------------

@app.route("/api/health-topics", methods=["GET"])
def api_health_topics():
    return jsonify({
        "success": True,
        "specialties": sorted(SPECIALTY_KEYWORDS.keys()),
        "topics": HEALTH_TOPICS,
        "document_types": sorted(set(DOCUMENT_TYPE_LABELS.values())),
    })


@app.route("/api/health-events", methods=["GET"])
def api_health_events():
    docs = load_documents()
    events = build_health_events(docs)
    filtered = filter_health_events(
        events,
        specialty=request.args.get("specialty"),
        topic=request.args.get("topic"),
        doc_type=request.args.get("doc_type"),
        year=request.args.get("year"),
        start_date=request.args.get("start_date"),
        end_date=request.args.get("end_date"),
        search=request.args.get("search", "").strip() or None,
    )
    return jsonify({"success": True, "events": filtered, "total": len(events)})


@app.route("/api/health-memory-stats", methods=["GET"])
def api_health_memory_stats():
    docs = load_documents()
    events = build_health_events(docs)
    years = sorted({e["date"][:4] for e in events if re.match(r"^\d{4}", e.get("date") or "")})
    specialties = sorted({s for e in events for s in e.get("specialties", [])})
    topics = sorted({t for e in events for t in e.get("topics", [])})
    return jsonify({
        "success": True,
        "stats": {
            "documents": len(docs),
            "health_events": len(events),
            "specialties": len(specialties),
            "specialties_list": specialties,
            "topics": len(topics),
            "year_range": f"{years[0]}–{years[-1]}" if years else "No records yet",
            "has_demo_data": any(d.get("is_demo") for d in docs),
        },
    })


@app.route("/api/health-query", methods=["POST"])
def api_health_query():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"success": False, "error": "Please enter a question about your health history."}), 400

    red_flags = detect_red_flags(query)
    docs = load_documents()
    events = build_health_events(docs)
    filters = parse_query_filters(query)
    matched = filter_health_events(
        events, specialty=filters["specialty"], topic=filters["topic"], doc_type=filters["doc_type"],
        start_date=filters["start_date"], end_date=filters["end_date"],
    )
    if not matched and not any([filters["specialty"], filters["topic"], filters["doc_type"],
                                 filters["start_date"], filters["end_date"]]):
        matched = events  # broad/unfiltered question — use everything available

    context_payload = {
        "query": query,
        "filters_applied": filters,
        "events": [_compact_event_for_context(e) for e in matched[:60]],
    }
    summary = run_structured_health_ai(HEALTH_QUERY_PROMPT, context_payload, matched, query)
    summary["filters_applied"] = filters
    summary["matched_document_count"] = len(matched)
    if red_flags:
        summary["safety_notice"] = EMERGENCY_MESSAGE

    return jsonify({"success": True, "result": summary})


@app.route("/api/specialist-prep", methods=["POST"])
def api_specialist_prep():
    data = request.get_json(silent=True) or {}
    specialty = (data.get("specialty") or "").strip()
    date_range = (data.get("date_range") or "all").strip()
    if not specialty:
        return jsonify({"success": False, "error": "Please choose a specialty to prepare for."}), 400

    start_date = None
    if date_range == "5y":
        start_date = f"{datetime.now().year - 5}-01-01"
    elif date_range == "10y":
        start_date = f"{datetime.now().year - 10}-01-01"
    elif date_range == "1y":
        start_date = f"{datetime.now().year - 1}-01-01"

    docs = load_documents()
    events = build_health_events(docs)
    matched = filter_health_events(events, specialty=specialty, start_date=start_date)

    context_payload = {
        "specialty": specialty,
        "date_range": date_range,
        "events": [_compact_event_for_context(e) for e in matched[:60]],
    }
    briefing = run_structured_health_ai(SPECIALIST_PREP_PROMPT, context_payload, matched, specialty)
    briefing.setdefault("specialty", specialty)
    briefing["matched_document_count"] = len(matched)

    return jsonify({"success": True, "briefing": briefing})


@app.route("/api/compile-records", methods=["POST"])
def api_compile_records():
    data = request.get_json(silent=True) or {}
    specialty = (data.get("specialty") or "").strip() or None
    topic = (data.get("topic") or "").strip() or None
    start_date = (data.get("start_date") or "").strip() or None
    end_date = (data.get("end_date") or "").strip() or None

    if not specialty and not topic:
        return jsonify({"success": False, "error": "Choose a specialty or topic to compile records for."}), 400

    docs = load_documents()
    events = build_health_events(docs)
    matched = filter_health_events(events, specialty=specialty, topic=topic, start_date=start_date, end_date=end_date)
    matched_sorted = sorted(matched, key=_event_date_sort_key)

    tests, meds, consultations = [], [], []
    for e in matched_sorted:
        for lv in e.get("lab_values", []):
            tests.append({**lv, "source_document_id": e["doc_id"]})
        for m in e.get("medications_mentioned", []):
            meds.append({"medication": m, "date": e["date"], "source_document_id": e["doc_id"]})
        if e.get("doctor_specialty") not in ("Not identified", None):
            consultations.append({"date": e["date"], "specialty": e["doctor_specialty"],
                                   "doctor": e.get("doctor_name", "Not identified"), "source_document_id": e["doc_id"]})

    label = specialty or topic
    questions = []
    if not matched_sorted:
        questions = []
    compiled = {
        "title": f"{label} Health Record",
        "range": f"{start_date[:4] if start_date else (matched_sorted[0]['date'][:4] if matched_sorted else 'Unknown')}"
                 f"–{end_date[:4] if end_date else (matched_sorted[-1]['date'][:4] if matched_sorted else 'Unknown')}",
        "generated_at": datetime.now().isoformat(),
        "timeline": [{"date": e["date"], "title": e["title"], "type": e["type"], "source_document_id": e["doc_id"]} for e in matched_sorted],
        "consultations": consultations,
        "tests": tests,
        "medications": meds,
        "documents": [{"id": e["doc_id"], "title": e["title"], "date": e["date"]} for e in matched_sorted],
        "questions_for_doctor": questions,
        "disclaimer": "This compilation is an organizational summary of uploaded records and is not a medical diagnosis.",
    }
    return jsonify({"success": True, "compiled": compiled})


@app.route("/api/seed-demo-data", methods=["POST"])
def api_seed_demo_data():
    docs = load_documents()
    if any(d.get("is_demo") for d in docs):
        return jsonify({"success": True, "message": "Demo data is already loaded.", "added": 0})

    demo_docs = build_demo_dataset()
    docs = demo_docs + docs
    save_documents(docs)

    timeline = load_timeline()
    for d in demo_docs:
        timeline.insert(0, {
            "id": "tl_" + uuid.uuid4().hex[:10],
            "date": d["uploaded_at"],
            "type": "document",
            "title": f"[Prototype Demo Data] {d['display_name']}",
            "description": f"Category: {d['category'].replace('_', ' ').title()} · Demo record",
            "related_doc_id": d["id"],
        })
    save_timeline(timeline)

    return jsonify({"success": True, "message": "Prototype demo data loaded.", "added": len(demo_docs)})


@app.route("/api/seed-demo-data", methods=["DELETE"])
def api_clear_demo_data():
    docs = load_documents()
    remaining = [d for d in docs if not d.get("is_demo")]
    removed = len(docs) - len(remaining)
    save_documents(remaining)

    timeline = load_timeline()
    timeline = [t for t in timeline if not (t.get("related_doc_id") or "").startswith("demo_doc_")]
    save_timeline(timeline)

    return jsonify({"success": True, "removed": removed})


# ---------------------------------------------------------------------------
# API: Health Timeline
# ---------------------------------------------------------------------------

@app.route("/api/timeline", methods=["GET"])
def api_get_timeline():
    timeline = load_timeline()
    cat = request.args.get("category")
    if cat and cat != "all":
        timeline = [t for t in timeline if t.get("type") == cat]
    return jsonify({"success": True, "timeline": timeline})


@app.route("/api/timeline", methods=["POST"])
def api_add_timeline_entry():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"success": False, "error": "Title is required."}), 400

    entry = {
        "id": "tl_" + uuid.uuid4().hex[:10],
        "date": data.get("date") or datetime.now().isoformat(),
        "type": data.get("type", "note"),
        "title": title,
        "description": (data.get("description") or "").strip(),
        "related_doc_id": data.get("related_doc_id"),
    }
    timeline = load_timeline()
    timeline.insert(0, entry)
    save_timeline(timeline)
    return jsonify({"success": True, "entry": entry})


@app.route("/api/timeline/<entry_id>", methods=["DELETE"])
def api_delete_timeline_entry(entry_id):
    timeline = load_timeline()
    updated = [t for t in timeline if t["id"] != entry_id]
    save_timeline(updated)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# API: Symptom Tracker
# ---------------------------------------------------------------------------

@app.route("/api/symptoms-log", methods=["GET"])
def api_get_symptoms_log():
    logs = load_symptoms_log()
    return jsonify({"success": True, "logs": logs})


@app.route("/api/symptoms-log", methods=["POST"])
def api_add_symptom_log():
    data = request.get_json(silent=True) or {}
    symptom_name = (data.get("symptom_name") or "").strip()
    if not symptom_name:
        return jsonify({"success": False, "error": "Symptom name is required."}), 400

    severity = data.get("severity", "moderate")
    notes = (data.get("notes") or "").strip()
    combined_text = f"{symptom_name} {notes}"
    red_flags = detect_red_flags(combined_text)

    entry = {
        "id": "sym_" + uuid.uuid4().hex[:10],
        "symptom_name": symptom_name,
        "severity": severity,
        "date": data.get("date") or datetime.now().isoformat()[:10],
        "time": data.get("time") or datetime.now().strftime("%H:%M"),
        "duration": (data.get("duration") or "").strip(),
        "trigger": (data.get("trigger") or "").strip(),
        "notes": notes,
        "red_flags": red_flags,
    }

    logs = load_symptoms_log()
    logs.insert(0, entry)
    save_symptoms_log(logs)

    # Log to timeline as well
    timeline = load_timeline()
    timeline.insert(0, {
        "id": "tl_" + uuid.uuid4().hex[:10],
        "date": entry["date"] + "T" + entry["time"],
        "type": "symptom",
        "title": f"Symptom Recorded: {symptom_name} ({severity.title()})",
        "description": f"Duration: {entry['duration'] or 'N/A'} · Trigger: {entry['trigger'] or 'None'}",
    })
    save_timeline(timeline)

    return jsonify({
        "success": True,
        "entry": entry,
        "red_flags": red_flags,
        "emergency_message": EMERGENCY_MESSAGE if red_flags else None,
    })


@app.route("/api/symptoms-log/<log_id>", methods=["DELETE"])
def api_delete_symptom_log(log_id):
    logs = load_symptoms_log()
    updated = [l for l in logs if l["id"] != log_id]
    save_symptoms_log(updated)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# API: Medication Information Organizer
# ---------------------------------------------------------------------------

@app.route("/api/medications", methods=["GET"])
def api_get_medications():
    meds = load_medications()
    return jsonify({"success": True, "medications": meds})


@app.route("/api/medications", methods=["POST"])
def api_add_medication():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"success": False, "error": "Medication name is required."}), 400

    entry = {
        "id": "med_" + uuid.uuid4().hex[:10],
        "name": name,
        "strength": (data.get("strength") or "").strip(),
        "instructions": (data.get("instructions") or "").strip(),
        "doctor": (data.get("doctor") or "").strip(),
        "start_date": data.get("start_date") or "",
        "notes": (data.get("notes") or "").strip(),
        "added_at": datetime.now().isoformat(),
    }

    meds = load_medications()
    meds.insert(0, entry)
    save_medications(meds)
    return jsonify({"success": True, "medication": entry})


@app.route("/api/medications/<med_id>", methods=["DELETE"])
def api_delete_medication(med_id):
    meds = load_medications()
    updated = [m for m in meds if m["id"] != med_id]
    save_medications(updated)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# API: Medical Term Explainer & Doctor Visit Summary
# ---------------------------------------------------------------------------

@app.route("/api/explain-term", methods=["POST"])
def api_explain_term():
    data = request.get_json(silent=True) or {}
    term = (data.get("term") or "").strip()
    if not term:
        return jsonify({"success": False, "error": "Please enter a medical term."}), 400

    messages = [{"role": "user", "content": f"Medical term to explain: {term}"}]
    result, err_kind, used_fallback = _call_active_provider(messages, TERM_EXPLAINER_PROMPT)

    if result is None:
        return jsonify({
            "success": True,
            "explanation": {
                "term": term,
                "plain_meaning": f"'{term}' is a medical term commonly found in diagnostic reports.",
                "why_it_appears": "It appears in clinical evaluation notes and laboratory findings.",
                "important_context": "Medical terms must be interpreted alongside your personal health history.",
                "questions_for_doctor": [f"What does '{term}' mean in the context of my results?"],
            }
        })

    parsed = _extract_json_object(result.get("text", ""))
    if parsed is None:
        return jsonify({
            "success": True,
            "explanation": {
                "term": term,
                "plain_meaning": result.get("text", ""),
                "why_it_appears": "",
                "important_context": "",
                "questions_for_doctor": [f"What does '{term}' mean for my overall health?"],
            }
        })

    return jsonify({"success": True, "explanation": parsed})


@app.route("/api/doctor-visit-summary", methods=["POST"])
def api_doctor_visit_summary():
    data = request.get_json(silent=True) or {}
    symptoms = load_symptoms_log()[:5]
    docs = load_documents()[:5]
    meds = load_medications()
    notes = (data.get("custom_notes") or "").strip()

    lines = ["MediGuide AI — Comprehensive Doctor Appointment Summary", "=" * 55, ""]

    lines.append(f"Generated Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    if symptoms:
        lines.append("Recent Logged Symptoms:")
        for s in symptoms:
            lines.append(f"  - {s.get('symptom_name')} ({s.get('severity', '').title()}) on {s.get('date')} - Duration: {s.get('duration', 'N/A')}")
        lines.append("")

    if docs:
        lines.append("Uploaded Relevant Documents:")
        for d in docs:
            lines.append(f"  - [{d.get('category').upper()}] {d.get('display_name')} ({d.get('uploaded_at')[:10]})")
        lines.append("")

    if meds:
        lines.append("Current User-Logged Medications:")
        for m in meds:
            inst = f" - {m.get('instructions')}" if m.get('instructions') else ""
            lines.append(f"  - {m.get('name')} {m.get('strength', '')}{inst}")
        lines.append("")

    if notes:
        lines.append("Patient Concerns & Notes:")
        lines.append(f"  {notes}")
        lines.append("")

    lines.append("Suggested Key Questions for Your Doctor:")
    lines.append("  - What do these recent symptoms or lab findings indicate?")
    lines.append("  - Are any additional diagnostic tests or follow-ups recommended?")
    lines.append("  - Are there any lifestyle modifications or warning signs I should watch for?")
    lines.append("")
    lines.append("Note: Generated by MediGuide AI based solely on information provided by the patient. Not a medical diagnosis.")

    return jsonify({"success": True, "summary": "\n".join(lines)})


# ---------------------------------------------------------------------------
# API: Healthcare Services — Doctor & Specialist Discovery
# ---------------------------------------------------------------------------

@app.route("/api/specialties")
def api_specialties():
    return jsonify({"success": True, "specialties": DEMO_SPECIALTIES})


@app.route("/api/doctors", methods=["GET"])
def api_search_doctors():
    provider = get_doctor_provider()
    if provider is None:
        results = list(DEMO_DOCTORS)
        demo_mode = True
    else:
        results = []  # would call provider.search(...) here
        demo_mode = False

    name_q = (request.args.get("name") or "").strip().lower()
    specialty_q = (request.args.get("specialty") or "").strip()
    location_q = (request.args.get("location") or "").strip().lower()
    online_only = request.args.get("online") == "1"
    in_person_only = request.args.get("in_person") == "1"
    fee_min = request.args.get("fee_min", type=int)
    fee_max = request.args.get("fee_max", type=int)
    sort_by = request.args.get("sort", "distance")

    if name_q:
        results = [d for d in results if name_q in d["name"].lower()]
    if specialty_q and specialty_q != "all":
        results = [d for d in results if d["specialty"] == specialty_q]
    if location_q:
        results = [d for d in results if location_q in d["location"].lower()]
    if online_only:
        results = [d for d in results if d["online"]]
    if in_person_only:
        results = [d for d in results if d["in_person"]]
    if fee_min is not None:
        results = [d for d in results if d["fee"] >= fee_min]
    if fee_max is not None:
        results = [d for d in results if d["fee"] <= fee_max]

    sort_keys = {
        "distance": lambda d: d["distance_km"],
        "rating": lambda d: -d["rating"],
        "experience": lambda d: -d["experience_years"],
        "fee": lambda d: d["fee"],
    }
    results = sorted(results, key=sort_keys.get(sort_by, sort_keys["distance"]))

    return jsonify({"success": True, "doctors": results, "demo_mode": demo_mode, "count": len(results)})


@app.route("/api/doctors/<doctor_id>", methods=["GET"])
def api_get_doctor(doctor_id):
    doctor = next((d for d in DEMO_DOCTORS if d["id"] == doctor_id), None)
    if not doctor:
        return jsonify({"success": False, "error": "Doctor not found."}), 404
    return jsonify({"success": True, "doctor": doctor, "demo_mode": get_doctor_provider() is None})


@app.route("/api/doctors/<doctor_id>/slots", methods=["GET"])
def api_doctor_slots(doctor_id):
    doctor = next((d for d in DEMO_DOCTORS if d["id"] == doctor_id), None)
    if not doctor:
        return jsonify({"success": False, "error": "Doctor not found."}), 404

    date_str = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    all_times = ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
                 "14:00", "14:30", "15:00", "15:30", "16:00", "16:30"]

    # Deterministic pseudo-booking so the same doctor/date always looks consistent
    booked = load_appointments()
    booked_times = {a["time"] for a in booked
                     if a.get("doctor_id") == doctor_id and a.get("date") == date_str
                     and a.get("status") not in ("cancelled",)}
    seed = sum(ord(c) for c in doctor_id + date_str)
    demo_unavailable = {all_times[i % len(all_times)] for i in range(seed % 3)}

    available = [t for t in all_times if t not in booked_times and t not in demo_unavailable]
    return jsonify({"success": True, "date": date_str, "available_slots": available, "demo_mode": get_doctor_provider() is None})


# ---------------------------------------------------------------------------
# API: Healthcare Services — Appointment Booking
# ---------------------------------------------------------------------------

@app.route("/api/appointments", methods=["GET"])
def api_list_appointments():
    appts = load_appointments()
    status = request.args.get("status")
    if status and status != "all":
        appts = [a for a in appts if a.get("status") == status]
    appts = sorted(appts, key=lambda a: (a.get("date", ""), a.get("time", "")))
    return jsonify({"success": True, "appointments": appts})


@app.route("/api/appointments/<appt_id>", methods=["GET"])
def api_get_appointment(appt_id):
    appts = load_appointments()
    appt = next((a for a in appts if a["id"] == appt_id), None)
    if not appt:
        return jsonify({"success": False, "error": "Appointment not found."}), 404
    return jsonify({"success": True, "appointment": appt})


@app.route("/api/appointments", methods=["POST"])
def api_create_appointment():
    data = request.get_json(silent=True) or {}
    doctor_id = data.get("doctor_id")
    doctor = next((d for d in DEMO_DOCTORS if d["id"] == doctor_id), None)
    if not doctor:
        return jsonify({"success": False, "error": "Please select a valid doctor."}), 400

    date = (data.get("date") or "").strip()
    time_slot = (data.get("time") or "").strip()
    consultation_type = data.get("consultation_type", "in_person")
    if not date or not time_slot:
        return jsonify({"success": False, "error": "Please select a date and time."}), 400
    if consultation_type == "in_person" and not doctor["in_person"]:
        return jsonify({"success": False, "error": "This doctor does not offer in-person visits."}), 400
    if consultation_type != "in_person" and not doctor["online"]:
        return jsonify({"success": False, "error": "This doctor does not offer online consultations."}), 400

    appt = {
        "id": "apt_" + uuid.uuid4().hex[:10],
        "doctor_id": doctor["id"],
        "doctor_name": doctor["name"],
        "specialty": doctor["specialty"],
        "clinic": doctor["clinic"],
        "location": doctor["location"],
        "fee": doctor["fee"],
        "date": date,
        "time": time_slot,
        "timezone": data.get("timezone", "Asia/Kolkata"),
        "consultation_type": consultation_type,
        "reason": (data.get("reason") or "").strip(),
        "symptoms_notes": (data.get("symptoms_notes") or "").strip(),
        "additional_notes": (data.get("additional_notes") or "").strip(),
        "preferred_language": data.get("preferred_language", "English"),
        "attached_document_ids": data.get("attached_document_ids", []),
        "questions_for_doctor": data.get("questions_for_doctor", []),
        "status": "upcoming",
        "demo": True,
        "consultation_link": None,
        "reminder_settings": data.get("reminder_settings", {"enabled": True, "before": ["one_day_before"]}),
        "created_at": datetime.now().isoformat(),
    }
    appts = load_appointments()
    appts.append(appt)
    save_appointments(appts)

    add_timeline_event("visit", f"Appointment booked with {doctor['name']}",
                        f"{doctor['specialty']} · {date} at {time_slot} · Demo appointment (not a real booking)",
                        related_id=appt["id"])

    return jsonify({"success": True, "appointment": appt,
                     "notice": "This is a demo appointment. It is not connected to a real provider and is not a confirmed booking."})


@app.route("/api/appointments/<appt_id>/cancel", methods=["POST"])
def api_cancel_appointment(appt_id):
    appts = load_appointments()
    appt = next((a for a in appts if a["id"] == appt_id), None)
    if not appt:
        return jsonify({"success": False, "error": "Appointment not found."}), 404
    appt["status"] = "cancelled"
    save_appointments(appts)
    add_timeline_event("visit", f"Appointment cancelled with {appt['doctor_name']}", "", related_id=appt_id)
    return jsonify({"success": True, "appointment": appt})


@app.route("/api/appointments/<appt_id>/reschedule", methods=["POST"])
def api_reschedule_appointment(appt_id):
    data = request.get_json(silent=True) or {}
    appts = load_appointments()
    appt = next((a for a in appts if a["id"] == appt_id), None)
    if not appt:
        return jsonify({"success": False, "error": "Appointment not found."}), 404
    new_date = (data.get("date") or "").strip()
    new_time = (data.get("time") or "").strip()
    if not new_date or not new_time:
        return jsonify({"success": False, "error": "Please select a new date and time."}), 400
    appt["date"] = new_date
    appt["time"] = new_time
    appt["status"] = "upcoming"
    save_appointments(appts)
    add_timeline_event("visit", f"Appointment rescheduled with {appt['doctor_name']}",
                        f"New date: {new_date} at {new_time}", related_id=appt_id)
    return jsonify({"success": True, "appointment": appt})


@app.route("/api/appointments/<appt_id>/complete", methods=["POST"])
def api_complete_appointment(appt_id):
    data = request.get_json(silent=True) or {}
    appts = load_appointments()
    appt = next((a for a in appts if a["id"] == appt_id), None)
    if not appt:
        return jsonify({"success": False, "error": "Appointment not found."}), 404
    appt["status"] = "completed"
    appt["consultation_notes"] = (data.get("consultation_notes") or "").strip()
    save_appointments(appts)
    add_timeline_event("visit", f"Consultation completed with {appt['doctor_name']}",
                        appt["consultation_notes"] or "Notes saved.", related_id=appt_id)
    return jsonify({"success": True, "appointment": appt})


@app.route("/api/appointments/<appt_id>/ics", methods=["GET"])
def api_appointment_ics(appt_id):
    appts = load_appointments()
    appt = next((a for a in appts if a["id"] == appt_id), None)
    if not appt:
        return jsonify({"success": False, "error": "Appointment not found."}), 404
    ics_content = _generate_ics(appt)
    from flask import Response
    return Response(
        ics_content, mimetype="text/calendar",
        headers={"Content-Disposition": f"attachment; filename=appointment_{appt_id}.ics"},
    )


# ---------------------------------------------------------------------------
# API: Health Calendar — Custom Tasks & Reminders
# ---------------------------------------------------------------------------

@app.route("/api/calendar-tasks", methods=["GET"])
def api_list_calendar_tasks():
    return jsonify({"success": True, "tasks": load_calendar_tasks()})


@app.route("/api/calendar-tasks", methods=["POST"])
def api_add_calendar_task():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    date = (data.get("date") or "").strip()
    if not title or not date:
        return jsonify({"success": False, "error": "Title and date are required."}), 400

    task = {
        "id": "ct_" + uuid.uuid4().hex[:10],
        "title": title,
        "date": date,
        "time": data.get("time", ""),
        "type": data.get("type", "general"),  # lab_test, vaccination, follow_up, general
        "notes": (data.get("notes") or "").strip(),
        "reminder_settings": data.get("reminder_settings", {"enabled": True, "before": ["one_day_before"]}),
        "status": "upcoming",
        "created_at": datetime.now().isoformat(),
    }
    tasks = load_calendar_tasks()
    tasks.append(task)
    save_calendar_tasks(tasks)
    add_timeline_event(task["type"], f"Health task scheduled: {title}", task["notes"], related_id=task["id"])
    return jsonify({"success": True, "task": task})


@app.route("/api/calendar-tasks/<task_id>", methods=["DELETE"])
def api_delete_calendar_task(task_id):
    tasks = load_calendar_tasks()
    updated = [t for t in tasks if t["id"] != task_id]
    save_calendar_tasks(updated)
    return jsonify({"success": True})


@app.route("/api/calendar-tasks/<task_id>/complete", methods=["POST"])
def api_complete_calendar_task(task_id):
    tasks = load_calendar_tasks()
    task = next((t for t in tasks if t["id"] == task_id), None)
    if not task:
        return jsonify({"success": False, "error": "Task not found."}), 404
    task["status"] = "completed"
    save_calendar_tasks(tasks)
    return jsonify({"success": True, "task": task})


@app.route("/api/calendar-events", methods=["GET"])
def api_calendar_events():
    """Combined feed: appointments + custom calendar tasks, for the Health Calendar view."""
    events = []
    for a in load_appointments():
        if a.get("status") == "cancelled":
            continue
        events.append({
            "id": a["id"], "title": f"{a['specialty']} — {a['doctor_name']}",
            "date": a["date"], "time": a.get("time", ""), "type": "appointment",
            "status": a.get("status"), "source": "appointment",
            "consultation_type": a.get("consultation_type"),
        })
    for t in load_calendar_tasks():
        if t.get("status") == "cancelled":
            continue
        events.append({
            "id": t["id"], "title": t["title"], "date": t["date"], "time": t.get("time", ""),
            "type": t.get("type", "general"), "status": t.get("status"), "source": "task",
        })
    events.sort(key=lambda e: (e["date"], e.get("time", "")))
    return jsonify({"success": True, "events": events})


# ---------------------------------------------------------------------------
# API: Online Consultation
# ---------------------------------------------------------------------------

@app.route("/api/consultations", methods=["GET"])
def api_list_consultations():
    appts = [a for a in load_appointments()
             if a.get("consultation_type") in ("video", "audio", "chat") and a.get("status") != "cancelled"]
    appts = sorted(appts, key=lambda a: (a.get("date", ""), a.get("time", "")))
    return jsonify({"success": True, "consultations": appts, "demo_mode": get_doctor_provider() is None})


# ---------------------------------------------------------------------------
# API: Prescriptions & Online Medicine Ordering
# ---------------------------------------------------------------------------

PRESCRIPTION_ALLOWED_EXT = {"pdf", "jpg", "jpeg", "png"}

@app.route("/api/prescriptions", methods=["GET"])
def api_list_prescriptions():
    return jsonify({"success": True, "prescriptions": load_prescriptions()})


@app.route("/api/prescriptions/upload", methods=["POST"])
def api_upload_prescription():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in PRESCRIPTION_ALLOWED_EXT:
        return jsonify({"success": False, "error": "Only PDF, JPG, JPEG, and PNG files are supported."}), 400

    rx_id = "rx_" + uuid.uuid4().hex[:10]
    safe_name = secure_filename(file.filename)
    stored_name = f"{rx_id}_{safe_name}"
    file.save(os.path.join(PRESCRIPTION_UPLOAD_DIR, stored_name))

    record = {
        "id": rx_id,
        "filename": safe_name,
        "stored_name": stored_name,
        "uploaded_at": datetime.now().isoformat(),
        "verification_status": "pending_verification",
    }
    items = load_prescriptions()
    items.insert(0, record)
    save_prescriptions(items)
    return jsonify({"success": True, "prescription": record})


@app.route("/api/prescriptions/<rx_id>", methods=["DELETE"])
def api_delete_prescription(rx_id):
    items = load_prescriptions()
    target = next((p for p in items if p["id"] == rx_id), None)
    if target:
        try:
            os.remove(os.path.join(PRESCRIPTION_UPLOAD_DIR, target["stored_name"]))
        except OSError:
            pass
    updated = [p for p in items if p["id"] != rx_id]
    save_prescriptions(updated)
    return jsonify({"success": True})


@app.route("/api/medicines", methods=["GET"])
def api_search_medicines():
    provider = get_pharmacy_provider()
    results = list(DEMO_MEDICINES)
    q = (request.args.get("q") or "").strip().lower()
    category = (request.args.get("category") or "").strip()
    if q:
        results = [m for m in results if q in m["name"].lower()]
    if category and category != "all":
        results = [m for m in results if m["category"] == category]
    categories = sorted({m["category"] for m in DEMO_MEDICINES})
    return jsonify({"success": True, "medicines": results, "categories": categories, "demo_mode": provider is None})


@app.route("/api/medicine-orders", methods=["GET"])
def api_list_medicine_orders():
    orders = sorted(load_medicine_orders(), key=lambda o: o.get("created_at", ""), reverse=True)
    return jsonify({"success": True, "orders": orders})


@app.route("/api/medicine-orders", methods=["POST"])
def api_create_medicine_order():
    data = request.get_json(silent=True) or {}
    items = data.get("items", [])
    if not items:
        return jsonify({"success": False, "error": "Your cart is empty."}), 400

    resolved_items = []
    total = 0
    requires_prescription = False
    for it in items:
        med = next((m for m in DEMO_MEDICINES if m["id"] == it.get("medicine_id")), None)
        if not med:
            continue
        qty = max(1, int(it.get("quantity", 1)))
        if med["prescription_required"]:
            requires_prescription = True
        line_total = med["price"] * qty
        total += line_total
        resolved_items.append({"medicine_id": med["id"], "name": med["name"], "strength": med["strength"],
                                "quantity": qty, "unit_price": med["price"], "line_total": line_total,
                                "prescription_required": med["prescription_required"]})

    if not resolved_items:
        return jsonify({"success": False, "error": "No valid medicines in cart."}), 400
    if requires_prescription and not data.get("prescription_id"):
        return jsonify({"success": False, "error": "One or more items require a verified prescription. Please upload a prescription first."}), 400

    order = {
        "id": "ord_" + uuid.uuid4().hex[:10],
        "items": resolved_items,
        "total": total,
        "delivery_address": (data.get("delivery_address") or "").strip(),
        "pharmacy": data.get("pharmacy", "Demo Pharmacy Partner"),
        "prescription_id": data.get("prescription_id"),
        "status": "demo_placed",
        "estimated_delivery": "2-4 business days (demo estimate)",
        "demo": True,
        "created_at": datetime.now().isoformat(),
    }
    orders = load_medicine_orders()
    orders.append(order)
    save_medicine_orders(orders)
    add_timeline_event("note", "Medicine order placed (demo)",
                        f"{len(resolved_items)} item(s), total ₹{total} — not a real transaction", related_id=order["id"])
    return jsonify({"success": True, "order": order,
                     "notice": "This is a demo order. It is not connected to a real pharmacy and no payment was processed."})


@app.route("/api/medicine-orders/<order_id>/cancel", methods=["POST"])
def api_cancel_medicine_order(order_id):
    orders = load_medicine_orders()
    order = next((o for o in orders if o["id"] == order_id), None)
    if not order:
        return jsonify({"success": False, "error": "Order not found."}), 404
    order["status"] = "cancelled"
    save_medicine_orders(orders)
    return jsonify({"success": True, "order": order})


# ---------------------------------------------------------------------------
# API: Health Analytics
# ---------------------------------------------------------------------------

@app.route("/api/vitals", methods=["GET"])
def api_list_vitals():
    metric = request.args.get("metric")
    vitals = load_vitals()
    if metric:
        vitals = [v for v in vitals if v.get("metric") == metric]
    vitals = sorted(vitals, key=lambda v: v.get("date", ""))
    return jsonify({"success": True, "vitals": vitals})


VITAL_UNITS = {
    "blood_pressure": "mmHg", "blood_glucose": "mg/dL", "weight": "kg",
    "heart_rate": "bpm", "temperature": "°F", "cholesterol": "mg/dL",
    "hemoglobin": "g/dL",
}

@app.route("/api/vitals", methods=["POST"])
def api_add_vital():
    data = request.get_json(silent=True) or {}
    metric = data.get("metric")
    value = (data.get("value") or "").strip()
    date = (data.get("date") or "").strip()
    if metric not in VITAL_UNITS or not value or not date:
        return jsonify({"success": False, "error": "Please provide a metric, value, and date."}), 400

    entry = {
        "id": "vit_" + uuid.uuid4().hex[:10], "metric": metric, "value": value,
        "unit": VITAL_UNITS[metric], "date": date, "source": "user_entered",
        "notes": (data.get("notes") or "").strip(), "created_at": datetime.now().isoformat(),
    }
    vitals = load_vitals()
    vitals.append(entry)
    save_vitals(vitals)
    return jsonify({"success": True, "vital": entry})


@app.route("/api/vitals/<vital_id>", methods=["DELETE"])
def api_delete_vital(vital_id):
    vitals = load_vitals()
    updated = [v for v in vitals if v["id"] != vital_id]
    save_vitals(updated)
    return jsonify({"success": True})


@app.route("/api/analytics/symptoms", methods=["GET"])
def api_analytics_symptoms():
    logs = load_symptoms_log()
    severity_rank = {"mild": 1, "moderate": 2, "severe": 3}
    severity_over_time = sorted(
        [{"date": l["date"], "severity": l["severity"], "severity_score": severity_rank.get(l["severity"], 2),
          "symptom_name": l["symptom_name"]} for l in logs],
        key=lambda x: x["date"]
    )
    frequency: Dict[str, int] = {}
    triggers: Dict[str, int] = {}
    for l in logs:
        frequency[l["symptom_name"]] = frequency.get(l["symptom_name"], 0) + 1
        if l.get("trigger"):
            triggers[l["trigger"]] = triggers.get(l["trigger"], 0) + 1

    return jsonify({
        "success": True,
        "has_data": len(logs) > 0,
        "severity_over_time": severity_over_time,
        "frequency": [{"symptom_name": k, "count": v} for k, v in sorted(frequency.items(), key=lambda x: -x[1])],
        "top_triggers": [{"trigger": k, "count": v} for k, v in sorted(triggers.items(), key=lambda x: -x[1])[:5]],
        "total_logged": len(logs),
    })


@app.route("/api/analytics/medications", methods=["GET"])
def api_analytics_medications():
    meds = load_medications()
    return jsonify({
        "success": True,
        "has_data": len(meds) > 0,
        "active_medications": len(meds),
        "medications": [{"name": m["name"], "strength": m.get("strength", ""), "start_date": m.get("start_date", "")} for m in meds],
    })


@app.route("/api/analytics/vitals", methods=["GET"])
def api_analytics_vitals():
    vitals = load_vitals()
    by_metric: Dict[str, List[Dict[str, Any]]] = {}
    for v in vitals:
        by_metric.setdefault(v["metric"], []).append({"date": v["date"], "value": v["value"], "unit": v["unit"]})
    for m in by_metric:
        by_metric[m].sort(key=lambda x: x["date"])
    return jsonify({"success": True, "has_data": len(vitals) > 0, "by_metric": by_metric})


@app.route("/api/analytics/summary", methods=["GET"])
def api_analytics_summary():
    today = datetime.now().date()
    appts = load_appointments()
    upcoming = [a for a in appts if a.get("status") == "upcoming"]
    week_ago = (datetime.now().date().toordinal() - 7)
    symptoms_this_week = [s for s in load_symptoms_log()
                           if s.get("date") and _safe_date_ordinal(s["date"]) and _safe_date_ordinal(s["date"]) >= week_ago]
    docs = load_documents()
    meds = load_medications()
    tasks = [t for t in load_calendar_tasks() if t.get("status") == "upcoming"]

    all_upcoming_events = sorted(
        [{"title": a["doctor_name"], "date": a["date"], "time": a.get("time", "")} for a in upcoming] +
        [{"title": t["title"], "date": t["date"], "time": t.get("time", "")} for t in tasks],
        key=lambda e: (e["date"], e["time"])
    )
    next_reminder = all_upcoming_events[0] if all_upcoming_events else None

    return jsonify({
        "success": True,
        "summary": {
            "upcoming_appointments": len(upcoming),
            "symptoms_logged_this_week": len(symptoms_this_week),
            "reports_uploaded": len(docs),
            "active_medications": len(meds),
            "pending_follow_ups": len(tasks),
            "next_reminder": next_reminder,
        }
    })


def _safe_date_ordinal(date_str: str):
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date().toordinal()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# API: Healthcare Services Settings
# ---------------------------------------------------------------------------

@app.route("/api/health-prefs", methods=["GET"])
def api_get_health_prefs():
    return jsonify({"success": True, "prefs": load_health_prefs()})


@app.route("/api/health-prefs", methods=["POST"])
def api_save_health_prefs():
    data = request.get_json(silent=True) or {}
    prefs = load_health_prefs()
    for key in DEFAULT_HEALTH_PREFS:
        if key in data:
            prefs[key] = data[key]
    save_health_prefs(prefs)
    return jsonify({"success": True, "prefs": prefs})


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Not found."}), 404


@app.errorhandler(500)
def server_error(e):
    app.logger.error("Unhandled server error: %s", traceback.format_exc())
    return jsonify({"success": False, "error": "An unexpected server error occurred."}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # Debug mode (Werkzeug debugger) is OFF by default; opt in explicitly for local development only.
    debug = os.getenv("FLASK_DEBUG", "0").strip().lower() in ("1", "true", "yes")
    app.run(host="127.0.0.1", port=port, debug=debug)