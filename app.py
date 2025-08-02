import streamlit as st
import re
import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

# Load environment variables
load_dotenv()

# Hugging Face model configuration
# Using BioBERT for medical text analysis - no API key required!
MEDICAL_MODEL = "dmis-lab/biobert-base-cased-v1.1"
NER_MODEL = "alvaroalon2/biobert_diseases_ner"

# Initialize Hugging Face models
@st.cache_resource
def initialize_medical_models():
    """Initialize Hugging Face models for medical text analysis"""
    try:
        # Load tokenizer and model for medical NER
        tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
        model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
        
        # Create NER pipeline
        ner_pipeline = pipeline(
            "ner", 
            model=model, 
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Create text generation pipeline for drug extraction
        text_generator = pipeline(
            "text2text-generation", 
            model="google/flan-t5-base",
            device=0 if torch.cuda.is_available() else -1
        )
        
        return ner_pipeline, text_generator
    except Exception as e:
        st.error(f"Failed to initialize Hugging Face models: {str(e)}")
        return None, None

# Drug interaction data (mock)
drug_interactions = {
    ("metformin", "simvastatin"): "Moderate interaction – Risk of lactic acidosis.",
    ("ibuprofen", "aspirin"): "High interaction – Increased bleeding risk.",
}

# Safer alternatives (mock)
safe_alternatives = {
    "ibuprofen": "acetaminophen",
    "simvastatin": "atorvastatin",
}

# Age-based dosage rules (mock logic)
dosage_warnings = {
    "metformin": {
        "min_age": 10,
        "max_dose": 2000  # mg, not used in this sample
    }
}

# Hugging Face model-powered drug extractor
def extract_drugs_with_huggingface(prescription_text, ner_pipeline, text_generator):
    """Extract drug names from prescription text using Hugging Face models"""
    if not ner_pipeline or not text_generator:
        # Fallback to regex if models are not available
        return extract_drugs_fallback(prescription_text)
    
    try:
        # Method 1: Use NER pipeline to find medical entities
        ner_results = ner_pipeline(prescription_text)
        drugs_from_ner = []
        
        for entity in ner_results:
            if entity['label'] in ['DISEASE', 'CHEMICAL'] and entity['score'] > 0.5:
                drug_name = entity['word'].lower().strip()
                if len(drug_name) > 2:
                    drugs_from_ner.append(drug_name)
        
        # Method 2: Use text generation for more comprehensive extraction
        prompt = f"""Extract all medication and drug names from this prescription text. Return only the drug names separated by commas, in lowercase:

{prescription_text}

Drugs:"""
        
        try:
            generation_result = text_generator(prompt, max_length=100, num_return_sequences=1)
            generated_text = generation_result[0]['generated_text'].lower()
            
            # Parse generated response
            drugs_from_generation = []
            if generated_text and 'drugs:' in generated_text:
                drug_part = generated_text.split('drugs:')[-1].strip()
                drugs_from_generation = [drug.strip() for drug in drug_part.split(',') if drug.strip()]
        except:
            drugs_from_generation = []
        
        # Combine results from both methods
        all_drugs = list(set(drugs_from_ner + drugs_from_generation))
        
        # Filter out common non-drug words
        filtered_drugs = []
        for drug in all_drugs:
            drug_clean = drug.strip().lower()
            if (len(drug_clean) > 2 and 
                drug_clean not in ['mg', 'ml', 'tablet', 'capsule', 'daily', 'twice', 'once', 'the', 'and', 'or', 'with']):
                filtered_drugs.append(drug_clean)
        
        return list(set(filtered_drugs)) if filtered_drugs else extract_drugs_fallback(prescription_text)
        
    except Exception as e:
        st.error(f"Error with Hugging Face models: {str(e)}")
        return extract_drugs_fallback(prescription_text)

# Granite model-powered drug extractor
def extract_drugs_with_granite(prescription_text, model):
    """Extract drug names from prescription text using IBM Granite model"""
    if not model:
        # Fallback to regex if model is not available
        return extract_drugs_fallback(prescription_text)
    
    prompt = f"""
You are a medical AI assistant specialized in extracting drug names from prescription text. 
Your task is to identify all medication names mentioned in the following prescription text.

Prescription text: "{prescription_text}"

Instructions:
1. Extract only the drug/medication names (generic or brand names)
2. Return only the drug names, separated by commas
3. Use lowercase for all drug names
4. Do not include dosages, frequencies, or instructions
5. If no drugs are found, return "none"

Drug names:"""

    try:
        response = model.generate_text(prompt=prompt)
        
        # Parse the response to extract drug names
        if response and response.lower() != "none":
            # Clean and split the response
            drugs = [drug.strip().lower() for drug in response.split(',')]
            # Filter out empty strings and common non-drug words
            drugs = [drug for drug in drugs if drug and len(drug) > 2 and drug not in ['mg', 'ml', 'tablet', 'capsule', 'daily', 'twice']]
            return list(set(drugs))  # Remove duplicates
        else:
            return []
    except Exception as e:
        st.error(f"Error with Granite model: {str(e)}")
        # Fallback to regex method
        return extract_drugs_fallback(prescription_text)

# Fallback regex-based drug extractor
def extract_drugs_fallback(prescription_text):
    """Fallback drug extraction using regex (original method)"""
    known_drugs = [
        "metformin", "simvastatin", "aspirin", "ibuprofen", "warfarin",
        "lisinopril", "amlodipine", "omeprazole", "atorvastatin", "losartan",
        "hydrochlorothiazide", "albuterol", "furosemide", "prednisone", "tramadol"
    ]
    found = []
    for drug in known_drugs:
        # Regex finds drug as a separate word, case-insensitive
        if re.search(r'\b' + re.escape(drug) + r'\b', prescription_text, re.IGNORECASE):
            found.append(drug.lower())
    return list(set(found))

# Improved drug extractor using regex for standalone word matching
def extract_drugs(prescription_text):
    known_drugs = ["metformin", "simvastatin", "aspirin", "ibuprofen", "warfarin"]
    found = []
    for drug in known_drugs:
        # Regex finds drug as a separate word, case-insensitive
        if re.search(r'\b' + re.escape(drug) + r'\b', prescription_text, re.IGNORECASE):
            found.append(drug)
    return list(set(found))

def check_interactions(drugs):
    warnings = []
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            pair = tuple(sorted([drugs[i], drugs[j]]))
            if pair in drug_interactions:
                warnings.append((pair, drug_interactions[pair]))
    return warnings

def get_alternatives(drugs):
    return {drug: safe_alternatives[drug] for drug in drugs if drug in safe_alternatives}

def validate_dosage(drugs, age):
    notes = []
    for drug in drugs:
        if drug in dosage_warnings:
            rule = dosage_warnings[drug]
            if age < rule["min_age"]:
                notes.append(f"{drug.title()} is not recommended for age below {rule['min_age']}.")
    return notes

# -------------- Streamlit UI --------------

st.set_page_config(page_title="AI Prescription Verifier with IBM Granite", layout="centered")
st.title("🧠 AI Medical Prescription Verifier")
st.markdown("**Powered by IBM Granite Models** - Advanced drug extraction and safety analysis")
st.markdown("This tool uses IBM's Granite AI model to extract medicine data from prescription text and checks for safety.")

# Initialize the model (cached for performance)
@st.cache_resource
def get_granite_model():
    return initialize_granite_model()

prescription = st.text_area("📝 Enter Prescription Text", height=150, 
                          placeholder="Example: Take Metformin 500mg twice daily and Simvastatin 20mg once at bedtime...")
age = st.number_input("👶 Enter Patient Age (optional)", min_value=0, max_value=120, value=30)

# Model status indicator
model = get_granite_model()
if model:
    st.success("✅ IBM Granite Model: Connected and Ready")
else:
    st.warning("⚠️ IBM Granite Model: Not connected (using fallback method)")

if st.button("🔍 Analyze Prescription"):
    if not prescription.strip():
        st.warning("Please enter a prescription.")
    else:
        with st.spinner("🤖 Analyzing with IBM Granite AI Model..."):
            # Use Granite model for drug extraction
            drugs = extract_drugs_with_granite(prescription, model)
            
            st.subheader("💊 Extracted Drugs (via IBM Granite)")
            if drugs:
                st.success(f"Found: {', '.join([drug.title() for drug in drugs])}")
            else:
                st.error("No known drugs found. Please check the prescription text or try another prescription.")

            st.subheader("⚠ Drug Interactions")
            interactions = check_interactions(drugs)
            if interactions:
                for pair, msg in interactions:
                    st.error(f"{pair[0].title()} + {pair[1].title()}: {msg}")
            else:
                st.success("No harmful drug interactions detected.")

            st.subheader("👶 Age-Based Dosage Warnings")
            dosage_warnings_ = validate_dosage(drugs, age)
            if dosage_warnings_:
                for note in dosage_warnings_:
                    st.warning(note)
            else:
                st.info("Dosage is acceptable for this age.")

            st.subheader("✅ Safer Alternatives")
            alternatives = get_alternatives(drugs)
            if alternatives:
                for drug, alt in alternatives.items():
                    st.info(f"Instead of {drug.title()}, consider using {alt.title()}.")
            else:
                st.success("No alternatives needed.")