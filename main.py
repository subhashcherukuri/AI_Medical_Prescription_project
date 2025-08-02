import streamlit as st
import re
from PIL import Image
import pytesseract

# Optional: Set path to tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# -------- Mock Data --------
drug_interactions = {
    ("metformin", "simvastatin"): "Moderate interaction – Risk of lactic acidosis.",
    ("ibuprofen", "aspirin"): "High interaction – Increased bleeding risk.",
}

safe_alternatives = {
    "ibuprofen": "acetaminophen",
    "simvastatin": "atorvastatin",
}

dosage_warnings = {
    "metformin": {
        "min_age": 10,
        "max_dose": 2000
    }
}

def extract_drugs(prescription_text):
    known_drugs = ["metformin", "simvastatin", "aspirin", "ibuprofen", "warfarin"]
    found = []
    for drug in known_drugs:
        if re.search(r'\b' + re.escape(drug) + r'\b', prescription_text, re.IGNORECASE):
            found.append(drug.lower())
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

# OCR extraction from uploaded image
def ocr_image(img):
    try:
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"Error during OCR: {e}"

# -------------- Streamlit UI --------------

st.set_page_config(page_title="AI Prescription Verifier", layout="centered")
st.title("🧠 AI Medical Prescription Verifier")
st.markdown("This tool extracts medicine data from prescription text or image and checks for safety.")

option = st.radio("Choose Input Type", ["📝 Text Input", "🖼 Image Upload"])

prescription_text = ""
if option == "📝 Text Input":
    prescription_text = st.text_area("Enter Prescription Text", height=150)

elif option == "🖼 Image Upload":
    uploaded_img = st.file_uploader("Upload Image of Prescription", type=["png", "jpg", "jpeg"])
    if uploaded_img is not None:
        image = Image.open(uploaded_img)
        st.image(image, caption="Uploaded Prescription", use_column_width=True)
        with st.spinner("Extracting text using OCR..."):
            prescription_text = ocr_image(image)
            st.text_area("Extracted Text", prescription_text, height=150)

age = st.number_input("👶 Enter Patient Age (optional)", min_value=0, max_value=120, value=30)

if st.button("🔍 Analyze Prescription"):
    if not prescription_text.strip():
        st.warning("Please enter or upload a valid prescription.")
    else:
        with st.spinner("Analyzing..."):
            drugs = extract_drugs(prescription_text)

            st.subheader("💊 Extracted Drugs")
            if drugs:
                st.success(", ".join(drugs))
            else:
                st.error("No known drugs found.")

            st.subheader("⚠ Drug Interactions")
            interactions = check_interactions(drugs)
            if interactions:
                for pair, msg in interactions:
                    st.error(f"{pair[0].title()} + {pair[1].title()}: {msg}")
            else:
                st.success("No harmful drug interactions detected.")

            st.subheader("👶 Age-Based Dosage Warnings")
            dosage_issues = validate_dosage(drugs, age)
            if dosage_issues:
                for note in dosage_issues:
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