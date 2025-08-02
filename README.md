# AI Prescription Verifier with IBM Granite

A medical prescription analysis tool powered by IBM's Granite AI models for intelligent drug extraction and safety verification.

## Features

- **IBM Granite AI Model**: Advanced drug name extraction from prescription text
- **Drug Interaction Detection**: Identifies potentially harmful drug combinations
- **Age-Based Dosage Warnings**: Checks age-appropriate medication usage
- **Safer Alternatives**: Suggests alternative medications when available
- **Fallback System**: Uses regex-based extraction if AI model is unavailable

## Setup Instructions

### 1. IBM watsonx.ai Setup

1. **Create IBM Cloud Account**: Go to [IBM Cloud](https://cloud.ibm.com) and sign up
2. **Create watsonx.ai Project**:
   - Navigate to watsonx.ai service
   - Create a new project
   - Note your Project ID

3. **Get API Key**:
   - Go to IBM Cloud → Manage → Access (IAM) → API keys
   - Create a new API key
   - Copy the API key

4. **Get Service URL**: Choose based on your region:
   - US South: `https://us-south.ml.cloud.ibm.com`
   - EU Germany: `https://eu-de.ml.cloud.ibm.com`
   - US East: `https://us-east.ml.cloud.ibm.com`
   - London: `https://eu-gb.ml.cloud.ibm.com`
   - Tokyo: `https://jp-tok.ml.cloud.ibm.com`

### 2. Environment Configuration

1. **Update .env file** with your credentials:
   ```
   IBM_API_KEY=your_actual_api_key_here
   PROJECT_ID=your_actual_project_id_here
   IBM_URL=your_region_url_here
   ```

### 3. Installation

1. **Install Dependencies**:
   ```bash
   pip install streamlit ibm-watsonx-ai python-dotenv
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## IBM Granite Model Used

This application uses **IBM Granite 13B Chat v2** model, which is:
- A 13-billion parameter large language model
- Optimized for chat and instruction-following tasks
- Excellent for medical text understanding and entity extraction
- Part of IBM's enterprise-grade AI model family

## Model Configuration

- **Decoding Method**: Greedy (for consistent results)
- **Temperature**: 0.1 (low randomness for medical accuracy)
- **Max New Tokens**: 200 (sufficient for drug name extraction)
- **Top P**: 1.0 (considers all tokens)

## Usage

1. Enter prescription text in the text area
2. Optionally enter patient age
3. Click "Analyze Prescription"
4. Review the AI-powered analysis results

## Fallback System

If the IBM Granite model is unavailable, the system automatically falls back to a regex-based drug extraction method to ensure continuous functionality.

## Security Note

Never commit your actual API keys to version control. Always use environment variables or the .env file for sensitive credentials.
