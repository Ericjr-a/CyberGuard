# CyberGuard: Malicious Content and URL Detector

**CyberGuard** is a real-time malware and online threat detection system that combines machine learning with threat intelligence APIs to analyze URLs and uploaded files. It features continuous learning, allowing the system to improve over time with new user-submitted data.

---

## Features

- **URL Threat Detection**
  - Detects phishing, malware, defacement, and benign URLs.
  - Utilizes a Random Forest classifier with handcrafted features.
  - Cross-verification with:
    - Google Safe Browsing
    - VirusTotal
    - Gemini AI

- **File Threat Detection**
  - Supports `.pdf` and `.txt` uploads.
  - Extracts text and URLs from files.
  - Classifies content using:
    - TF-IDF + Random Forest model
    - Google Safe Browsing for embedded URLs
    - VirusTotal API
    - Gemini AI text classification

- **Model Retraining**
  - Automatically retrains URL detection model every 24 hours.
  - Incorporates new user-labeled data.
  - Archives processed URLs to avoid duplication.

---

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript  
- **Backend**: FastAPI  
- **Machine Learning**: scikit-learn (Random Forest, TF-IDF)  
- **External APIs**:
  - [VirusTotal](https://www.virustotal.com/)
  - [Google Safe Browsing](https://developers.google.com/safe-browsing)
  - [Gemini AI](https://deepmind.google/technologies/gemini)

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

---

## 🧰 Setup Instructions

1. **Create a virtual environment: python -m venv venv**
 
2. **Activate the virtual environment:**
      **Windows: venv\Scripts\activate**
      **macOS/Linux: source venv/bin/activate**



3. **Install required dependencies:pip install -r requirements.txt**

4. **Run the application using Uvicorn: uvicorn app:app --reload**

5. **Open your browser and visit:**
      **API Docs: http://127.0.0.1:8000/docs**
      **Web Interface: http://127.0.0.1:8000**
