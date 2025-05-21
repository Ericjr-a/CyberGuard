from fastapi import FastAPI, Request, Form,File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from tld import get_tld
import numpy as np
from urllib.parse import urlparse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import PyPDF2
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import requests
import json
import time
from sklearn.ensemble import RandomForestClassifier
import base64
import io
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates") 



with open("models/rf_content.pkl", 'rb') as file:
    rf_content_classifier = pickle.load(file)



#with open("models/rf.pkl", 'rb') as file:
 #   rf_classifier = pickle.load(file)




# Loadimg original model (before any retraining)
with open("models/rf.pkl", "rb") as file:
    rf_startup_snapshot = pickle.load(file) 


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' 
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0

def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        return 1
    else:
        return 0

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0

def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0

#First Directory Length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters

def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits

def extract_features(url):
    features = []
    features.append(having_ip_address(url))
    features.append(abnormal_url(url))
    features.append(url.count('.'))
    features.append(url.count('www'))
    features.append(url.count('@'))
    features.append(no_of_dir(url))
    features.append(no_of_embed(url))
    features.append(shortening_service(url))
    features.append(url.count('https'))
    features.append(url.count('http'))
    features.append(url.count('%'))
    features.append(url.count('-'))
    features.append(url.count('='))
    features.append(len(url))
    features.append(len(urlparse(url).netloc))
    features.append(suspicious_words(url))
    features.append(fd_length(url))
    tld = get_tld(url, fail_silently=True)
    features.append(tld_length(tld))
    features.append(digit_count(url))
    features.append(letter_count(url))
    return np.array(features).reshape(1, -1)


with open("models/tfidf.pkl", 'rb') as file:
    tfidf_vectorizer = pickle.load(file)


def clean_text(text):
    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'http\S+', '', text)

    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    stemmer = PorterStemmer()

    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    cleaned_text = ' '.join(stemmed_words)

    return cleaned_text


def predict_fake_or_real(text):
    cleaned_text = clean_text(text)
    text_tfidf = tfidf_vectorizer.transform([cleaned_text])
    prediction = rf_content_classifier.predict(text_tfidf)
    return prediction[0]




def check_google_safe_browsing_file(urls):
    """
    Checks extracted URLs from a file against Google Safe Browsing API.
    Returns a list of malicious URLs or 'safe'.
    """
    if not urls:
        return "safe", []  

    api_url = f"https://webrisk.googleapis.com/v1/uris:search?key={GOOGLE_API_KEY}"
    detected_malicious_urls = []  

    for url in urls:
        params = {"uri": url, "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"]}
        try:
            response = requests.get(api_url, params=params)
            result = response.json()

            if "threat" in result:
                print(f"🚨 Malicious URL found in file: {url}")  
                detected_malicious_urls.append(url) 

        except Exception as e:
            print(f"⚠️ Error checking URL against Google Safe Browsing: {e}")

    return ("malicious" if detected_malicious_urls else "safe", detected_malicious_urls)


from fastapi.responses import JSONResponse


import chardet



 

def check_gemini_file(text):
    """
    Sends the extracted text from a file to Google's Gemini AI for classification.
    Returns whether the text is malicious or safe.
    """
    prompt = f"""
        You are an expert in cybersecurity and forensic document analysis.
        Analyze the given text, which has been extracted from a **file** (PDF, TXT, etc.).
        Classify its intent based on the presence of scams, phishing, or malware-related content.

        ### **Classification Categories:**
        1️⃣ **Legitimate Document** - The file content appears **safe** and **authentic**.  
        2️⃣ **Scam/Fake Document** - The file contains **fraudulent content, misleading claims, or deceptive instructions**.  
        3️⃣ **Phishing Attempt** - The document includes **social engineering techniques** aimed at stealing **credentials** or **personal information**.  
        4️⃣ **Malware Distribution** - The document contains **URLs, scripts, or indicators** suggesting it is used to **spread malware or viruses**.  

        ### **Text Extracted from File:**  
        {text}

        ### **Instructions for Analysis:**  
        - If the document contains **scam, phishing, or malware-related content**, explain **why** it is flagged.  
        - If the document is **legitimate**, confirm that it appears **safe** and **why** it seems real.  
        - If the document contains **suspicious URLs**, identify them and explain why they might be dangerous.  
        - **Do NOT return empty or null responses.** Always provide a **clear classification**.  

        ### **Expected AI Response Format:**  
        ✅ **Legitimate Document:** "This file contains normal, safe content with no indicators of fraud or malicious intent."  
        🚨 **Scam/Fake Document:** "The document includes misleading offers and fake claims, indicating potential fraud."  
        ⚠️ **Phishing Attempt:** "The file contains deceptive login forms designed to steal user credentials."  
        **Malware Distribution:** "Suspicious scripts or links detected, possibly used for spreading malware."

"""



    try:
        response = model.generate_content(prompt)
        classification = response.text.strip()
        return classification
    except Exception as e:
        print(f"⚠️ Error connecting to Gemini API: {e}")
        return "Error analyzing text"






@app.post("/scam/")
async def detect_scam(file: UploadFile = File(...)):
    raw_data = await file.read()

    detected_encoding = chardet.detect(raw_data).get("encoding")

    if detected_encoding is None:
        detected_encoding = "utf-8"  

    try:
        extracted_text = raw_data.decode(detected_encoding, errors="ignore")
    except UnicodeDecodeError:
        extracted_text = raw_data.decode("utf-8", errors="ignore")

    if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        return JSONResponse(content={"error": "Invalid file type. Please upload a PDF or TXT file."}, status_code=400)

    if file.filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(raw_data))  
        extracted_text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

    if not extracted_text.strip():
        return JSONResponse(content={"error": "File is empty or text could not be extracted."}, status_code=400)



    ml_prediction = predict_fake_or_real(extracted_text)
    ml_message = (
        "✅ Legitimate Content: No malicious indicators detected by ML model."
        if ml_prediction == 0
        else "🚨 Warning: This document contains SCAM or MALICIOUS content."
    )



    extracted_urls = extract_urls_from_text(extracted_text)



    gsb_result, malicious_urls = check_google_safe_browsing_file(extracted_urls)


    gemini_result = check_gemini_file(extracted_text)
    gemini_message = (
        "🚨 Gemini AI Detected Malicious Content" if "malicious" in gemini_result.lower()
        else "✅ Gemini AI Found No Threats"
    )

    if gsb_result == "malicious":
        gsb_message = f"🚨 Warning: This document contains MALICIOUS content (Google Safe Browsing flagged URLs in file).<br><br>🚨 Detected Malicious URLs:<br>"
        gsb_message += "<br>".join(f"<a href='{url}' target='_blank'>{url}</a>" for url in malicious_urls)  
    else:
        gsb_message = "✅ Safe: No malicious URLs detected in file by Google Safe browsing."




    vt_result = check_virustotal_file(file)
    vt_message = (
        "🚨 Warning: This file is flagged as MALICIOUS by VirusTotal."
        if vt_result == "malicious"
        else "✅ Safe: No malware detected by VirusTotal."
        if vt_result == "safe"
        else " Virus Total was not able to complete the scan, analyze again to get your results"
    )


    api_results = {
        "virustotal_result": vt_message,
        "google_safe_browsing_result": gsb_message,
        "gemini_result": gemini_message
    }


    return JSONResponse(content={
        "ml_result": ml_message,
        "gsb_result": gsb_message,
        "vt_result": vt_message,
        "gemini_result": gemini_message

    })



def check_virustotal_file(file):
    headers = {"x-apikey": VIRUSTOTAL_API_KEY}
    vt_upload_url = "https://www.virustotal.com/api/v3/files"

    files = {"file": (file.filename, file.file, file.content_type)}

    response = requests.post(vt_upload_url, headers=headers, files=files)

    if response.status_code == 200:
        vt_response = response.json()
        file_id = vt_response.get("data", {}).get("id")

        if not file_id:
            print("⚠️ Error: VirusTotal did not return a valid file ID.")
            return "error"

        report_url = f"https://www.virustotal.com/api/v3/analyses/{file_id}"

        max_retries = 10
        retry_delay = 7  

        for attempt in range(max_retries):
            report_response = requests.get(report_url, headers=headers)

            if report_response.status_code == 200:
                report_data = report_response.json()

                if "data" not in report_data:
                    print("⚠️ Error: Unexpected VirusTotal response format.")
                    return "error"

                status = report_data["data"]["attributes"].get("status", "unknown")

                if status in ["queued", "in-progress"]:
                    print(f" VirusTotal scan still in progress... retrying ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue  

                stats = report_data["data"]["attributes"].get("stats", {})
                malicious_count = stats.get("malicious", 0)

                return "malicious" if malicious_count > 0 else "safe"

            else:
                print(f"⚠️ VirusTotal report not ready yet (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)

        return "pending"  
    else:
        print("⚠️ Error submitting file to VirusTotal")
        return "error"



import hashlib

def extract_urls_from_text(text):
    """
    Extracts URLs from a given text using regex.
    """
    url_pattern = r'https?://[^\s]+'
    return re.findall(url_pattern, text)


def get_file_hash(file):
    """
    Computes the SHA256 hash of an uploaded file.
    """
    hasher = hashlib.sha256()
    file.file.seek(0)  
    while chunk := file.file.read(4096):  
        hasher.update(chunk)  
    file.file.seek(0)  
    return hasher.hexdigest()


import requests


def base64_encode_url(url):
    url_bytes = url.encode('utf-8')
    base64_bytes = base64.urlsafe_b64encode(url_bytes)
    return base64_bytes.decode('utf-8').strip("=")


VIRUSTOTAL_API_KEY = "bab24ad21f58fadf76705fe22d59f337dbd2584a2be9e735a98d138ebcb47b41"  


"""
def check_virustotal(url):
  
    headers = {"x-apikey": VIRUSTOTAL_API_KEY}
    params = {"url": url}
    vt_url = "https://www.virustotal.com/api/v3/urls"

    response = requests.post(vt_url, headers=headers, data=params)

    if response.status_code == 200:
        scan_data = response.json()
        print(" VirusTotal Submission Response:", json.dumps(scan_data, indent=4))  # Debugging

        # Extracting the analysis ID
        url_id = scan_data["data"]["id"]  

        report_url = f"https://www.virustotal.com/api/v3/analyses/{url_id}"

        max_retries = 10  # Maximum retries
        retry_delay = 7   # Wait time in seconds before retrying

        for attempt in range(max_retries):
            report_response = requests.get(report_url, headers=headers)

            if report_response.status_code == 200:
                report_data = report_response.json()
                print("🔍 VirusTotal Report Response:", json.dumps(report_data, indent=4))  

                status = report_data["data"]["attributes"]["status"]
                if status in ["queued", "in-progress"]:
                    print(f"⏳ VirusTotal scan still in progress... retrying ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue  # Retry the request

                try:
                    stats = report_data["data"]["attributes"]["stats"]
                    malicious_count = stats.get("malicious", 0)  

                    if malicious_count > 0:
                        print("🚨 VirusTotal detected MALICIOUS URL!")
                        return "malicious"  # URL flagged as unsafe
                    else:
                        print("✅ VirusTotal detected SAFE URL.")
                        return "safe"  # No issues detected
                except KeyError:
                    print("⚠️ Error extracting VirusTotal stats.")
                    return "unknown"

            else:
                print("⚠️ Could not retrieve VirusTotal report, retrying...")
                time.sleep(retry_delay)  # Wait before retrying

        print(" VirusTotal report not available after multiple retries.")
        return "unknown"

    else:
        print("⚠️ VirusTotal submission failed.")
        return "unknown"


"""



GOOGLE_API_KEY = "AIzaSyBnYL1TD_AROg1w3f24bJqIDfkdHAIX4pM"
def check_google_safe_browsing(url):
    """
    Checks a URL against Google Safe Browsing API.
    Returns "malicious", "safe", or "unknown".
    """
    api_url = f"https://webrisk.googleapis.com/v1/uris:search?key={GOOGLE_API_KEY}"
    params = {
        "uri": url,
        "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"]
    }

    print("\n Sending request to Google Safe Browsing API...")
    print(f" Checking URL: {url}")

    try:
        response = requests.get(api_url, params=params)
        result = response.json()
        print("\n Google Safe Browsing API Response:")
        print(json.dumps(result, indent=4))  

        if "threat" in result:
            print("🚨 Google Safe Browsing detected MALICIOUS URL!")
            return "malicious"

        elif result == {}:
            print("✅ Google Safe Browsing detected SAFE URL.")
            return "safe"

        return "unknown"

    except Exception as e:
        print(f"⚠️ Error connecting to Google Safe Browsing API: {e}")
        return "unknown"





def check_virustotal(url):
    """
    Checks the given URL using VirusTotal API.
    Returns "malicious", "safe", or "unknown".
    """
    headers = {"x-apikey": VIRUSTOTAL_API_KEY}
    params = {"url": url}
    vt_url = "https://www.virustotal.com/api/v3/urls"

    response = requests.post(vt_url, headers=headers, data=params)

    if response.status_code == 200:
        scan_data = response.json()
        url_id = scan_data["data"]["id"]  

        report_url = f"https://www.virustotal.com/api/v3/analyses/{url_id}"
        max_retries = 10
        retry_delay = 5  

        for attempt in range(max_retries):
            report_response = requests.get(report_url, headers=headers)

            if report_response.status_code == 200:
                report_data = report_response.json()
                print("🔍 VirusTotal Report Response:", json.dumps(report_data, indent=4)) 

                status = report_data["data"]["attributes"]["status"]
                if status in ["queued", "in-progress"]:
                    print(f" VirusTotal scan still in progress... retrying ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue

                stats = report_data["data"]["attributes"]["stats"]
                malicious_count = stats.get("malicious", 0)  

                if malicious_count > 0:
                    print(" VirusTotal detected MALICIOUS URL!")
                    return "malicious"
                else:
                    print(" VirusTotal detected SAFE URL.")
                    return "safe"

            else:
                print("⚠️ Could not retrieve VirusTotal report, retrying...")
                time.sleep(retry_delay) 

        print(" VirusTotal report not available after multiple retries.")
        return "unknown"

    else:
        print("⚠️ VirusTotal submission failed.")
        return "unknown"
    






import google.generativeai as genai



os.environ["GOOGLE_API_KEY"] = "AIzaSyBC05viOiDmADdE1VdUbEKBHaKtjjzJ9D4"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")  


def check_gemini_url(url):
    """
    Checks a URL using Google's Gemini AI.
    Returns a classification as 'benign', 'phishing', 'malware', or 'defacement'.
    """
    prompt = f"""
    You are an advanced AI model specializing in URL security classification. Analyze the given URL and classify it as one of the following categories:

    1. Benign: Safe, trusted, and non-malicious websites such as google.com, wikipedia.org, amazon.com.
    2. Phishing: Fraudulent websites designed to steal personal information. Indicators include misspelled domains (e.g., paypa1.com instead of paypal.com), unusual subdomains, and misleading content.
    3. Malware: URLs that distribute viruses, ransomware, or malicious software. Often includes automatic downloads or redirects to infected pages.
    4. Defacement: Hacked or defaced websites that display unauthorized content, usually altered by attackers.

    **Input URL:** {url}

    **Output Format:**  
    - Return only a single word: benign, phishing, malware, or defacement.
    """

    try:
        response = model.generate_content(prompt)
        classification = response.text.strip().lower()
        return classification if classification in ["benign", "phishing", "malware", "defacement"] else "unknown"
    except Exception as e:
        print(f"Error connecting to Gemini API: {e}")
        return "unknown"



@app.post("/predict")
def predict_url(data: dict):
    try:
        url = data.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="Missing URL in request")

        print(f"Checking: {url}")


        url_features = extract_features(url).flatten().reshape(1, -1)

        ml_prediction_before = rf_startup_snapshot.predict(url_features)[0]
        ml_label_before = "malicious" if ml_prediction_before == 1 else "safe"

        #API Checks
        #GSB
        gsb_result = check_google_safe_browsing(url)


        #VIRUS TOTAL 
        vt_result = check_virustotal(url)

        #Gemini
        gemini_result = check_gemini_url(url)



        gsb_label = "malicious" if gsb_result == "malicious" else "safe"
        vt_label = "malicious" if vt_result == "malicious" else "safe"
        gemini_label = "malicious" if gemini_result in ["malicious", "phishing", "defacement", "malware"] else "safe"

        #  Majority Voting from APIs
        label_counts = {"safe": 0, "malicious": 0}
        label_counts[gsb_label] += 1
        label_counts[vt_label] += 1
        label_counts[gemini_label] += 1
        api_majority_label = "malicious" if label_counts["malicious"] > label_counts["safe"] else "safe"

        save_to_dataset(url, api_majority_label)

        retrain_url_model()

        with open("models/rf.pkl", "rb") as file:
            rf_updated = pickle.load(file)

        ml_prediction_after = rf_updated.predict(url_features)[0]
        ml_label_after = "malicious" if ml_prediction_after == 1 else "safe"

        return {
            "url": url,
            "ml_prediction_before": ml_label_before,
            "ml_prediction": ml_label_after,
            "virustotal_result": vt_result,
            "google_safe_browsing_result": gsb_result,
            "gemini_result": gemini_result,
      
        }
    

    except Exception as e:
        print(f" Error in predict_url: {e}")
        raise HTTPException(status_code=500, detail=str(e))



"""
@app.post("/predict")
def predict_url(data: dict):
    try:
        url = data.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="Missing URL in request")

        print(f" Checking: {url}")


        start_ml = time.time()
        url_features = extract_features(url).flatten().reshape(1, -1)

        # Prediction BEFORE Retraining using frozen snapshot
        ml_prediction_before = rf_startup_snapshot.predict(url_features)[0]
        ml_label_before = "malicious" if ml_prediction_before == 1 else "safe"
        end_ml = time.time()
        ml_time = round((end_ml - start_ml) * 1000, 2)

        #API Checks
        #GSB
        start_gsb = time.time()
        gsb_result = check_google_safe_browsing(url)
        end_gsb = time.time()
        gsb_time = round((end_gsb - start_gsb) * 1000, 2)


        #VIRUS TOTAL 
        start_vt = time.time()
        vt_result = check_virustotal(url)
        end_vt = time.time()
        vt_time = round((end_vt - start_vt) * 1000, 2)

        #Gemini
        start_gemini = time.time()
        gemini_result = check_gemini_url(url)
        end_gemini = time.time()
        gemini_time = round((end_gemini - start_gemini) * 1000, 2)



        gsb_label = "malicious" if gsb_result == "malicious" else "safe"
        vt_label = "malicious" if vt_result == "malicious" else "safe"
        gemini_label = "malicious" if gemini_result in ["malicious", "phishing", "defacement", "malware"] else "safe"

        #  Majority Voting from APIs
        label_counts = {"safe": 0, "malicious": 0}
        label_counts[gsb_label] += 1
        label_counts[vt_label] += 1
        label_counts[gemini_label] += 1
        api_majority_label = "malicious" if label_counts["malicious"] > label_counts["safe"] else "safe"

        save_to_dataset(url, api_majority_label)

        retrain_url_model()

        with open("models/rf.pkl", "rb") as file:
            rf_updated = pickle.load(file)

        ml_prediction_after = rf_updated.predict(url_features)[0]
        ml_label_after = "malicious" if ml_prediction_after == 1 else "safe"


        total_time = ml_time + gsb_time + vt_time + gemini_time

           # 🔍 Print latencies to terminal
        print("\nLATENCY MEASUREMENT")
        print(f"ML Time: {ml_time} ms")
        print(f"Google Safe Browsing Time: {gsb_time} ms")
        print(f"VirusTotal Time: {vt_time} ms")
        print(f"Gemini AI Time: {gemini_time} ms")
        print(f"Total Time: {total_time} ms\n")


        # Return all results
        return {
            "url": url,
            "ml_prediction_before": ml_label_before,
            "ml_prediction": ml_label_after,
            "virustotal_result": vt_result,
            "google_safe_browsing_result": gsb_result,
            "gemini_result": gemini_result,
            "latencies": {
                "ml_time_ms": ml_time,
                "gsb_time_ms": gsb_time,
                "vt_time_ms": vt_time,
                "gemini_time_ms": gemini_time,
                "total_pipeline_time_ms": total_time
                }
        }
    
    
    except Exception as e:
        print(f" Error in predict_url: {e}")
        raise HTTPException(status_code=500, detail=str(e))

"""





@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})








































### FILE DETECTION USING APIs

"""   

@app.post("/predict")
def predict_url(data: URLInput):
    try:
        url = data.url


# ✅ Step 1: Check VirusTotal First
        vt_result = check_virustotal(url)

        # Ensure vt_message is assigned correctly
        if vt_result == "malicious":
            vt_message = "Malicious (VirusTotal flagged)"
        elif vt_result == "safe":
            vt_message = "Safe (VirusTotal)"
        else:
            vt_message = "Unknown (VirusTotal check failed)"

        # ✅ If VirusTotal already marked it as malicious/safe, return immediately
        if vt_result in ["malicious", "safe"]:
            return {
                "url": url,
                "predicted_class": vt_message,  # Directly use VirusTotal classification
                "virustotal_result": vt_result
            }
        
        # ✅ Check Google Safe Browsing
        gsb_result = check_google_safe_browsing(url)
        gsb_message = (
            "🚨 Malicious (Google Safe Browsing flagged)" if gsb_result == "malicious"
            else "✅ Safe URL (Verified by Google Safe Browsing)" if gsb_result == "safe"
            else "⚠️ Unknown (Google Safe Browsing check failed)"
        )


        features = extract_features(url)

        # Feature names extracted from the trained model
        feature_names = ['use_of_ip', 'abnormal_url', 'count.', 'count-www', 'count@', 'count_dir',
                         'count_embed_domian', 'short_url', 'count-https', 'count-http', 'count%',
                         'count-', 'count=', 'url_length', 'hostname_length', 'sus_url', 'fd_length',
                         'tld_length', 'count-digits', 'count-letters']

        # Convert features to a DataFrame with matching column names
        features_df = pd.DataFrame(features, columns=feature_names)

        # Make the prediction
        prediction = rf_classifier.predict(features_df)
        predicted_class = le.inverse_transform([prediction[0]])[0]

        return {
            "url": url,
            "predicted_class": predicted_class,
            "virustotal_result": vt_message,
            "google_safe_browsing_result": gsb_message

        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




  """




import pandas as pd
import pickle
import requests
import json
import csv

NEW_DATASET_FILE = "new_url_data.csv" 
from datetime import datetime  

import numpy as np
NEW_DATASET_FILE = "new_url_data.csv"  
PROCESSED_DATASET_FILE = "processed_urls.csv"  





def retrain_url_model():
    """Retrains the ML model using new dataset and archives processed URLs without duplicates."""
    
    if not os.path.exists(NEW_DATASET_FILE) or os.stat(NEW_DATASET_FILE).st_size == 0:
        print("No new URLs found for retraining.")
        return

    try:
        df_new = pd.read_csv(NEW_DATASET_FILE)
    except pd.errors.EmptyDataError:
        print("new_url_data.csv is empty. Skipping retraining.")
        return

    if df_new.shape[0] < 10:
        print(f"Not enough data for retraining (Only {df_new.shape[0]} samples).")
        return  

    print("Retraining model using new user-submitted URLs...")

    X = np.array([extract_features(url).flatten() for url in df_new["url"]])
    y = df_new["label"].values  # Use the majority vote labels

    #Handle class imbalance
    if len(set(y)) == 1:
        print("Retraining skipped: Only one class present in data.")
        return 

    rf_startup_snapshot.fit(X, y)

    with open("models/rf.pkl", "wb") as file:
        pickle.dump(rf_startup_snapshot, file)

    print(" ML model retrained with new user-submitted URLs!")

    if os.path.exists(PROCESSED_DATASET_FILE) and os.stat(PROCESSED_DATASET_FILE).st_size > 0:
        df_existing = pd.read_csv(PROCESSED_DATASET_FILE)
        df_combined = pd.concat([df_existing, df_new[:-10]], ignore_index=True)  # Archive all but last 10
    else:
        df_combined = df_new[:-10]  # If no existing archive, store new data except last 10

    # Remove duplicates before saving
    df_combined.drop_duplicates(subset=["url"], keep="last", inplace=True)

    df_combined.to_csv(PROCESSED_DATASET_FILE, index=False)

    print(f" Processed URLs archived in {PROCESSED_DATASET_FILE} (Duplicates removed)")

    df_new.tail(10).to_csv(NEW_DATASET_FILE, index=False)

    if os.stat(NEW_DATASET_FILE).st_size == 0:
        with open(NEW_DATASET_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["url", "label"])

    print("Unprocessed URLs retained for future retraining!")






PROCESSED_DATASET_FILE = "processed_urls.csv" 

def save_to_dataset(url, label):
    """Saves URL and label (safe/malicious) to dataset for retraining, avoiding duplicates."""
    try:
        file_exists = os.path.exists(NEW_DATASET_FILE)

        existing_urls = set()
        if file_exists:
            try:
                df_new = pd.read_csv(NEW_DATASET_FILE)
                existing_urls.update(df_new["url"].values)  
            except pd.errors.EmptyDataError:
                print("⚠️ new_url_data.csv is empty. Proceeding with new entry.")

        if os.path.exists(PROCESSED_DATASET_FILE):
            try:
                df_processed = pd.read_csv(PROCESSED_DATASET_FILE)
                if url in df_processed["url"].values:
                    print(f"⚠️ URL {url} is already archived in processed_urls.csv. Skipping storage in new_url_data.csv.")
                    return  
            except pd.errors.EmptyDataError:
                print("⚠️ processed_urls.csv is empty.")

        if url not in existing_urls:
            with open(NEW_DATASET_FILE, "a", newline="") as file:
                writer = csv.writer(file)

                if not file_exists or os.stat(NEW_DATASET_FILE).st_size == 0:
                    writer.writerow(["url", "label"])

                cleaned_label = label.strip().lower()
                writer.writerow([url, 1 if cleaned_label == "malicious" else 0])

                print(f" URL saved: {url} as {label}")
        else:
            print(f" URL already exists in new_url_data.csv: {url}")

    except Exception as e:
        print(f" Error saving dataset: {e}")














import schedule
import threading

def run_scheduler():
    """
    Keeps the schedule running in a separate thread 
    so it doesn't block the FastAPI main thread.
    """
    while True:
        schedule.run_pending()
        time.sleep(1)

@app.on_event("startup")
def start_scheduler():
    """
    Called by FastAPI when the server starts.
    Retrains model every 24 hours.
    """
    schedule.every(24).hours.do(retrain_url_model)  

    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()



"""
@app.post("/timing")
def measure_latency(data: dict):
    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing URL")

    #  Start timing for ML model
    start_ml = time.time()
    features = extract_features(url)
    _ = rf_startup_snapshot.predict(features)
    end_ml = time.time()
    ml_time = round((end_ml - start_ml) * 1000, 2)

    #  Google Safe Browsing
    start_gsb = time.time()
    gsb_result = check_google_safe_browsing(url)
    end_gsb = time.time()
    gsb_time = round((end_gsb - start_gsb) * 1000, 2)

    #  VirusTotal
    start_vt = time.time()
    vt_result = check_virustotal(url)
    end_vt = time.time()
    vt_time = round((end_vt - start_vt) * 1000, 2)

    #  Gemini
    start_gemini = time.time()
    gemini_result = check_gemini_url(url)
    end_gemini = time.time()
    gemini_time = round((end_gemini - start_gemini) * 1000, 2)

    #  Combined latency
    total_time = ml_time + gsb_time + vt_time + gemini_time

    #  Print to terminal
    print("\n🔍 LATENCY REPORT")
    print(f"ML Prediction Time: {ml_time} ms")
    print(f"Google Safe Browsing Time: {gsb_time} ms")
    print(f"VirusTotal Time: {vt_time} ms")
    print(f"Gemini AI Time: {gemini_time} ms")
    print(f"TOTAL Pipeline Time: {total_time} ms\n")

    return {
        "ml_model_prediction_time_ms": ml_time,
        "google_safe_browsing_time_ms": gsb_time,
        "virustotal_time_ms": vt_time,
        "gemini_time_ms": gemini_time,
        "total_pipeline_time_ms": total_time
    }

    

    """