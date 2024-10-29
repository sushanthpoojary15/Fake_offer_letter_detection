from flask import Flask, redirect, render_template, request, url_for
import io
from PyPDF2 import PdfReader

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split

import whois
import dns.resolver
import re
from transformers import pipeline
import language_tool_python
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk,re
from textblob import TextBlob
import os
from joblib import load
from sklearn.preprocessing import StandardScaler
import pandas as pd

from joblib import dump
# Initialize LanguageTool for grammar checking
tool = language_tool_python.LanguageTool('en-US')

corrector = pipeline("text2text-generation", model="facebook/bart-large")




app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'email' not in request.form:
        return redirect(request.url)
    
    file = request.files['file']
    email = request.form.get('email')
    interview_format = request.form.get('interview_format')
    rounds = request.form.get('rounds')
    duration = request.form.get('duration')
    if duration=='Other':
        duration=request.form.get('other_duration')
    difficulty_level = request.form.get('difficulty_level')
    money_requested = request.form.get('money_requested')
    paid_course = request.form.get('paid_course')

    form_data = {
    "interview_format": interview_format,
    "rounds": rounds,
    "duration": duration,
    "difficulty_level": difficulty_level,
    "money_requested": money_requested,
    "paid_course": paid_course
    }

    # Basic email format validation
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return redirect(request.url)

    domain = email.split('@')[1]
    startDate = check_domain_start_date(domain)
    mx_record_info = check_mx_records(domain)

    if file.filename == '' or not file.filename.endswith('.pdf'):
        return redirect(request.url)

    pdf_content = file.read()
    pdf_text = extract_text_from_pdf(pdf_content)

    # Pass the extracted text for grammar checking
    #########MElvin complete the function whihc u already done for checking spelling and grammar mistake
    #correct_percentage, mistake_percentage = check_gramma_mistak_per(pdf_text)
    #############return the value as  same above correct_p..., mistake_p..... dont implement the extra pie chart its already impleent just return value
    



    genuine_Accuracy=Predic_offerltr_genuiness(form_data)
    
#
    return render_template('res.html', text=pdf_text, startDate=startDate, mx_record_info=mx_record_info,correct_percentage=correct_percentage,mistake_percentage=mistake_percentage,genuine_Accuracy=genuine_Accuracy)

def extract_text_from_pdf(pdf_content):
    pdf_stream = io.BytesIO(pdf_content)
    reader = PdfReader(pdf_stream)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

def check_domain_start_date(domain):
    try:
        w = whois.whois(domain)
        return w.creation_date
    except Exception:
        return 'No Record present'

def check_mx_records(domain):
    try:
        mx_records = dns.resolver.resolve(domain, 'MX')
        mx_hosts = [str(r.exchange) for r in mx_records]
        known_providers = ['google.com', 'yahoo.com', 'outlook.com', 'microsoft.com']
        for mx in mx_hosts:
            if any(provider in mx for provider in known_providers):
                return f"The domain {domain} uses a reputable email provider: {mx}"
        return f"The domain {domain} has MX records, but no known providers found."
    except dns.resolver.NoAnswer:
        return f"The domain {domain} does not have MX records (potentially suspicious)."
    except dns.resolver.NXDOMAIN:
        return f"The domain {domain} does not exist."
    except Exception as e:
        return f"Error checking MX records: {str(e)}"

# def preprocess_text(text):
#     text = re.sub(r'\s+', ' ', text)  # Remove extra spaces and newlines
#     return text.strip()



#MElVIN Section 

#####################
def batch_correct_sentences(sentences):
    corrected_sentences = []
    for sentence in sentences:
        corrected = corrector(sentence)
        corrected_sentences.append(corrected[0]['generated_text'])
    return corrected_sentences

def check_gramma_mistak_per(text):
    text="We cleaned all the kitchen while our parents were out to dinner"
    
    print("Start processing...")
    
    # Preprocess the text to remove unnecessary spaces and format
    cleaned_text = preprocess_text(text)
    
    # Check for grammar mistakes using LanguageTool
    matches = tool.check(cleaned_text)  # Get grammar mistakes
    total_errors = len(matches)  # Count the number of errors
    
    # Tokenize the cleaned text to get the total word count
    total_words = len(word_tokenize(cleaned_text))

    if total_words > 0:
        mistake_percentage = (total_errors / total_words) * 100
        correct_percentage = 100 - mistake_percentage
    else:
        correct_percentage = 100
        mistake_percentage = 0
    return correct_percentage, mistake_percentage



def preprocess_text(text):
    return re.sub(r'\s+', ' ', text).strip()  # Simplified preprocess function

def check_text(text):
    
    cleaned_text = preprocess_text(text)
    print("Original Text:", text)
    
    # Using TextBlob for spelling correction
    blob = TextBlob(cleaned_text)
    corrected_text = str(blob.correct())

    # Check for grammar mistakes using LanguageTool
    matches = tool.check(corrected_text)  # Get grammar mistakes
    total_errors = len(matches)  # Count the number of errors

    # Tokenize the cleaned text to get the total word count
    total_words = len(corrected_text.split())

    if total_words > 0:
        mistake_percentage = (total_errors / total_words) * 100
        correct_percentage = 100 - mistake_percentage
    else:
        correct_percentage = 100
        mistake_percentage = 0

    # Identify mistakes and their context
    mistakes_info = []
    for match in matches:
        mistake_info = {
            'error': match.context,           # Context of the mistake
            'message': match.message,         # Description of the error
            'suggestions': match.replacements  # Suggested corrections
        }
        mistakes_info.append(mistake_info)

    return correct_percentage, mistake_percentage, mistakes_info



#END MELVING SECTION ############################################


def Predic_offerltr_genuiness(formData):
    print(formData)
    interview_format = formData["interview_format"]
    rounds = formData["rounds"]
    duration = formData["duration"]
    difficulty_level = formData["difficulty_level"]
    money_requested = formData["money_requested"]
    paid_course = formData["paid_course"]

    interview_format = 1 if interview_format.strip().lower() == "in-person" else 0  # 0 for Video Call, 1 for In-Person
    money_requested = 1 if money_requested.strip().lower() == "yes" else 0
    paid_course = 1 if paid_course.strip().lower() == "yes" else 0
    difficulty_map = {'Low': 1, 'Medium': 2, 'High': 3}
    difficulty = difficulty_map.get(difficulty_level.strip().capitalize(), 0)  # Default to 0 if not found

    # Create a DataFrame for the user input
    user_input = pd.DataFrame({
        'Interview Format': [interview_format],
        'Rounds': [rounds],
        'Duration (mins)': [duration],
        'Difficulty Level': [difficulty],
        'Money Requested': [money_requested],
        'Paid Course': [paid_course]
    })

    data = pd.read_csv("fake_job.csv")

    # Preprocessing
    # Convert categorical columns to numerical
    data['Interview Format'] = data['Interview Format'].apply(lambda x: 1 if x == 'In-Person' else 0)
    data['Money Requested'] = data['Money Requested'].apply(lambda x: 1 if x == 'Yes' else 0)
    data['Paid Course'] = data['Paid Course'].apply(lambda x: 1 if x == 'Yes' else 0)
    data['Difficulty Level'] = data['Difficulty Level'].map(difficulty_map)  # Convert difficulty level

    # Define features (X) and target (y)
    X = data.drop(columns=['Genuineness (%)'])
    y = data['Genuineness (%)']

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling (optional but beneficial for some algorithms)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Preprocess the user input before making predictions
    user_input_scaled = scaler.transform(user_input)

    # Make a prediction for the user's input
    user_prediction = model.predict(user_input_scaled)[0]

    

    # Save the model for future use
    dump(model, 'job_offer_genuineness_model.joblib')
    return user_prediction

if __name__ == '__main__':
    app.run(debug=True)
