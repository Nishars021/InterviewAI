# 🎤 AI Mock Interview System

## 📌 Overview

AI Mock Interview System is an AI-powered interview preparation platform that helps users practice both HR and Technical interviews through voice-based interaction.

The system records user responses, converts speech into text using Whisper Speech Recognition, evaluates technical answers using TF-IDF and Cosine Similarity, and generates performance feedback.

---

## 🚀 Features

### HR Interview Module

* Tell me about yourself
* What are your strengths?
* What are your weaknesses?
* Why should we hire you?
* Where do you see yourself in 5 years?

### Technical Interview Module

* Random technical question generation
* No repeated questions
* Questions selected from a custom dataset

### Voice Processing

* Audio recording support
* Speech-to-Text conversion using Faster-Whisper
* Multiple audio format support

### Answer Evaluation

* TF-IDF Vectorization
* Cosine Similarity Analysis
* Relevance Score Generation
* Keyword Matching
* Word Frequency Analysis

### Feedback System

* Similarity Score
* Relevance Score
* Performance Feedback
* Interview Summary Report

---

## 🏗️ System Workflow

Candidate Details ->
HR Questions ->
Voice Recording ->
Speech-to-Text Conversion ->
Technical Questions ->
TF-IDF Evaluation ->
Cosine Similarity Calculation ->
Relevance Scoring ->
Feedback Generation ->
Final Interview Report

---

## 🛠️ Technologies Used

### Frontend

* Streamlit

### Backend

* Python

### Libraries

* Faster Whisper
* Scikit-Learn
* Streamlit
* JSON
* Collections

### NLP Techniques

* Speech-to-Text
* TF-IDF
* Cosine Similarity
* Keyword Analysis
* Word Frequency Analysis

---

## 📂 Project Structure

```text
InterviewAI/
│
├── app.py
├── questions.json
├── pipeline.py
├── requirements.txt
├── README.md
│
└── assets/
```

---

## 📊 Dataset

The dataset contains:

* Technical Questions
* Expected Answers
* Keywords

Topics Covered:

* DBMS
* SQL
* Data Structures
* Algorithms
* Machine Learning
* Artificial Intelligence

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/InterviewAI.git
cd InterviewAI
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📸 Screenshots

Add screenshots here:

* Home Page
* Interview Screen
* Audio Recording
* Transcript Output
* Evaluation Results
* Final Report

---

## 👥 Team Members

### Nisha R S

* Project Lead
* System Architecture
* Dataset Design
* TF-IDF Evaluation

### Jahnavi Krishnan

* Audio Pipeline
* Speech-to-Text Processing
* Audio Handling
* * Integration

### Varji P

* UI/UX Design
* Streamlit Frontend
* Interview Flow Management

---

## 🔮 Future Enhancements

* AI Follow-Up Questions
* Confidence Analysis
* Emotion Detection
* Resume-Based Interviews
* Cloud Deployment
* LLM-Based Evaluation

---

## 🎯 Conclusion

The AI Mock Interview System provides an interactive platform for interview preparation by combining speech recognition, NLP, and automated evaluation techniques. The system helps users improve their communication and technical interview skills through continuous practice and feedback.

---

⭐ If you like this project, consider giving it a star!
