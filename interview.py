import json, random
import pyttsx3
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pipeline import process_audio, text_fallback

# --- Collapse repeated words ---
def collapse_repeats(text):
    words = text.split()
    cleaned = []
    for w in words:
        if not cleaned or cleaned[-1] != w:
            cleaned.append(w)
    return " ".join(cleaned)

# --- Mic capture with silence detection ---
def record_until_silence(filename="mic_input.wav", samplerate=16000,silence_threshold=0.01, silence_duration=2):
    print("Recording... speak now!")
    buffer = []
    silence_counter = 0
    blocksize = 1024

    with sd.InputStream(samplerate=samplerate, channels=1, blocksize=blocksize) as stream:
        while True:
            audio_block, _ = stream.read(blocksize)
            buffer.append(audio_block)

            if np.abs(audio_block).mean() < silence_threshold:
                silence_counter += blocksize / samplerate
            else:
                silence_counter = 0

            if silence_counter >= silence_duration:
                break

    audio = np.concatenate(buffer, axis=0)

    if len(audio) < samplerate * 0.5:  # less than 0.5 seconds
        print("No speech detected.")
        return None

    sf.write(filename, audio, samplerate)
    print("Recording stopped.")
    return filename

# --- TTS helper ---
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# --- User response ---
def get_user_response():
    choice = input("Type your answer, 'mic' to record, or 'exit': ")
    if choice.lower() == "exit":
        return None
    if choice.lower() == "mic":
        filename = record_until_silence()
        if filename is None:   # silence case
            print("Mic was silent, switching to text fallback...")
            typed = input("Type your answer instead: ")
            return text_fallback(typed)

        transcript = process_audio(filename)
        transcript = collapse_repeats(transcript)

        # ✅ Give user option to edit
        print("Transcribed:", transcript)
        edit_choice = input("Press Enter to accept, or type a corrected version: ")
        if edit_choice.strip():
            return text_fallback(edit_choice)  # run cleanup on edited text
        return transcript

    # Typed fallback
    return text_fallback(choice)
# --- Load questions ---
with open("questions.json", "r") as f:
    questions = json.load(f)

# --- Feedback logic ---
def handle_response(transcript, score, pause_duration=0):
    transcript = transcript.lower().strip()
    words = transcript.split()
    if pause_duration >= 10:
        return "You paused for a while, let’s move forward."
    if any(w in ["hi", "hello"] for w in words):
        return "Hi there, glad you’re here!"
    if any(w in transcript for w in ["bye", "thank you"]):
        return "Thanks for your time, goodbye!"
    if score >= 0.7:
        return "Excellent answer, very clear."
    elif score >= 0.5:
        return "Good effort, keep it up."
    elif score >= 0.3:
        return "That’s okay, let’s move on."
    else:
        return "No worries, let’s try another one."

# --- Interview loop ---


def format_duration(seconds):
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    return f"{minutes}m {sec}s"


def interview_loop():
    print("Interview simulation started. Type 'exit' to stop.")
    start_time = time.time()
    max_questions = 15
    max_duration = 4200  # 1 hour 10 minutes interview timer
    asked_questions = 0

    # Greeting phase
    choice = input("Say hi/hello to begin (or 'exit'): ")
    if choice.lower() == "exit":
        return
    feedback = handle_response(choice, score=0)
    print("System:", feedback)
    speak_text(feedback)

    # Question loop
    while True:
        # ✅ End after 1 hour or 15 questions
        elapsed = time.time() - start_time
        remaining = max_duration - elapsed
        print(f"Elapsed time: {format_duration(elapsed)} | Remaining: {format_duration(max(0, remaining))}")

        if elapsed >= max_duration:
            print("System: Interview time is over. Thank you!")
            speak_text("Interview time is over. Thank you!")
            break
        if asked_questions >= max_questions:
            print("System: You have completed the interview questions. Thank you!")
            speak_text("You have completed the interview questions. Thank you!")
            break

        q = random.choice(questions)
        speak_text(q["question"])
        print("System:", q["question"])
        asked_questions += 1

        transcript = get_user_response()
        if transcript is None or transcript == "[No speech detected]":
            print("System: No speech detected, let’s try another one.")
            continue

        print("You said:", transcript)

        # ✅ Exit early if user says bye/thank you
        if any(w in transcript.lower() for w in ["bye", "thank you"]):
            speak_text("Interview ended, thank you for participating.")
            print("System: Interview ended, thank you for participating.")
            break

        # Evaluate similarity
        expected_answer = q["expected_answer"]
        documents = [expected_answer, transcript]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
        score = similarity[0][0]

        # Feedback
        feedback = handle_response(transcript, score)
        print("System:", feedback)
        speak_text(feedback)

# Run
if __name__ == "__main__":
    interview_loop()
