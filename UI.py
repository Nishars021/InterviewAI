import json
import os
import random
import time

import streamlit as st
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Interview Simulator", page_icon="🎤", layout="wide")


# Defer importing the heavy `pipeline` module until it's actually needed
pipeline = None


def load_dataset_questions():
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "questions.json")
    if not os.path.exists(dataset_path):
        dataset_path = "datasets/questions.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


DATASET_QUESTIONS = load_dataset_questions()
ANSWER_MAP = {
    item["question"].strip(): item["expected_answer"].strip()
    for item in DATASET_QUESTIONS
}

# HR Questions at the beginning
hr_start = [
    "Tell me about yourself.",
    "What are you good at and interested in?"
]

# Technical Question Bank
technical_questions = [
    "What is DBMS and why is it used?",
    "What is the difference between DBMS and RDBMS?",
    "What are Primary Key and Foreign Key?",
    "Explain database normalization and its benefits.",
    "What are the different types of SQL commands?",
    "What is a transaction in DBMS?",
    "Explain the ACID properties in DBMS.",
    "What is indexing and why is it important?",
    "What is the difference between DELETE, TRUNCATE, and DROP?",
    "Explain the concept of joins in SQL.",
    "What is the difference between an array and a linked list?",
    "Explain time complexity and space complexity.",
    "What are stacks and queues? How do they differ?",
    "Explain the working of Binary Search.",
    "What is recursion? Give an example.",
    "What is the difference between BFS and DFS?",
    "What is a hash table and how does hashing work?",
    "What is dynamic programming?",
    "What is a Binary Search Tree (BST)?",
    "What is Machine Learning?",
    "What is Artificial Intelligence?",
    "What is supervised and unsupervised learning?",
    "What is overfitting and underfitting?",
    "What is feature engineering?",
    "What is Computer Vision?"
]

# HR Questions at the end
hr_end = [
    "What are your strengths?",
    "Why should we hire you?",
    "Where do you see yourself in 5 years?"
]

# Create interview questions ONLY ONCE
if "questions" not in st.session_state:

    technical_pool = [
        item["question"].strip()
        for item in DATASET_QUESTIONS
        if item["question"].strip() not in hr_start + hr_end
    ]

    selected_technical = random.sample(technical_pool, 5)

    # Final interview order
    st.session_state.questions = (
        hr_start +
        selected_technical +
        hr_end
    )

# Use this list throughout the interview
questions = st.session_state.questions


def tts_html(text):
    return f"""
    <script>
    const text = {json.dumps(text)};
    if ('speechSynthesis' in window) {{
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        utterance.rate = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
    }}
    </script>
    """


def embed_html(html: str, height: int = 1):
    """Embed HTML/JS in the app with best-effort compatibility.

    Tries `st.iframe(srcdoc=...)`, falls back to older `st.iframe(html, ...)`,
    then to `st.components.v1.html(...)`, and finally to a data: URI iframe.
    """
    try:
        # Preferred new API (some Streamlit versions accept srcdoc)
        st.iframe(srcdoc=html, height=height)
        return
    except TypeError:
        pass

    try:
        # Some versions accept the HTML as the first positional arg
        st.iframe(html, height=height)
        return
    except Exception:
        pass

    try:
        # Fallback to legacy components API which runs scripts reliably
        st.components.v1.html(html, height=height)
        return
    except Exception:
        pass

    # Last-resort: use data URL with iframe src
    try:
        data = "data:text/html;charset=utf-8," + urllib.parse.quote(html)
        st.iframe(data, height=height)
    except Exception:
        # If everything fails, show the HTML as plain text for debugging
        st.text("[embed failed]" + html[:200])


WELCOME_FEEDBACK = "Hi there, glad you’re here!"
END_FEEDBACK = "Interview ended, thank you for participating."

def handle_response(transcript, score, pause_duration=0):
    transcript = transcript.lower().strip()
    words = transcript.split()
    if pause_duration >= 10:
        return "You paused for a while, let’s move forward."
    if any(w in ["hi", "hello"] for w in words):
        return WELCOME_FEEDBACK
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


def get_expected_answer(question):
    return ANSWER_MAP.get(question.strip())


def evaluate_answer(answer, question):
    # For HR questions we don't provide automated scoring/feedback because
    # answers are unique to each candidate. Only evaluate technical questions.
    if question.strip() in hr_start + hr_end:
        return 0.0, ""

    expected_answer = get_expected_answer(question)
    if expected_answer:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([expected_answer, answer])
        score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    else:
        score = 0.0

    feedback = handle_response(answer, score)
    return score, feedback


def get_summary_metrics():
    total_score = sum(item.get("score", 0.0) for item in st.session_state.history)
    question_count = len(st.session_state.history)
    average_score = total_score / question_count if question_count else 0.0
    skipped = sum(1 for item in st.session_state.history if item.get("score", 0.0) < 0)
    return total_score, average_score, skipped


MAX_INTERVIEW_DURATION = 4200  # 1 hour 10 minutes
SKIP_PENALTY = -0.25
SKIP_FEEDBACK = "You skipped this question. A penalty has been applied to your final score."


def format_duration(seconds):
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    return f"{minutes}m {sec}s"

# Best-effort browser focus detection
embed_html("""
<script>
if (!localStorage.getItem("interview_status")) {
    localStorage.setItem("interview_status","active");
}

document.addEventListener("visibilitychange", function() {
    if (document.hidden) {
        localStorage.setItem("interview_status","terminated");
    }
});
window.addEventListener("blur", function() {
    localStorage.setItem("interview_status","terminated");
});
</script>
""", height=1)

if "started" not in st.session_state:
    st.session_state.started = False
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "answer_key" not in st.session_state:
    st.session_state.answer_key = 0
if "name" not in st.session_state:
    st.session_state.name = ""
if "enrollment" not in st.session_state:
    st.session_state.enrollment = ""
if "start_error" not in st.session_state:
    st.session_state.start_error = ""
if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = ""
if "last_feedback_spoken" not in st.session_state:
    st.session_state.last_feedback_spoken = False
if "question_spoken_for" not in st.session_state:
    st.session_state.question_spoken_for = ""
if "interview_start_time" not in st.session_state:
    st.session_state.interview_start_time = None
if "interview_timed_out" not in st.session_state:
    st.session_state.interview_timed_out = False
if "end_feedback_shown" not in st.session_state:
    st.session_state.end_feedback_shown = False

status = st.query_params.get("status", "")

title = """
<h1 style='text-align:center;color:#1E40AF;'>🎤 AI Interview Simulator</h1>
<p style='text-align:center;'>Practice • Improve • Succeed</p>
"""
st.markdown(title, unsafe_allow_html=True)

# Debug: try lazy import of the pipeline module (may require heavy deps)
try:
    if pipeline is None:
        import pipeline as _pipeline
        pipeline = _pipeline
    st.write("Using pipeline module:", pipeline.__file__)
except Exception as e:
    st.write("Pipeline not available (deferred):", e)

if "start_error" not in st.session_state:
    st.session_state.start_error = ""

if not st.session_state.started:

    name = st.text_input("Full Name", key="name")
    enrollment = st.text_input("Enrollment Number", key="enrollment")

    def start_interview():
        if st.session_state.name and st.session_state.enrollment:
            st.session_state.started = True
            st.session_state.start_error = ""
            st.session_state.interview_start_time = time.time()
            st.session_state.interview_timed_out = False
            st.session_state.last_feedback = WELCOME_FEEDBACK
            st.session_state.last_feedback_spoken = False
            st.session_state.end_feedback_shown = False
            st.session_state.question_spoken_for = ""
        else:
            st.session_state.start_error = "Enter all details"

    st.button("🚀 Start Interview", key="start_interview", on_click=start_interview)

    if st.session_state.start_error:
        st.warning(st.session_state.start_error)

else:

    st.sidebar.write("👤", st.session_state.get("name", ""))
    st.sidebar.write(st.session_state.get("enrollment", ""))

    if st.session_state.interview_start_time:
        elapsed = time.time() - st.session_state.interview_start_time
        remaining = max(0, MAX_INTERVIEW_DURATION - elapsed)
        st.sidebar.write(f"Elapsed: {format_duration(elapsed)}")
        st.sidebar.write(f"Remaining: {format_duration(remaining)}")
        if remaining <= 0:
            st.sidebar.error("Interview time is over.")
            st.session_state.interview_timed_out = True

    if st.session_state.last_feedback and not st.session_state.last_feedback_spoken:
        st.info(st.session_state.last_feedback)
        embed_html(tts_html(st.session_state.last_feedback), height=1)
        st.session_state.last_feedback_spoken = True

    if (st.session_state.current_question >= len(questions) or st.session_state.interview_timed_out) and not st.session_state.end_feedback_shown:
        st.info(END_FEEDBACK)
        embed_html(tts_html(END_FEEDBACK), height=1)
        st.session_state.end_feedback_shown = True

    progress = st.session_state.current_question / len(questions)
    st.sidebar.progress(progress)

    embed_html("""
    <script>
    const s = localStorage.getItem("interview_status");
    if (s === "terminated") {
        window.parent.postMessage({type:"terminated"}, "*");
    }
    </script>
    """, height=1)

    if st.session_state.interview_timed_out:
        st.error("Interview ended because the 70-minute time limit has been reached.")
        st.write("Review your answers below:")
    elif st.session_state.current_question >= len(questions):

        st.success("Interview Completed")

    if st.session_state.current_question >= len(questions) or st.session_state.interview_timed_out:
        total_score, average_score, skipped = get_summary_metrics()
        st.write("### Final Score")
        st.write(f"Total score: {total_score:.2f}")
        st.write(f"Average score: {average_score:.2f}")
        st.write(f"Skipped questions: {skipped}")
        st.write("---")
        for i, item in enumerate(st.session_state.history, start=1):
            st.write(f"### Question {i}")
            st.write(item["question"])
            st.write(item["answer"])
            if item.get("feedback"):
                st.write(f"**Feedback:** {item['feedback']}")
            st.write(f"**Score:** {item.get('score', 0.0):.2f}")

    else:

        if st.session_state.interview_timed_out:
            st.error("Time is up. The interview has ended.")
            st.stop()

        q = questions[st.session_state.current_question]

        st.subheader(f"Question {st.session_state.current_question+1}")
        st.info(q)

        if st.session_state.question_spoken_for != q:
            embed_html(tts_html(q), height=1)
            st.session_state.question_spoken_for = q

        # Use browser microphone input (Streamlit's audio_input)
        transcription = None
        audio = st.audio_input("🎤 Click here to record")

        if audio is not None:
            with open("candidate_answer.wav", "wb") as f:
                f.write(audio.getbuffer())
            
            # Show transcribing status
            status_placeholder = st.empty()
            status_placeholder.info("⏳ Transcribing...")
            
            try:
                import pipeline as pipeline_module
                transcription = pipeline_module.process_audio("candidate_answer.wav")
                status_placeholder.success("✅ Transcription complete")
            except ModuleNotFoundError:
                status_placeholder.error("Transcription unavailable: missing pipeline dependencies (torch/transformers).")
            except Exception as e:
                status_placeholder.error(f"Transcription failed: {e}")

        key_name = f"answer_{st.session_state.answer_key}"

        if transcription is not None:
            st.session_state[key_name] = transcription
        elif key_name not in st.session_state:
            st.session_state[key_name] = ""

        answer = st.text_area(
            "Your Answer",
            key=key_name,
            height=200
        )

        next_col, skip_col = st.columns([3, 1])
        with next_col:
            if st.button("➡️ Next Question", key=f"next_{st.session_state.current_question}"):
                answer_text = st.session_state.get(key_name, "")

                if answer_text.strip():
                    score, feedback = evaluate_answer(answer_text, q)
                    st.session_state.history.append({
                        "question": q,
                        "answer": answer_text,
                        "score": score,
                        "feedback": feedback
                    })
                    st.session_state.last_feedback = feedback
                    st.session_state.last_feedback_spoken = False
                    st.session_state.current_question += 1

                    # Completely clears textbox by creating new widget key
                    st.session_state.answer_key += 1
                    st.session_state.question_spoken_for = ""

                    st.rerun()
                else:
                    st.warning("Enter an answer first.")

        with skip_col:
            if st.button("⏭️ Skip", key=f"skip_{st.session_state.current_question}"):
                st.session_state.history.append({
                    "question": q,
                    "answer": "Skipped",
                    "score": SKIP_PENALTY,
                    "feedback": SKIP_FEEDBACK
                })
                st.session_state.last_feedback = SKIP_FEEDBACK
                st.session_state.last_feedback_spoken = False
                st.session_state.current_question += 1
                st.session_state.answer_key += 1
                st.session_state.question_spoken_for = ""

                embed_html(tts_html(SKIP_FEEDBACK), height=1)
                st.rerun()
