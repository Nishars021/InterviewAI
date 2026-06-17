# Lazy initialization of heavy ML models to avoid import-time errors
whisper_model = None
grammar_corrector = None


def ensure_models():
    """Import and initialize Whisper and grammar correction on first use."""
    global whisper_model, grammar_corrector
    if whisper_model is not None and grammar_corrector is not None:
        return

    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError:
        raise

    try:
        from transformers import pipeline as hf_pipeline
    except ModuleNotFoundError:
        raise

    # Initialize WhisperModel (tiny for fastest CPU transcription)
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

    # Grammar correction only (skip slow semantic rewriter)
    try:
        grammar_corrector = hf_pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")
    except Exception as e:
        print(f"Warning: Grammar correction unavailable: {e}")
        grammar_corrector = None

# --- Remove repeated phrases (Whisper hallucination) ---
def collapse_repeats(text):
    """Remove consecutive duplicate words and repeated sentence patterns."""
    words = text.split()
    cleaned = []
    
    # Remove consecutive duplicate words
    for w in words:
        if not cleaned or cleaned[-1] != w:
            cleaned.append(w)
    
    # Remove repeated 3-word phrases (Whisper hallucination)
    result = cleaned[:]
    i = 0
    while i < len(result) - 6:
        phrase = result[i:i+3]
        if phrase == result[i+3:i+6]:
            result = result[:i+3] + result[i+6:]
        else:
            i += 1
    
    return " ".join(result)

# --- Audio transcription pipeline ---
def process_audio(audio_file):
    print("Transcribing...")
    ensure_models()
    segments, _ = whisper_model.transcribe(audio_file)
    raw_transcript = " ".join([seg.text for seg in segments])

    if not raw_transcript.strip():
        return "[No speech detected]"

    # Remove hallucinations and repeats
    cleaned = collapse_repeats(raw_transcript.strip())
    
    # Optional grammar correction (comment out if too slow)
    if grammar_corrector:
        try:
            result = grammar_corrector(cleaned, max_length=256)
            cleaned = result[0]["generated_text"] if result else cleaned
        except Exception as e:
            print(f"Grammar correction failed: {e}")
    
    return cleaned

# --- Typed fallback ---
def text_fallback(user_text):
    ensure_models()
    cleaned = collapse_repeats(user_text.strip())
    
    if grammar_corrector:
        try:
            result = grammar_corrector(cleaned, max_length=256)
            cleaned = result[0]["generated_text"] if result else cleaned
        except Exception as e:
            print(f"Grammar correction failed: {e}")
    
    return cleaned
