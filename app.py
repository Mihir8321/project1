import streamlit as st
from gtts import gTTS
import base64
import os
import uuid
from huggingface_hub import InferenceClient

# Replace 'your_api_key_here' with your actual API key
API_KEY = "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"
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# Initialize the Inference Client
client = InferenceClient(api_key=API_KEY)

def generate_summary(user_input, max_length, min_length):
    # Prepare your input message
    messages = [
        {"role": "system", "content": f"You are a helpful assistant who creates summaries of {min_length} to {max_length} words, suitable for a podcast format. Do not mention the word podcast anywhere."},
        {"role": "user", "content": f"{user_input}"}
    ]

    # Make a request to the model
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=max_length,
    )

    summary = response['choices'][0]['message']['content']
    return summary

def text_to_speech(text, unique_id):
    """Convert text to speech using gTTS and save as audio file."""
    filename = f"summary_audio_{unique_id}.mp3"
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(filename)
    return filename

def get_audio_html(audio_file):
    """Generate an audio player HTML for the given file."""
    try:
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        return f'''
            <audio controls autoplay="false">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        '''
    except Exception as e:
        return None

def cleanup_audio_files():
    """Clean up old audio files to manage storage."""
    try:
        for file in os.listdir():
            if file.startswith("summary_audio_") and file.endswith(".mp3"):
                os.remove(file)
    except Exception as e:
        print(f"Error in cleanup: {str(e)}")

def main():
    st.title("Patient Details Summarizer with Voice")

    # Initialize session state to track summaries and audio files
    if 'summaries' not in st.session_state:
        st.session_state['summaries'] = []

    cleanup_audio_files()

    # Input and controls
    user_input = st.text_area(
        "Enter Patient Details",
        height=200,
        placeholder="Enter the patient details you want to summarize..."
    )
    max_length = st.slider("Maximum Length", 100, 500, 200)
    min_length = st.slider("Minimum Length", 50, 400, 100)

    if st.button("Summarize and Generate Voice"):
        if user_input:
            try:
                with st.spinner("Generating summary..."):
                    summary = generate_summary(user_input, max_length, min_length)

                with st.spinner("Converting to speech..."):
                    unique_id = str(uuid.uuid4())
                    audio_file = text_to_speech(summary, unique_id)

                    # Track the summary and audio file
                    st.session_state.summaries.append({
                        'summary': summary,
                        'audio_file': audio_file
                    })

                    # Display the summary and audio
                    st.subheader("Summary:")
                    st.write(summary)
                    st.subheader("Audio Summary:")
                    audio_html = get_audio_html(audio_file)
                    if audio_html:
                        st.markdown(audio_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter some text to summarize.")

    # Display previous summaries and audio
    st.subheader("Previous Summaries:")
    for item in st.session_state.summaries:
        st.write(item['summary'])
        st.write("*"*200)
        

if __name__ == "__main__":
    main()
