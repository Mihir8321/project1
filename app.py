import streamlit as st
from transformers import pipeline
import pyttsx3
import base64
import os
import time

def text_to_speech(text):
    """Convert text to speech and save as audio file"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)    # Speaking rate
        engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        timestamp = str(int(time.time()))
        filename = f"summary_audio_{timestamp}.mp3"
        
        engine.save_to_file(text, filename)
        engine.runAndWait()
        
        return filename
    except Exception as e:
        st.error(f"Error in text to speech conversion: {str(e)}")
        return None

def get_audio_html(audio_file):
    """Create audio HTML with base64 encoded audio data"""
    try:
        if audio_file and os.path.exists(audio_file):
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
            audio_html = f'''
                <audio controls autoplay="false">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
            '''
            return audio_html
    except Exception as e:
        st.error(f"Error creating audio player: {str(e)}")
    return None

def cleanup_audio_files():
    """Clean up old audio files"""
    try:
        current_time = time.time()
        for file in os.listdir():
            if file.startswith("summary_audio_") and file.endswith(".mp3"):
                file_timestamp = int(file.split("_")[2].split(".")[0])
                if current_time - file_timestamp > 300:
                    os.remove(file)
    except Exception as e:
        print(f"Error in cleanup: {str(e)}")

def main():
    st.title("Patient Details Summarizer with Voice")
    
    @st.cache_resource
    def load_summarizer():
        return pipeline("summarization", model="facebook/bart-large-cnn")
    
    summarizer = load_summarizer()
    
    cleanup_audio_files()
    
    if 'last_audio_file' not in st.session_state:
        st.session_state['last_audio_file'] = None
    
    # Create tabs for user interface
    tab1, tab2 = st.tabs(["Summarize Patient Details", "Example Input"])
    
    with tab1:
        user_input = st.text_area(
            "Enter Patient Details",
            height=200,
            placeholder="Enter the patient details you want to summarize..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            max_length = st.slider("Maximum Length", 100, 500, 200)
        with col2:
            min_length = st.slider("Minimum Length", 50, 400, 100)
        
        if st.button("Summarize and Generate Voice"):
            if user_input:
                try:
                    with st.spinner("Generating summary..."):
                        summary = summarizer(
                            user_input,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=False
                        )
                        
                        st.subheader("Summary:")
                        summary_text = summary[0]['summary_text']
                        st.write(summary_text)
                        
                        with st.spinner("Converting to speech..."):
                            audio_file = text_to_speech(summary_text)
                            
                            if audio_file:
                                st.session_state['last_audio_file'] = audio_file
                                
                                st.subheader("Audio Summary:")
                                audio_html = get_audio_html(audio_file)
                                if audio_html:
                                    st.markdown(audio_html, unsafe_allow_html=True)
                                else:
                                    st.error("Failed to create audio player")
                            else:
                                st.error("Failed to generate audio file")
                                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please enter some text to summarize.")
    
    with tab2:
        # Example input section
        example_input = """
        {
            "PatientID": 202,
            "PatientName": "Jane Smith",
            "Age": 32,
            "Gender": "Female",
            "Symptoms": [
                "Severe headache",
                "Nausea",
                "Blurred vision"
            ],
            "MedicalHistory": [
                "Migraine"
            ],
            "CurrentMedications": [
                "Sumatriptan"
            ],
            "AppointmentDetails": {
                "AppointmentID": 2,
                "DoctorID": 102,
                "DoctorName": "Dr. Michael Brown",
                "AppointmentDate": "January 20, 2025, 3:00 PM",
                "Status": "Pending",
                "Purpose": "Evaluation of recurring migraines"
            },
            "Notes": {
                "PatientConcerns": [
                    "Increased frequency of migraines",
                    "Difficulty concentrating at work"
                ],
                "RecommendedActions": [
                    "Schedule an MRI scan",
                    "Discuss preventive treatment options"
                ]
            }
        }
        """
        st.subheader("Example Patient Input:")
        st.code(example_input)

if __name__ == "__main__":
    main()
