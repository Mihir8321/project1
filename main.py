from huggingface_hub import InferenceClient

# Replace 'your_api_key_here' with your actual API key
API_KEY = "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"
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# Initialize the Inference Client
client = InferenceClient(api_key=API_KEY)

def generate_summary(user_input, max_length, min_length):
    # Prepare your input message
    messages = [
        {"role": "system", "content": f"You are a helpful assistant who creates summaries of {min_length} to {max_length} words, suitable for a podcast format. , dont mention the word podcast anywhere."},
        {"role": "user", "content": f"{user_input}"}
    ]

    # Make a request to the model with token limits
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=max_length,  # Set maximum number of tokens
           # Set minimum number of tokens
    )

    # Extract and return the summary content
    summary = response['choices'][0]['message']['content']
    return summary

# Example patient information input
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

# Generate and print the summary
print(generate_summary(example_input, 200, 100))
