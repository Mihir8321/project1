from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
user_input = input("enter Patinets details")
ARTICLE = f""""
{user_input}
"""
summarization_text = summarizer(ARTICLE, max_length=350, min_length=300, do_sample=False)

import pyttsx3 # type: ignore
engine = pyttsx3.init()
engine.say(summarization_text)
engine.runAndWait()
