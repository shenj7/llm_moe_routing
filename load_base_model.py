from transformers import pipeline

pipe = pipeline("text-generation", model="GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
messages = [
    {"role": "user", "content": "Who are you?"},
    {"role": "user", "content": "Does it smell like updog in here?"},
]
pipe(messages)
