from transformers import pipeline

print("Loading quiz generator model...")

qa_gen = pipeline("text2text-generation", model="google/flan-t5-base")

summary_text = """
Artificial intelligence is transforming industries across the world.
Companies are using machine learning to automate tasks and improve efficiency.
One exciting application is natural language models that summarize documents automatically.
"""

prompt = f"Generate 5 simple multiple-choice questions from the following text:\n{summary_text}"

result = qa_gen(prompt, max_length=256)

print("\n===== GENERATED QUIZ =====\n")
print(result[0]["generated_text"])
