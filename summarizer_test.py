from transformers import pipeline

# The text we want to summarize
text = """
Artificial intelligence is transforming industries across the world.
Companies are using machine learning to automate tasks, improve efficiency,
and create new products. One of the most exciting applications is the use
of natural language models to summarize large documents automatically.
"""

print("Loading summarizer model...")

summarizer = pipeline("summarization", model="t5-base")

summary = summarizer(text, max_length=60, min_length=20, do_sample=False)

print("\n===== SUMMARY =====\n")
print(summary[0]['summary_text'])
