from transformers import pipeline

# Use a working NER model
ner_model = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",   # ✅ valid model
    aggregation_strategy="simple"
)

# Sample text
text = """
Alice Corporation, a Delaware corporation with offices at 123 Market Street,
signed a service agreement with Bob Industries Pvt. Ltd. on March 12, 2023.
"""

# Run NER
entities = ner_model(text)

# Print results
print(entities)
