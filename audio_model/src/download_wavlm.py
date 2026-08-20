from transformers import AutoProcessor, AutoModel

MODEL_NAME = "microsoft/wavlm-base-plus"

print("Downloading WavLM...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

print("WavLM downloaded successfully!")
print("Hidden size:", model.config.hidden_size)