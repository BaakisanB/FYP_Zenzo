from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI

app = FastAPI()

MODEL_PATH = "my_finetuned_roberta_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "Model is running!"}