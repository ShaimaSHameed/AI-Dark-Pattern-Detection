import os

print('1. Starting imports...')
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
print('2. FastAPI imported OK')

from pydantic import BaseModel
from typing import List
import joblib
import torch
import numpy as np
print('3. Basic imports OK')

from transformers import AlbertTokenizer, AlbertForSequenceClassification
print('4. Transformers imported OK')

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
print('5. Sentiment imports OK')

app = FastAPI(title='Test')
print('6. FastAPI app created OK')

SAVE_DIR = 'hybrid_albert_model'
print('7. About to load ALBERT model...')

print('8. Does folder exist?', os.path.exists(SAVE_DIR))

if os.path.exists(SAVE_DIR):
    print('9. Folder contents:', os.listdir(SAVE_DIR))
else:
    print('9. FOLDER NOT FOUND!')

print('10. Trying to load ALBERT model...')
try:
    MODEL_NAME = f"{SAVE_DIR}/albert_finetuned"
    ALBERT = AlbertForSequenceClassification.from_pretrained(MODEL_NAME)
    print('11. ALBERT loaded OK!')
except Exception as e:
    print('11. ALBERT FAILED:', e)

print('12. Trying to load tokenizer...')
try:
    tokenizer = AlbertTokenizer.from_pretrained(f"{SAVE_DIR}/tokenizer")
    print('13. Tokenizer loaded OK!')
except Exception as e:
    print('13. Tokenizer FAILED:', e)

print('14. Done!')