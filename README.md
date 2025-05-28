# Few-Shot-Learning-with-SetFit
# IMDB Sentiment Analysis with SetFit
This project performs sentiment analysis on the IMDB 50k movie reviews dataset using the SetFit classifier, a few-shot learning model. Trained on 200 samples, it achieves 72% test accuracy on 100 samples.
# Overview

Dataset: IMDB 50k (25,000 train, 25,000 test, balanced positive/negative).
Preprocessing: Clean text (lowercase, remove HTML/URLs), remove stop words, lemmatize.
Model: SetFit (paraphrase-MiniLM-L3-v2).
Results:
Test Accuracy: 72%
Training Accuracy: 98%
Confusion Matrix:
True Negatives (TN): 34
False Positives (FP): 15
False Negatives (FN): 13
True Positives (TP): 38





# Files

extranlp (3).ipynb: Notebook with preprocessing, training, and evaluation.
README.md: This file.

# Setup

Clone Repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Virtual Environment:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


Install Dependencies:
pip install numpy==1.23.5 pandas scikit-learn nltk setfit==1.0.3 sentence-transformers==2.2.2 torch==2.0.1 jax==0.4.13 jaxlib==0.4.13 transformers==4.34.1 seaborn matplotlib


NLTK Data:
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


Dataset: Download train.csv and test.csv from Kaggle and place in data/ or update paths in notebook.



# Usage
Predict sentiment on new text:
from setfit import SetFitClassifier
clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")
# Load or retrain model
text = "This movie was amazing!"
cleaned = clean_text(text)  # From notebook
normalized = normalize_text(cleaned)  # From notebook
pred = clf.predict([normalized])
print("Sentiment:", "Positive" if pred[0] == 1 else "Negative")




