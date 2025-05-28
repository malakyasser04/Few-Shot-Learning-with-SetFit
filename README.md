# Sentiment Analysis of IMDB Movie Reviews with SetFit

This repository implements sentiment analysis on the IMDB 50k movie reviews dataset using the SetFit classifier, a few-shot learning model. The model is trained on 200 samples and achieves 72% test accuracy on 100 samples.

## Project Overview

The project performs binary sentiment classification (positive/negative) on IMDB movie reviews. Key steps include:
- **Dataset**: IMDB 50k dataset (25,000 train, 25,000 test samples, balanced).
- **Preprocessing**: Text cleaning (lowercase, remove HTML/URLs), stop word removal, lemmatization.
- **Model**: SetFit classifier (`paraphrase-MiniLM-L3-v2`).
- **Evaluation**: Test accuracy of 72%, training accuracy of 98%, with confusion matrix:
  - True Negatives (TN): 34
  - False Positives (FP): 15
  - False Negatives (FN): 13
  - True Positives (TP): 38

## Requirements

```bash
numpy==1.23.5
pandas
scikit-learn
nltk
setfit==1.0.3
sentence-transformers==2.2.2
torch==2.0.1
jax==0.4.13
jaxlib==0.4.13
transformers==4.34.1
seaborn
matplotlib
```

## Setup Instructions

1. **Clone Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install numpy==1.23.5 jax==0.4.13 jaxlib==0.4.13 transformers==4.34.1 sentence-transformers==2.2.2 setfit==1.0.3 torch==2.0.1 pandas scikit-learn nltk seaborn matplotlib
   ```

4. **Download NLTK Data**:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

5. **Dataset**: Download `train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-50k-movie-reviews-test-your-bert) and place in `data/` or update notebook paths.


## Results

- **Test Accuracy**: 72% (100 samples).
- **Training Accuracy**: 98% (200 samples).
- **Confusion Matrix**:
  - TN: 34 (correct negative predictions)
  - FP: 15 (negative predicted as positive)
  - FN: 13 (positive predicted as negative)
  - TP: 38 (correct positive predictions)

The model shows slight positive bias (more FP than FN) and potential overfitting due to high training accuracy.

## Usage

To predict sentiment on new text:
```python
from setfit import SetFitClassifier
clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")
# Load trained model or retrain
new_text = "This movie was amazing!"
cleaned = clean_text(new_text)  # From notebook
normalized = normalize_text(cleaned)  # From notebook
prediction = clf.predict([normalized])
print("Predicted sentiment:", "Positive" if prediction[0] == 1 else "Negative")
```


