# Legal Notice Classification, Sentiment Lab 68299

## Project Description
A multi-class text classifier that categorizes short legal notices into three classes:
Contract Dispute, Intellectual Property Claim, and Regulatory Compliance. Built for a
legal tech startup prototyping scenario, comparing Logistic Regression and Naive Bayes
across Bag-of-Words and TF-IDF feature representations.

## Setup Instructions
1. Clone the repo: `git clone https://github.com/javeriakhalid15/sentiment-lab-68299.git`
2. Create and activate a virtual environment:
   - `python -m venv venv`
   - `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Open `notebooks/sentiment_analysis.ipynb` in VS Code and select the venv kernel
5. Run all cells top to bottom

## Config File
`config.json` holds all hyperparameters and the random seed, read programmatically
in `src/evaluate.py` — no hardcoded values in the training code:
- `random_seed`: fixed seed (42) for reproducible train/test splits
- `test_size`: 0.2 (80/20 split)
- `max_features`: 5000 (vocabulary cap for both vectorizers)
- `model_1` / `model_2`: model names and hyperparameters for Logistic Regression and Naive Bayes

## Results Summary
| Model      | Features | Accuracy | F1 Macro | Train Time |
|------------|----------|----------|----------|------------|
| NaiveBayes | BoW      | 1.0      | 1.0      | 0.001s     |
| NaiveBayes | TFIDF    | 1.0      | 1.0      | 0.002s     |
| LogReg     | BoW      | 1.0      | 1.0      | 0.008s     |
| LogReg     | TFIDF    | 1.0      | 1.0      | 0.015s     |

All four configurations achieve perfect scores due to strong class-distinctive
vocabulary in the dataset (confirmed via 5-fold cross-validation, std = 0.0).
NaiveBayes is recommended for production due to lower training and inference time
at equivalent accuracy.
