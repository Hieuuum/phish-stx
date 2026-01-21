# PhishSTX

PhishSTX is a Streamlit app and supporting notebooks for detecting phishing and spam emails. It pairs classic TF-IDF text vectorization with custom keyword and rarity features, then feeds them through a scikit-learn classifier (trained artifacts are bundled in `spam_classifier_pipeline.pkl`).

## What it does
- Interactive Streamlit UI to paste email content and get a spam/ham prediction with confidence.
- Recreates the full training-time preprocessing: HTML stripping, URL/email removal, tokenization, lemmatization, TF-IDF, and engineered rarity/keyword features.
- Includes cleaning and training notebooks to reproduce the model pipeline and data preparation steps.

## Repo highlights
- app: [app.py](app.py) for the Streamlit front end and feature extraction logic.
- data: raw datasets in `DS/` and cleaned versions in `cleaned_DS/`; pipeline outputs in `pipline/` (CSV aggregates, notebooks). Note: the classifier artifacts load from `spam_classifier_pipeline.pkl` (expected in repo root).
- notebooks: data cleaning, combining, and model training notebooks in `notebooks/`.
- config: Python deps in [requirements.txt](requirements.txt); devcontainer for reproducible setup.

## Quickstart
1) Install dependencies:
```
pip install -r requirements.txt
```
2) Ensure `spam_classifier_pipeline.pkl` is present in the project root.
3) Run the app:
```
streamlit run app.py
```
4) Paste an email body in the UI to see the prediction and internal feature values.

## Notes
- NLTK resources (`punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`) are downloaded on first run and cached by Streamlit.
- If you retrain the model, keep the feature order consistent with `extract_features` in [app.py](app.py) to avoid shape or scaler mismatches.
