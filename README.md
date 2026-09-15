# Spam Detector

A machine learning web app that classifies SMS/email messages as **Spam** or **Not Spam** using TF-IDF feature extraction and a Multinomial Naive Bayes classifier, served through a Streamlit interface.

**Live demo:** https://spam-detector-machine-learning.streamlit.app/

## Overview

Paste any message into the app and it returns a classification along with a confidence score, computed by a model trained on the classic SMS Spam Collection dataset.

## Tech Stack

- **Python**
- **Streamlit** — web app framework
- **scikit-learn** — TF-IDF vectorization, Multinomial Naive Bayes
- **pandas** — data loading and preprocessing

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 real, labeled SMS messages (4,825 ham / 747 spam), tab-separated with `label` and `message` columns.

## Model Pipeline

1. Load and clean the dataset, mapping labels to `ham → 0` / `spam → 1`
2. Split into train/test sets (80/20, stratified, `random_state=42`)
3. Vectorize message text with **TF-IDF** (lowercased, English stop words removed)
4. Train a **Multinomial Naive Bayes** classifier on the TF-IDF features
5. Classify new input by transforming it with the fitted vectorizer and predicting with the trained model

## Performance

Evaluated on the held-out 20% test split:

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 97.04% |
| Precision | 100%   |
| Recall    | 77.85% |
| F1-Score  | 87.55% |

## Project Structure

```
.
├── app.py               # Streamlit app: UI + model training + inference
├── SMSSpamCollection     # Training dataset (tab-separated, label + message)
├── requirements.txt      # Python dependencies
└── README.md
```

## How It Works

The model trains once per session (cached via `@st.cache_resource`) directly inside the app — there's no separate training script or saved model file. On each classification request, the input message is transformed using the same fitted TF-IDF vectorizer and passed to the trained Naive Bayes model, which returns a predicted class and its associated probability.

## License

This project is for educational purposes.
