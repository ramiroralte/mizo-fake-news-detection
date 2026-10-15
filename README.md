# Mizo Fake News Detection Pipeline

An end-to-end Python pipeline built to scrape, clean, synthetically augment, and classify news articles in Mizo—a low-resource language with limited labeled NLP datasets.

## What This Project Does

Finding quality labeled datasets for fake news detection in low-resource languages like Mizo is a major hurdle. To tackle this, this project builds a complete workflow that handles everything from data collection to model evaluation:

1. **Web Scraping & Cleaning:** Collects authentic Mizo articles from *Vanglaini* and strips site boilerplate (QR app download promotions, technical support footers, HTML tags).
2. **Synthetic Data Generation:** Generates synthetic negative examples from real articles using controlled text mutations—swapping politician names, inverting sentiment or negation, changing locations, and shuffling sentences.
3. **Group-Safe Data Splitting:** Uses `GroupShuffleSplit` on original article URLs to ensure real articles and their synthetic variants never leak across train, validation, and test sets.
4. **Model Benchmarking:** Evaluates classical TF-IDF baselines alongside fine-tuned multilingual Transformer models.

---

## Pipeline Overview

```text
Scrape Raw Articles (Vanglaini)
              |
              v
Strip Boilerplate & Clean Text
              |
              v
     Synthetic Perturbation
       (Synthetic Negatives)
              |
              v
   Group-Safe Train / Val / Test Split
              |
       +------+------+
       |             |
       v             v
  Baseline TF-IDF   Multilingual Transformers
       Models              Models
       |                    |
       v                    v
 Logistic Regression     mBERT
 Linear SVC              XLM-RoBERTa
 Random Forest           MizBERT
       |                    |
       +---------+----------+
                 |
                 v
        Comparative Evaluation
```

---

## How Synthetic Negatives Are Created

Because manual annotation for low-resource languages takes time, the dataset uses rule-based perturbations to turn original articles into synthetic negative examples.

- **Entity Swapping:** Replaces Mizo political figures with other politicians in the same domain.
- **Negation Insertion:** Alters meaning using Mizo negation phrases, such as changing `a ṭha` to `a ṭha lo`.
- **Location & Organization Swaps:** Switches target towns, districts, or organizations.
- **Outcome Reversal:** Swaps directional words, numbers, and key verbs to change sentence intent.
- **Sentence Shuffling & Injection:** Scrambles paragraph order or inserts sentences from other articles.

*Note: Synthetic perturbations do not guarantee that an article is factually false or that the resulting text is realistic misinformation.*

---

## Models Evaluated

### Classical Machine Learning

TF-IDF with unigram, bigram, and trigram features, paired with:

- Logistic Regression
- Linear Support Vector Classifier (Linear SVC)
- Random Forest

### Transformer Models

Fine-tuned pretrained multilingual models:

- `xlm-roberta-base` (XLM-RoBERTa)
- `bert-base-multilingual-cased` (mBERT)
- `robzchhangte/MizBERT` (MizBERT)

---

## How to Run

### Requirements

- Python 3.8 or higher
- GPU recommended for Transformer fine-tuning

### Setup

```bash
git clone https://github.com/ramiroralte/mizo-fake-news-detection.git
cd mizo-fake-news-detection
pip install -r requirements.txt
```



---

## Citation & Acknowledgments

This project uses MizBERT, a Mizo-oriented pretrained language model.

```bibtex
@article{lalramhluna2024mizbert,
  title     = {MizBERT: A Mizo BERT Model},
  author    = {Lalramhluna, Robert and Dash, Sandeep and Pakray, Partha},
  journal   = {ACM Transactions on Asian and Low-Resource Language Information Processing},
  year      = {2024},
  publisher = {ACM}
}
```


---

## Limitations

- Synthetic negative examples may contain grammatical errors or unrealistic claims.
- Models may learn perturbation patterns rather than general indicators of misinformation.
- Articles from a single publisher may not represent all Mizo-language news.
- Performance on synthetic examples does not necessarily generalize to real-world misinformation.

