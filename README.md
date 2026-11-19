# 🔎 Multilingual Misinformation Detector

Paste a short social-media post in English. A fine-tuned multilingual transformer (DistilBERT) labels it as real or fake, shows a confidence score, and uses **SHAP** to highlight the words that influenced the result.

> This is a demo classifier. It learns patterns from its training data, so it can be wrong. It is not a fact-checker.

## 🖼️ Screenshot

![App screenshot](screenshot.png)

*An example prediction. See Results below for how the model performs on the test set.*

---

## 🧠 How It Works

```
COVID-19 social-media posts labelled real / fake (10,700 posts, Hugging Face)
→ Clean noisy text (links, @mentions, emojis, stretched words; Hindi characters kept)
→ Split 80 / 10 / 10 into train / validation / test
→ Baseline: TF-IDF + Naive Bayes
→ Fine-tune a multilingual transformer (DistilBERT-multilingual, or mBERT)
→ Compare both on a held-out test set
→ Streamlit app: label + confidence + SHAP word highlights
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Model | Hugging Face Transformers, PyTorch (`distilbert-base-multilingual-cased`, switchable to mBERT) |
| Baseline | scikit-learn (TF-IDF + Multinomial Naive Bayes) |
| Explainability | SHAP |
| Text cleaning | Python regular expressions (`text_cleaning.py`) |
| UI | Streamlit |

---

## 📊 Results

Measured on the 1,070 held-out test posts, which were not used for training or for choosing the best epoch.

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Naive Bayes | 0.916 | 0.915 |
| DistilBERT (multilingual) | 0.961 | 0.961 |

Per class (transformer): Real precision 0.97 / recall 0.96, Fake precision 0.95 / recall 0.96.

Dataset used: COVID-19 fake news dataset from the Constraint@AAAI 2021 shared task (Patwa et al.), loaded from Hugging Face as `nanyy1025/covid_fake_news`. It has 10,700 English social-media posts (5,600 real, 5,100 fake), split 8,560 / 1,070 / 1,070 for train / validation / test. Results come from a single training run.

---

## 🗂️ Data

The training notebook downloads the dataset automatically from Hugging Face, merges its train / validation / test parts, and turns the labels into `real = 0` and `fake = 1`. It can also use any CSV with one text column and one label column: edit `TEXT_COL`, `LABEL_COL` and `LABEL_NAMES` in the settings cell.

The model was trained on English COVID-19 posts only. The transformer itself is multilingual and the cleaning keeps Hindi characters, but Hindi and Hinglish were not trained on or tested.

---

## 🏋️ Train

1. Open `Misinformation_Detector_Training.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Run the cells in order. The dataset is downloaded automatically (no upload needed).
4. At the end it saves a `fake_news_model` folder and downloads it as a zip.

---

## 🏃 Run the App

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Unzip the downloaded model so you have a `fake_news_model` folder next to `app.py`, then:

```bash
python -m streamlit run app.py
```

To load the model from somewhere else, such as a Hugging Face Hub repo, set `MODEL_PATH` first:

```bash
set MODEL_PATH=your-username/your-model-name        # Windows
export MODEL_PATH=your-username/your-model-name     # Mac/Linux
```

The trained model is hundreds of MB, so it is not stored in this repo. It is hosted on Hugging Face: [Shaheen31/misinformation-detector](https://huggingface.co/Shaheen31/misinformation-detector). To run the app with it, set `MODEL_PATH` to `Shaheen31/misinformation-detector` as shown above.

---

## 🔍 Error Analysis

The transformer got 42 of 1,070 test posts wrong (3.9%). Reviewing 20 of them showed some patterns (a small sample, so these are observations, not proof):

- Short, headline-style real posts and real posts about studies or official advisories were sometimes called fake, for example "Advisory on Strategy for COVID-19 Testing in India (Version VI)".
- Fake posts written in a neutral, news-like or fact-check tone, including debunking statements such as "The Google API does not surreptitiously install COVIDSafe", were sometimes called real.
- Of the 20 reviewed, 11 were real posts called fake and 9 were fake posts called real, so the errors were not tilted toward one class.
- The model can be very confident when it is wrong (99.9% on one real post).

The model only sees the words, not the facts, so it cannot check whether a claim is actually true.

---

## ⚠️ Limitations

- Trained on English COVID-19 posts only. Results do not carry over to other topics, platforms, or to Hindi and Hinglish, which were not tested.
- Labels in public datasets reflect the source and its annotators, so the model can learn style and topic rather than truth. A score of about 96% on this dataset does not mean 96% on any news.
- Results come from one training run on one split (1,070 test posts), so differences of about a point are within noise.
- In a few quick manual tests, plain news-style posts written outside the dataset were also labelled Fake with high confidence, so the model works best on text that looks like its training data (COVID-19 tweets).
- Results on one dataset do not carry over to other platforms or topics.
- SHAP explanations are estimates of influence, not proof of why the model decided.
- Free hosting has limited memory. A transformer plus SHAP may not fit, so running locally is the reliable option.

---

## 📁 Project Structure

```
├── Misinformation_Detector_Training.ipynb   # Colab training notebook
├── app.py                                # Streamlit app
├── text_cleaning.py                      # Shared text cleaning
├── screenshot.png                        # App screenshot used in this README
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 👤 Author

**Shaheen Akram**
- GitHub: [@Shaheen3107](https://github.com/Shaheen3107)
