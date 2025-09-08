<p align="center">
  <img src="3 Assets/Banner.png" width="100%">
</p>

# 🎬 Film Review Sentiment Analysis

*Classifies movie reviews as positive or negative using machine learning and TF-IDF vectorization on curated sentiment-labeled datasets.*

- Topics covered: **Natural Language Processing (NLP)**
- Models used: **Logistic Regression**
- Skills demonstrated: **Text cleaning, TF-IDF, model tuning, evaluation, error analysis**
- Expected outcome:
    - **Automatically classify film reviews**
    - **Enable further deployment as a web app**

---

## 👥 Authors

**Sam Hossain**

---

## 🔍 Problem Statement

**Goal:**  

Build a machine learning model to classify film reviews into positive or negative sentiment based on the textual content.

**Challenges Addressed:**
- Dealing with informal, unstructured text data
- Cleaning and preprocessing textual reviews
- Extracting meaningful features from language
- Model tuning and cross-validation
- Handling subtle and ambiguous sentiment

This project applies a complete natural language processing (NLP) workflow to a real-world dataset of movie reviews. It begins by cleaning raw text, then vectorizes the data using TF-IDF, and trains a Logistic Regression classifier with hyperparameter tuning. Evaluation metrics like Accuracy and F1 Score are used to measure performance. The model is also analyzed for errors and saved for future deployment as a Streamlit-based review analyzer.

---

## 🔧 Workflow

### ✅ Data Preparation
- Loaded and combined labeled training and testing datasets
- Applied text cleaning (lowercase, punctuation removal, whitespace normalization)
- Vectorized text using `TfidfVectorizer`

<p align="center">
  <img src="3 Assets/Data1.png" width="80%">
</p>

### 🤖 Models Built
| Model | Type | Feature Extraction | Target | Technique |
|-------|------|---------------------|--------|-----------|
| Model 1 | Logistic Regression | TF-IDF | Polarity (0/1) | GridSearchCV + Evaluation |

✅ **Model verdict:** Achieved 78% accuracy and F1 score, demonstrating strong baseline performance.

<p align="center">
  <img src="3 Assets/Model1.png" width="80%">
  <img src="3 Assets/Model2.png" width="80%">
  <img src="3 Assets/Model3.png" width="80%">
  <img src="3 Assets/Model4.png" width="80%">
</p>


### 🧪 Sentiment Classification Process
- Preprocess all reviews (training + test)
- Vectorize using TF-IDF (max 5000 features)
- Train Logistic Regression with `GridSearchCV`
- Evaluate predictions on the test set
- Save trained model and vectorizer (`joblib`)
- Analyze 5 most interesting misclassifications

---

## 🎯 Key Findings

* TF-IDF is effective for feature extraction from review text
* Logistic Regression achieves good performance with tuned hyperparameters
* Misclassifications often involve negations, sarcasm, or subtle sentiment

<p align="center">
  <img src="3 Assets/Model5.png" width="80%">
  <img src="3 Assets/Model6.png" width="80%">
</p>

---

## 💡 Key Recommendations

* Use deep learning models like BERT for contextual understanding
* Enhance negation and sarcasm detection via custom rules or lexicons
* Build a real-time app with Streamlit or Gradio using the saved model

---

## 🚀 How to Run This Project

1. **Clone the repository**

   ```bash
   git clone https://github.com/SamHossain2025/Film-Review-Sentiment-Analysis.git
   cd Film-Review-Sentiment-Analysis
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

4. **Open and execute the notebook**

   * `2 Codes/MMA865_SamHossain.ipynb` → Full modeling pipeline

---

## 🧬 Data Sources

* 📌 Provided movie review datasets (`sentiment_train.csv`, `sentiment_test.csv`)
* 📌 Prepared in-house for academic coursework (MMA865)

---

## 🔓 License

This project is licensed under the [MIT License](LICENSE)
