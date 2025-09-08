# 🎬 Film Review Sentiment Analysis
**Natural Language Processing (NLP) | Logistic Regression | TF-IDF | Model Evaluation | Python**

This project builds a supervised machine learning pipeline to analyze sentiment in film reviews — classifying each review as either **Positive** or **Negative**. It applies core NLP techniques (text cleaning, TF-IDF) and uses Logistic Regression for classification, with performance evaluation through Accuracy and F1 Score.

---

## 📁 Folder Structure

Film-Review-Sentiment-Analysis/
│
├── 1 Raw Data/
│ ├── sentiment_train.csv # Training dataset
│ └── sentiment_test.csv # Test dataset
│
├── 2 Codes/
│ └── MMA865_SamHossain.ipynb # Main Jupyter Notebook (cleaned & commented)
│
├── README.md # This file
└── requirements.txt (optional) # Dependency list (to be added if needed)


---

## 🚀 Quick Start

### 🧰 Prerequisites

Install dependencies using pip:


pip install pandas numpy scikit-learn joblib



---

▶️ Run the Notebook

Open 2 Codes/MMA865_SamHossain.ipynb in Jupyter or Google Colab.

The notebook will:

Load data from 1 Raw Data/

Clean and vectorize the text

Train a Logistic Regression model using GridSearchCV

Evaluate performance (Accuracy, F1 Score)

Save model and vectorizer for deployment

Perform error analysis on misclassified samples
---

📊 Results

Accuracy: ~78%

F1 Score: ~78%

Model: Logistic Regression with TF-IDF features

Best Parameters: C=10, solver='liblinear'

💡 Key Observations:

Handles clear sentiment well

Struggles with:

Negations ("wasn't terrible")

Figurative language ("the camera likes her")

Subtle or ambiguous tones

🧠 What I Learned

Complete NLP pipeline for binary text classification

Text preprocessing: lowercasing, punctuation removal, whitespace cleanup

TF-IDF vectorization for turning text into machine-readable features

Hyperparameter tuning using GridSearchCV

Model evaluation using precision, recall, F1 score, and error analysis

Saved model/vectorizer for future Streamlit dashboard deployment

💻 What’s Next

✅ Turn this into a web app using Streamlit or Gradio

✅ Add saved model files (sentiment_model.pkl, tfidf_vectorizer.pkl)

📈 Explore deep learning alternatives (e.g., BERT)

🎯 Improve negation handling and sarcasm detection

📜 License

This project is open source under the MIT License
.

📫 Contact

Sam Hossain
Data × AI/ML × Finance | MMA (Smith School of Business, Queen’s University)
📧 Email: [Your Email Here]
🌐 Portfolio: https://hossainsam.ca

🔗 LinkedIn: https://linkedin.com/in/sam-hossain


---

## ✅ Next Steps for You

1. Create a new file `README.md` inside the root of your GitHub folder and paste the above content.
2. (Optional) Create `requirements.txt` by running:

```bash
pip freeze > requirements.txt










<p align="center">
  <img src="3 Assets/Banner.png" width="100%">
</p>

# 🎬 Film Review Sentiment Classifier with Logistic Regression

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
  <img src="3 Assets/Findings1.png" width="80%">
  <img src="3 Assets/Findings2.png" width="80%">
  <img src="3 Assets/Findings3.png" width="80%">
  <img src="3 Assets/Findings4.png" width="80%">
  <img src="3 Assets/Findings5.png" width="80%">
</p>

---

## 💡 Key Recommendations

* Use deep learning models like BERT for contextual understanding
* Enhance negation and sarcasm detection via custom rules or lexicons
* Build a real-time app with Streamlit or Gradio using the saved model

<p align="center">
  <img src="3 Assets/Recommendation1.png" width="80%">
  <img src="3 Assets/Recommendation2.png" width="80%">
  <img src="3 Assets/Recommendation3.png" width="80%">
  <img src="3 Assets/Recommendation4.png" width="80%">
  <img src="3 Assets/Recommendation5.png" width="80%">
</p>

---

## 📊 Output Dashboard

<p align="center">
  <img src="3 Assets/Dashboard1.png" width="80%">
  <img src="3 Assets/Dashboard2.png" width="80%">
  <img src="3 Assets/Dashboard3.png" width="80%">
  <img src="3 Assets/Dashboard4.png" width="80%">
  <img src="3 Assets/Dashboard5.png" width="80%">
</p>

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
