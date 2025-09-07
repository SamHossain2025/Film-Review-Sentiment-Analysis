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


