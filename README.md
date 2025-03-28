# 🔍 Customer Churn Prediction with Random Forest & Feature Engineering

This project uses machine learning to predict customer churn based on historical consumption and pricing data. The primary goal is to help businesses proactively identify at-risk customers and take preventive action. Built with a focus on explainability and practical use, this solution leverages a **Random Forest Classifier**, engineered features, and thorough evaluation techniques.

> 📊 This project is all about turning raw customer data into actionable churn insights.

---

## 🚀 Features

- 📂 Merged & cleaned client and pricing datasets  
- 🧠 Feature engineering: consumption ratios, tenure bins, forecast accuracy, and more  
- 🌲 Trained using a Random Forest Classifier  
- 📉 Evaluation includes Accuracy, Precision, Recall, F1, and ROC-AUC  
- 📈 Confusion matrix visualization for model interpretability  
- ⚙️ Easily runnable Python script with requirements file  

---

## 🧠 How It Works

1. **Data Loading**: Two datasets — customer behavior and energy pricing — are imported.  
2. **Feature Engineering**: New fields are created (e.g., tenure duration, consumption ratios, forecast vs actual gaps).  
3. **Preprocessing**: Merging, handling nulls, and filtering are performed.  
4. **Model Training**: A `RandomForestClassifier` is trained with a 75-25 train-test split.  
5. **Evaluation**: The model is tested and scored on multiple metrics, with a clear confusion matrix.  

---

## 📊 Model Performance

| Metric      | Score     |
|-------------|-----------|
| Accuracy    | 90.28%    |
| Precision   | 92.31%    |
| Recall      | 3.28%     |
| F1-Score    | 0.0633    |
| ROC-AUC     | 0.5162    |

### 📉 Confusion Matrix
```
[[3285    1]
 [ 354   12]]
```
👉 **Insight**: The model performs well on non-churners but struggles with detecting churners — this is a common issue with class imbalance.

---

## 🗂️ Project Structure

```
churn-prediction/
├── data/
│   ├── client_data.csv
│   └── price_data.csv
├── churn_prediction.py       # Main pipeline
├── churn_feature_creation.py
├── churn_data_cleaning.py
├── README.md                 # 📘 This file
```

---

## 📦 Tech Stack

| Layer        | Tool/Library       |
|--------------|--------------------|
| Language     | Python             |
| ML Model     | Random Forest      |
| Data Engine  | pandas, numpy      |
| Viz & Stats  | seaborn, matplotlib|
| Evaluation   | scikit-learn       |

---

## 💻 Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the churn prediction pipeline**
```bash
python churn_prediction.py
```

Ensure your datasets are placed in the `data/` folder as expected.

---

## ✅ Sample Input Features

| Feature                | Example Value     |
|------------------------|-------------------|
| cons_12m               | 400               |
| forecast_cons_12m      | 350               |
| margin_gross_pow_ele   | 55.5              |
| net_margin             | 10.5              |
| churn                  | 1 (yes), 0 (no)   |

---

## 👤 Maintainer

📬 **Bhuvaneshwari Balaji**  
🔗 GitHub: [github.com/060205b](https://github.com/060205b)  
📧 Email: bhuvaneshwaribalaji06@gmail.comb
