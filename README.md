# ✈️ Flight Price Prediction using Machine Learning

An end-to-end Machine Learning project that predicts flight prices using key features like airline, source, destination, stops, duration, and more. This project includes data preprocessing, model building (XGBoost, Random Forest, Decision Tree, Linear Regression), performance comparison, and a deployed Streamlit web app for live predictions.

---

## 🚀 Project Highlights

- Cleaned and preprocessed a real-world flight dataset  
- Feature engineering: datetime split, duration conversion, log transformation  
- Applied one-hot and ordinal encoding  
- Trained and evaluated multiple ML models  
- Built and deployed a user-friendly **Streamlit** app  
- Used `joblib` to save the full pipeline for fast inference  

---

## 📊 Model Performance Comparison

| Model              | Train Score | Test Score | R² Score | MAE     | MSE     | RMSE    |
|-------------------|-------------|------------|----------|---------|---------|---------|
| **XGBoost**        | 0.9574      | 0.9260     | 0.9353   | 0.2096  | 0.0773  | 0.2781  |
| **Random Forest**  | 0.9574      | 0.9260     | 0.9260   | 0.2002  | 0.0884  | 0.2973  |
| **Decision Tree**  | 0.9582      | 0.9148     | 0.9148   | 0.2081  | 0.1018  | 0.3190  |
| **Linear Regression** | 0.9137   | 0.9136     | 0.9136   | 0.2485  | 0.1032  | 0.3212  |

🔍 **Best Model Chosen:** `XGBoost Regressor` for final deployment based on R², MAE, and RMSE.

---

## 🧠 Tech Stack Used

| Tool/Library     | Purpose                          |
|------------------|----------------------------------|
| Python           | Core programming language        |
| Pandas, NumPy    | Data cleaning and manipulation   |
| Matplotlib, Seaborn | Data visualization            |
| Scikit-learn     | Preprocessing, modeling, metrics |
| XGBoost          | Primary regression model         |
| Streamlit        | Web app framework                |
| Joblib           | Save/load trained ML pipeline    |

---



## 🛠️ How to Run the Project

### 1. Clone the Repository

Directely convert fp_app.ipynb into .py 
or
Anaconda Prompt  in use jupyter nbconvert --to script your_notebook.ipynb

### 2. Run StramLit App
Anaconda Prompt Run:streamlit run fp_app.py 

Dharmik Bamrotiya
🎓 MCA Student | 💻 Aspiring Data Scientist | 📊 Data Analytics Enthusiast | ML Engineer
📍 India

