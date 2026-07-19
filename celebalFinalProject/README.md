# Lead Scoring ML Pipeline

A machine learning solution for X Education to identify and prioritize high-quality leads likely to convert into paying customers. This project builds a data-driven lead scoring model to improve sales team efficiency and increase conversion rates from 30% to a target of 80%.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Features](#features)
- [Model Details](#model-details)
- [Results](#results)

## 📊 Project Overview

X Education generates a large number of leads through various online channels, but only about 30% convert into paying customers. The sales team currently treats all leads equally, wasting time and resources on low-potential prospects.

**Problem:** Identify and prioritize high-quality leads that are more likely to convert.

**Solution:** Build a machine learning model that assigns a lead score to each prospect based on their likelihood of conversion, enabling the sales team to focus on the most promising leads.

**Target:** Improve conversion rate from 30% to 80%.
## 📦 Dataset

This project uses the **Lead Scoring Dataset** from Kaggle.

**Dataset Link:** [Lead Scoring Dataset](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset)

**Dataset Details:**
- Contains lead information from X Education's online sources
- Features include demographic information, engagement metrics, and behavioral patterns
- Target variable indicates whether a lead converted
- Used for training and evaluating the lead scoring model

**Download Instructions:**
1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset)
2. Download the dataset and place it in the project directory
3. Rename it to `Lead Scoring.csv` or update the file path in the code

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd celebalFinalProject
```

### Step 2: Create a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies included:**
- pandas >= 1.3 - Data manipulation and analysis
- scikit-learn >= 1.0 - ML algorithms and preprocessing
- xgboost - Gradient boosting classifier
- joblib >= 1.0 - Model serialization
- numpy >= 1.21 - Numerical computing
- shap >= 0.41 - Model interpretability
- streamlit >= 1.30 - Interactive web dashboard

## 📁 Project Structure

```
celebalFinalProject/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── problemStatement.txt                # Problem statement and objectives
├── Lead Scoring.csv                   # Dataset file
├── app.py                             # Streamlit dashboard application
├── lead_scoring_notebook.ipynb        # Jupyter notebook with full ML pipeline
├── models/
│   ├── model.joblib                  # Trained XGBoost classifier
│   ├── preprocessor.joblib           # Preprocessing pipeline (scaler, encoder)
│   └── lead_scores_table.csv         # Sample scored leads
```

##  Usage

### 1. Interactive Dashboard 

Run the Streamlit web application to explore the model and score new leads:

```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

**Features:**
- Upload CSV files with lead data
- View predictions and lead scores
- Filter and rank leads by score (Hot, Warm, Cold)
- Download scored leads as CSV
- Model performance metrics

### 2. Jupyter Notebook

For detailed exploration and model development, open the notebook:

```bash
jupyter notebook lead_scoring_notebook.ipynb
```

The notebook contains:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Model training and comparison
- Hyperparameter tuning
- Model evaluation and validation
- SHAP interpretability analysis



## 📝 Notes

- The model is pre-trained and ready to use. No retraining is required for basic scoring.
- For retraining with new data, modify the notebook and save new model files to `models/`
- Feature engineering and hyperparameter tuning details are in the Jupyter notebook
- SHAP analysis provides insights into which features drive predictions


