# Rainfall Prediction

> A machine learning project that predicts whether it will rain on a given day — with a focus on understanding the data, choosing an appropriate evaluation strategy, and being honest about the limits of the dataset.

---

## 🎯 Project Overview

This project predicts whether rainfall will occur on a given day using historical weather measurements.

I used a **Random Forest Classifier** and tuned it with `GridSearchCV`.

The project was built as a practical ML classification workflow:

```text
Weather Data
     ↓
EDA
     ↓
Data Cleaning
     ↓
Preprocessing
     ↓
Random Forest
     ↓
GridSearchCV
     ↓
Evaluation
     ↓
Interpretation
```

The goal was not simply to maximize the model score.

I wanted to understand:

> **Which weather signals actually matter, and what happens when the cost of missing an event is different from the cost of a false alarm?**

---

## 🌦️ Dataset

The dataset contains **365 daily observations** from one location and one year.

Features include:

- Pressure
- Maximum temperature
- Minimum temperature
- Average temperature
- Dew point
- Humidity
- Cloud cover
- Sunshine hours
- Wind direction
- Wind speed

Target:

```text
rainfall → yes / no
```

### Class distribution

Approximately:

- **68% rainy days**
- **32% dry days**

This means the dataset is moderately imbalanced, so simply looking at accuracy would not provide the complete picture.

---

## ⚠️ An Important Limitation

This is one of the most important things I learned from this project.

The dataset contains only **365 observations from a single location and a single year**.

That means the model has not seen:

- Multiple years of weather variation
- Different locations
- Different climate patterns
- Long-term seasonal changes

Therefore, even if the model performs well on the available test data, I would **not** describe this as a production-ready weather forecasting system.

It is better understood as a:

> **Proof of concept and ML learning project.**

A real forecasting system would require significantly more diverse and representative data.

I consider recognizing this limitation part of evaluating the model correctly.

---

## 🔍 Exploratory Data Analysis

Before modeling, I examined:

- Feature distributions
- Class balance
- Correlations
- Relationships between weather variables and rainfall

The purpose of EDA was not simply to generate visualizations.

It was to understand whether the model's eventual predictions could be connected to reasonable weather behavior.

---

## 🌧️ What the Data Revealed

Some of the strongest relationships with rainfall were:

### Cloud Cover
Correlation: approximately **+0.63**

Higher cloud cover was strongly associated with rainfall.

### Humidity
Correlation: approximately **+0.49**

Higher humidity also showed a positive relationship with rainfall.

### Sunshine
Correlation: approximately **-0.55**

More sunshine was associated with a lower probability of rainfall.

These relationships are also intuitively consistent with the weather phenomenon being modeled, which makes them more interesting than correlations that appear purely because of the dataset.

---

## 🤖 Model

I used a **Random Forest Classifier**.

Instead of relying on default parameters, I used `GridSearchCV` with **5-fold cross-validation**.

The search explored parameters including:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

The optimization target was **F1 score**.

---

## 🎯 Why F1 Instead of Only Accuracy?

The dataset is not perfectly balanced.

More importantly, the cost of a prediction error is not necessarily symmetric.

For example, missing a rainy day may be more disruptive for:

- Outdoor events
- Logistics
- Agriculture
- Travel planning

than warning someone about rain that does not eventually occur.

For that reason, I was interested in the balance between **Precision and Recall**, rather than treating accuracy as the only objective.

The model therefore leaned toward catching more rainy days, accepting some additional false alarms.

---

## 💡 What I Learned

### 1. Dataset quality matters as much as the algorithm

A sophisticated algorithm cannot compensate for a dataset that does not represent the real problem well.

A model trained on one year from one location cannot automatically generalize to the weather of an entire country.

---

### 2. Model performance needs context

A good score on a small dataset does not automatically mean a good real-world system.

The question should always be:

> **"Good performance on what data, representing what population?"**

---

### 3. Metrics should reflect the problem

Choosing F1 was not just a technical decision.

It was connected to how the model would actually be used.

If missing rainfall is more costly than generating an unnecessary warning, Recall becomes particularly important.

---

### 4. Correlation should still make sense

The strongest relationships in the dataset — cloud cover, humidity, and sunshine — were consistent with reasonable weather behavior.

That doesn't prove causation, but it gives us a more meaningful interpretation of the model's inputs.

---

## 🚀 How This Could Become a Real Forecasting System

To move beyond this proof of concept, the next version would need:

- Multiple years of historical data
- Multiple geographic locations
- Better temporal validation
- More observations
- External weather sources
- Feature engineering based on time and season
- Comparison against stronger baseline models
- Proper deployment and monitoring

That would turn this from a small ML project into a much more realistic forecasting system.

---

## 📁 Project Structure

```text
.
├── Rainfall_Prediction.ipynb
├── Rainfall.csv
├── requirements.txt
└── README.md
```

---

## 🧰 Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Random Forest
- GridSearchCV

---

## 👨‍💻 Author

**Laith Hussein**

AI / Machine Learning | Computer Vision | Intelligent Systems

[LinkedIn](https://www.linkedin.com/in/laith-hussein-abaub/)