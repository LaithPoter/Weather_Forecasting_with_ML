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

## 🧩 The Challenge

Two properties of this dataset shaped every modeling decision:

- **It is small.** After dropping the few rows with missing values, there are 365 days in total. With a chronological 80/20 split, the test set holds only **73 days**.
- **It is imbalanced.** About **68% of days are rainy** and only 32% are dry, so the *dry* days are the minority class here.

On data like this, **accuracy is misleading**. A model that answers "rain" every single day would already score about 66% accuracy on the test period while being useless for dry days.

So instead of optimizing accuracy, I tuned the model for **F1-score**, which only rewards a model that balances precision and recall on rainy days, and I checked per-class recall for both classes.

---

## 🛠️ Design Choices

### Chronological split instead of a random one

Weather is a time series. Each day is related to the days around it, and in real use a model is always asked about days it has not seen yet.

A random `train_test_split` would mix future days into training and test the model on days surrounded by its own training data. That leaks information and makes results look better than they would be in practice.

So the data is split by time:

- **Train:** the first 80% of the year
- **Test:** the last 20% (73 days the model never saw)

### `TimeSeriesSplit` for cross-validation

The same logic applies inside hyperparameter tuning. Normal K-Fold would still shuffle future data into training folds.

`TimeSeriesSplit(n_splits=5)` creates expanding windows: each fold trains only on the past and validates on the period right after it. The model is always tuned the way it would be used: **learn from the past, predict the future.**

### Pipeline + GridSearchCV

- A `Pipeline` combines `SimpleImputer` and the `RandomForestClassifier`, so preprocessing is fitted only on the training folds.
- `GridSearchCV` searched 324 combinations of `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` and `max_features`, scored by **F1**.

Best parameters found:

```text
n_estimators = 200
max_depth = None
max_features = None
min_samples_leaf = 2
min_samples_split = 2
class_weight = "balanced"
```

### `class_weight="balanced"`

Because dry days are the minority, an unweighted forest naturally drifts toward predicting "rain". Setting `class_weight="balanced"` makes mistakes on dry days count more during training, so the model is pushed to pay real attention to the minority class.

---

## 📊 Results

**Cross-validation F1 (TimeSeriesSplit, training period): 0.81**

**Held-out test period (last 73 days):**

| Class | Precision | Recall | F1 | Days |
|---|---|---|---|---|
| Dry (0) | 0.64 | 0.56 | 0.60 | 25 |
| Rain (1) | 0.78 | 0.83 | 0.81 | 48 |

- **Accuracy:** 0.74
- **Macro F1:** 0.70

In plain numbers:

- Out of **48 rainy days**, the model caught **40** and missed 8.
- Out of **25 dry days**, it correctly identified **14**. The other 11 got a rain warning that turned out to be a false alarm.

The test F1 for rain (0.81) is almost identical to the cross-validation F1 (0.81), which suggests the tuning did not overfit to the validation folds.

---

## ⚖️ The Trade-off

Even with `class_weight="balanced"`, the model is clearly stronger on rainy days (recall 0.83) than on dry days (recall 0.56).

That is an acceptable direction for this problem:

- **Missing a rainy day** (saying "no rain" and then it rains) disrupts outdoor events, logistics, agriculture and travel.
- **A false alarm** (saying "rain" on a dry day) mostly costs someone carrying an umbrella for nothing.

So the model accepts more false alarms in exchange for rarely being caught off guard by rain.

The weak spot is clear too: **dry-day recall of 0.56 is the main thing to improve**. With only 25 dry days in the test period, a few predictions change this number a lot, so more data matters more here than further tuning.

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
- Walk-forward validation across multiple years
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
├── weather forecasting project.ipynb   # preprocessing, tuning, evaluation
├── Weather forecasting EDA.ipynb       # exploratory data analysis
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
- TimeSeriesSplit

---

## 👨‍💻 Author

**Laith Hussein**

AI / Machine Learning | Computer Vision | Intelligent Systems

[LinkedIn](https://www.linkedin.com/in/laith-hussein-abaub/)