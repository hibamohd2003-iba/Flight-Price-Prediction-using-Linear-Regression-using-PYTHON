# ✈️ Flight Price Prediction using Linear Regression

A machine learning project that predicts airline ticket prices using **Linear Regression**, implemented in **Python** (scikit-learn + statsmodels).

---

## 📁 Repository Structure

```
flight-delay-analysis/
├── analysis.py         # Full Python analysis script
├── flight-data.csv     # Dataset (300,153 records)
├── report.pdf          # Complete project report with results
├── plot1.png           # Price by airline (Delhi → Mumbai)
├── plot2.png           # Price vs. booking window
├── plot3.png           # Departure/arrival time heatmap
├── plot4.png           # Price distribution before/after outlier removal
├── plot5.png           # Actual vs. predicted prices (M1)
├── plot6.png           # Feature importance (top 10 coefficients)
└── README.md           # This file
```

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| airline | Categorical | 6 airlines (SpiceJet, Vistara, AirAsia, Indigo, GO_FIRST, Air_India) |
| source_city | Categorical | Departure city |
| departure_time | Categorical | Early Morning, Morning, Afternoon, Evening, Night, Late Night |
| stops | Categorical | zero, one, two_or_more |
| arrival_time | Categorical | Arrival time bin |
| destination_city | Categorical | Destination city |
| class | Categorical | Economy or Business |
| duration | Numeric | Flight duration (hours) |
| days_left | Numeric | Days between booking and departure |
| **price** | **Target** | **Ticket price in INR** |

---

## 🔍 Key Questions Analysed

1. Does price vary with airlines for the same source → destination?
2. How does price change when bought 1–2 days before departure?
3. Does departure/arrival time affect ticket prices?
4. Price distribution — outlier removal using IQR method.
5. M1 (all features) vs M2 (top 5 features) — R² and Adjusted R² comparison.
6. Scikit-learn vs Statsmodels OLS — both implemented and compared.

---

## ⚙️ Methodology

```
Raw Data (300,153 rows)
        │
        ▼
  Data Cleaning → IQR Outlier Removal → One-Hot Encoding
        │
        ▼
  Train/Test Split: 80% / 20%  (seed = 42)
        │
        ├──► M1: LinearRegression (All Features)  → R² = 0.9129 | RMSE = ₹6,664
        └──► M2: LinearRegression (Top 5 Features) → R² = 0.9021 | RMSE = ₹7,067
```

---

## 📈 Results

| Metric | M1 – All Features | M2 – Top 5 Features |
|---|---|---|
| R² | **0.9129** | 0.9021 |
| RMSE | ₹6,664 | ₹7,067 |

**Top 5 Features:** class_Economy, stops_zero, airline_Vistara, airline_SpiceJet, airline_Indigo

---

## 🖼️ Visualisations

### Price by Airline (Delhi → Mumbai)
![plot1](plot1.png)

### Price vs. Booking Window
![plot2](plot2.png)

### Departure & Arrival Time Heatmap
![plot3](plot3.png)

### Actual vs. Predicted (M1)
![plot5](plot5.png)

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
python analysis.py
```

---

## 🔮 Future Scope

- Ensemble methods: Random Forest, Gradient Boosting, XGBoost
- Real-time flight pricing data integration
- Flask / Streamlit web app for live price estimation

---

## 📚 References

- Scikit-learn Documentation — https://scikit-learn.org
- Statsmodels Documentation — https://www.statsmodels.org
- Kaggle: Flight Price Prediction Dataset

---

## 👩‍💻 Author

**Hiba Muhammed** | Roll No: 25 | S6IE | Department of IndustrialEngineering
