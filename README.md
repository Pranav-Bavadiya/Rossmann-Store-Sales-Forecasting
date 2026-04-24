# 🛒 Rossmann Sales Predictor

Two completely independent deep-learning pipelines in one Streamlit app:

| Pipeline | Features | Input Shape | Models folder | Scaler |
|---|---|---|---|---|
| **Aggregate** | 22 | `(1, 60, 22)` or `(1, 1320)` for FNN | `saved_models/aggregate/` | 22-feature scaler |
| **Store-wise** | 30 | `(1, 60, 30)` or `(1, 1800)` for FNN | `saved_models/store/` | 30-feature scaler |

> ⚠️ **The two pipelines are NOT interchangeable.**  A model trained on 22 aggregate features
> cannot be used in the store-wise tab (different feature count and scaling).

---

## 📁 Project Structure

```
rossmann_predictor/
│
├── app.py                              ← Streamlit UI (run this)
│
├── models/
│   ├── load_models.py                  ← Separate registries for aggregate vs store models
│   └── predict.py                      ← Inference (point + quantile, FNN flat + 3-D RNN)
│
├── preprocessing/
│   ├── aggregate.py                    ← 22-feature aggregate pipeline (auto + manual)
│   └── store.py                        ← 30-feature per-store pipeline
│
├── utils/
│   ├── feature_engineering.py          ← Date, cyclical, lag, competition, Promo2 features
│   ├── sequence.py                     ← Builds (1,1320) for FNN or (1,60,F) for RNN/GRU
│   └── scaler.py                       ← Loads scalers from the correct pipeline subfolder
│
├── data/
│   ├── train.csv                       ← YOU PROVIDE (Rossmann training data)
│   └── store.csv                       ← YOU PROVIDE (store metadata)
│
├── saved_models/
│   │
│   ├── aggregate/                      ← AGGREGATE models (22 features, 60 timesteps)
│   │   ├── fnn_model.keras             ← input shape (None, 1320)  output (None, 1)
│   │   ├── gru_model.keras             ← input shape (None, 60, 22) output (None, 1)
│   │   ├── nhits_model.keras           ← input shape (None, 60, 22) output (None, 1)
│   │   ├── fnn_quantile_model.keras    ← input shape (None, 1320)  output (None, 3)
│   │   ├── rnn_quantile_model.keras    ← input shape (None, 60, 22) output (None, 3)
│   │   ├── gru_quantile_model.keras    ← input shape (None, 60, 22) output (None, 3)
│   │   ├── nhits_quantile_model.keras  ← input shape (None, 60, 22) output (None, 3)
│   │   ├── feature_scaler.pkl          ← MinMaxScaler fitted on 22 aggregate features
│   │   └── target_scaler.pkl           ← MinMaxScaler fitted on aggregate Sales
│   │
│   └── store/                          ← STORE-WISE models (30 features, 60 timesteps)
│       ├── fnn_model.keras             ← input shape (None, 1800)  output (None, 1)
│       ├── gru_model.keras             ← input shape (None, 60, 30) output (None, 1)
│       ├── nhits_model.keras           ← input shape (None, 60, 30) output (None, 1)
│       ├── fnn_quantile_model.keras    ← input shape (None, 1800)  output (None, 3)
│       ├── rnn_quantile_model.keras    ← input shape (None, 60, 30) output (None, 3)
│       ├── gru_quantile_model.keras    ← input shape (None, 60, 30) output (None, 3)
│       ├── nhits_quantile_model.keras  ← input shape (None, 60, 30) output (None, 3)
│       ├── feature_scaler.pkl          ← MinMaxScaler fitted on 30 store features
│       └── target_scaler.pkl           ← MinMaxScaler fitted on per-store Sales
│
└── requirements.txt
```

---

## ⚡ Quick Start

```bash
pip install -r requirements.txt

# 1. Add data files
cp /path/to/train.csv data/
cp /path/to/store.csv data/

# 2. Add AGGREGATE models (you already have these)
cp /path/to/fnn_model.keras            saved_models/aggregate/
cp /path/to/gru_model.keras            saved_models/aggregate/
cp /path/to/nhits_model.keras          saved_models/aggregate/
cp /path/to/fnn_quantile_model.keras   saved_models/aggregate/
cp /path/to/rnn_quantile_model.keras   saved_models/aggregate/
cp /path/to/gru_quantile_model.keras   saved_models/aggregate/
cp /path/to/nhits_quantile_model.keras saved_models/aggregate/
cp /path/to/feature_scaler.pkl         saved_models/aggregate/
cp /path/to/target_scaler.pkl          saved_models/aggregate/

# 3. Train STORE models (separate training) → place them in saved_models/store/
#    (see feature list below)

# 4. Run the app
streamlit run app.py
```

---

## 📐 Aggregate Feature Columns (22)

Training and inference must use these columns in this exact order:

```python
AGGREGATE_FEATURE_COLS = [
    "DayOfWeek", "Promo", "SchoolHoliday",
    "Year", "Month", "Day", "WeekOfYear", "IsWeekend",
    "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos",
    "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_21", "Sales_lag_28",
    "Rolling_mean_7", "Rolling_mean_14", "Rolling_mean_30", "Rolling_std_7",
    "Trend",
]  # 22 columns
```

## 📐 Store-wise Feature Columns (30)

```python
STORE_FEATURE_COLS = [
    "DayOfWeek", "Promo", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance",
    "Promo2", "CompetitionOpenDays", "Promo2RunningDays", "IsPromoMonth",
    "Year", "Month", "Day", "WeekOfYear", "IsWeekend",
    "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos",
    "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_21", "Sales_lag_28",
    "Rolling_mean_7", "Rolling_mean_14", "Rolling_mean_30", "Rolling_std_7",
    "Trend",
]  # 30 columns
```

---

## 🔧 Input Shapes by Model Type

| Model | Input for Aggregate | Input for Store |
|-------|--------------------|--------------------|
| FNN | `(1, 1320)` = 60×22 flat | `(1, 1800)` = 60×30 flat |
| GRU | `(1, 60, 22)` 3-D | `(1, 60, 30)` 3-D |
| RNN | `(1, 60, 22)` 3-D | `(1, 60, 30)` 3-D |
| N-HiTS | `(1, 60, 22)` 3-D | `(1, 60, 30)` 3-D |

`utils/sequence.py` handles this automatically based on model name.

---

## 🌟 App Features

- **Tab 1 — Aggregate Prediction**: Auto mode (pick date, data loaded automatically) + Manual mode (3 inputs)
- **Tab 2 — Store-wise Prediction**: Pick store ID + date + promo/holiday; all other features auto-computed; store metadata preview; multi-store comparison chart
- **Tab 3 — Data Insights**: Monthly sales trend, day-of-week analysis, promo impact, top stores, sales distribution
- **Sidebar status dashboard**: Shows which scalers and models are loaded/missing
- **Quantile output**: Lower / Median / Upper cards + bar chart
- **Point output**: Gauge chart + ₹ formatted with Lakh/Crore suffix
- **FNN flat input handled automatically** — no manual reshaping needed

---

## ❓ Why are there two separate model sets?

The aggregate pipeline **sums all stores** into a single global time series and trains on 22 date+lag features. The store pipeline trains on **individual store sales** and adds 8 more features describing the store itself (type, assortment, competition distance, Promo2 details). These are fundamentally different problems with different input dimensions — they cannot share models or scalers.
