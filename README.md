# 📈 Sentiment-Driven Stock Movement Prediction using XGBoost  

A machine-learning pipeline designed to predict **next-day stock price direction** using sentiment signals extracted from social media and market volatility indicators.  
The project uses **XGBoost**, optimized through **Time-Series Cross-Validation**, to forecast price movement with probabilistic confidence scores suitable for algorithmic trading.

---

## 🔗 Dataset  
The dataset used in this project is available on Kaggle:

👉 **Kaggle Link:** (https://www.kaggle.com/datasets/thedevastator/tweet-sentiment-s-impact-on-stock-returns)

---

## 🧠 Model Architecture  

| Component | Description |
|----------|-------------|
| Algorithm | XGBoostClassifier (boosted decision trees) |
| CV Strategy | TimeSeriesSplit for chronological training |
| Objective | Binary classification (Up vs Down) |
| Output | Class label + probability of upward movement |

### Feature Inputs  
| Feature | Meaning |
|--------|---------|
| `mean_sentiment` | overall polarity score |
| `sentiment_std` | sentiment volatility |
| `tweet_count` | sentiment message volume |
| `PX_VOLUME` | market liquidity indicator |
| `VOLATILITY_10D` | short-term market volatility |

---

## ⚙ Installation  

```bash
pip install xgboost scikit-learn numpy pandas matplotlib
