# TMDB_Movie_EDA_ML_MovieRecommendationSystem
# 🎬 Movie Recommendation System
A Hybrid Machine Learning Recommender using **Sentiment Mining**, **Autoencoder Embeddings**, and **LightGBM**

---

## 📌 Overview
This project builds a hybrid movie recommendation system using a **550K+ multilingual film dataset**.  
It combines:

- **Emotional similarity** (cosine similarity on autoencoder sentiment embeddings)  
- **Genre preference matching**  
- **Blockbuster probability prediction** using LightGBM  

The result is a personalized recommender that suggests movies similar to a user’s favourites while also considering genre preferences and mainstream popularity likelihood.  
The notebook also includes **EDA, feature engineering, and ML modeling** to understand what makes a movie a blockbuster.

---

## 🚀 Features

### 🎥 Hybrid Recommendation Engine
The system blends three components:

| Component              | Description                                               | Weight |
|------------------------|-----------------------------------------------------------|--------|
| **Sentiment Similarity** | Cosine similarity between compressed sentiment embeddings | 0.6    |
| **Blockbuster Probability** | LightGBM classifier predicting if a movie is a “hit”     | 0.4    |

This hybrid scoring method produces recommendations that are:
- Emotionally aligned with user taste  
- Genre-aware  
- Likely to be popular  

---

### 📊 Exploratory Data Analysis (EDA)
The notebook includes:
- Sentiment distributions (NRC EmoLex, VAD, Intensity)  
- Popularity trends  
- Genre-level insights  
- Cast/Crew reputation analysis  
- Correlation heatmaps  

---

### 🧠 Machine Learning
The project trains a **LightGBM classifier** to predict whether a movie is a blockbuster.  
Key steps include:
- Train/test split  
- Standard scaling  
- Feature engineering (budget, runtime, release year, genre count)  
- Cast/Crew reputation aggregation  
- Model evaluation (ROC-AUC, F1, Precision)  

---

### 🔍 Movie Title Search
Users can input any favourite movie titles, and the system will:
- Find exact matches  
- Fall back to partial matching if needed  
- Compute sentiment similarity and blockbuster probability  

---

### 📦 Clean Output
The recommender returns a compact DataFrame containing:
- `title`  
- `release_year`  
- `genres`  
- `vote_average`  
- `sentiment_similarity`  
- `blockbuster_probability`  
- `hybrid_score`  

Perfect for display or UI integration.  
Each recommendation also includes a **radar chart** comparing sentiment profiles.

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- LightGBM  
- Keras (Autoencoder)  
- NLTK, NRC Lexicons (EmoLex, VAD, Intensity)  
- Matplotlib (Radar Charts)  


---

## 📁 Project Structure


```text
Movie-Analysis-MachineLearning-RecommendationSystem/
│
├── data/
│   └── movies_db.csv             # Scored dataset (bb_prob + embeddings)
│
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_Phase1_ML.ipynb         # Blockbuster classification
│   └── 03_Phase2_RecSystem.ipynb  # Recommendation engine
│
├── artifacts/
│   ├── lgbm_model.pkl             # Trained LightGBM model
│   ├── encoder.h5                 # Trained autoencoder encoder
│   ├── scaler.pkl                 # StandardScaler
│   └── movie_embeddings.npy       # Precomputed sentiment embeddings
│
└── README.md                      # Project documentation

```
## ▶️ Example Usage
```python
recommend(
    favourite_titles = ["Inception"],
    preferred_genres = ["Science Fiction"],
    min_bb_prob      = 0.70,
    top_n            = 6
)

Building profile from: ['Inception']
Genre filter applied: ['Science Fiction'] → 14591 candidates
Blockbuster filter (≥0.7): 328 candidates

                               title  release_year  vote_average  sim_score   bb_prob  final_score
0                         Armageddon          1998         6.829   0.961245   0.846637     0.915402
1                 Sonic the Hedgehog          2020         7.345   0.962826   0.836518     0.912303
2             Avengers: Infinity War          2018         8.255   0.947106   0.846637     0.906919
3                       Transformers          2007         6.800   0.946983   0.846637     0.906845
4  Spider-Man: Into the Spider-Verse          2018         8.404   0.944431   0.846637     0.905314
5                       The Avengers          2012         7.710   0.941591   0.846637     0.903609
```

🙌 Acknowledgements

Dataset: [TMDB Movies Dataset (Kaggle)](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)    
Final Cleaned Dataset: [TMDB_purified_dataset](https://drive.google.com/file/d/1oAyc066ZgvVJUNBu7oPjjYs25ts6XTOW/view?usp=sharing)

This project was created to explore movie analytics, machine learning, and recommender system design.

📬 Contact 

For questions, collaboration, or feedback: James Nguyen
Data Analyst & Dashboard Designer (Linkedin: https://www.linkedin.com/in/tran-bao-minh-nguyen-296b01333/
For more projects finding:https://baominhnt.github.io/tbmnguyen.com/index.html)
