# Task 4 – Recommendation System (Collaborative Filtering)

## 🎯 Objective
Build a Recommendation System using **Collaborative Filtering** with matrix factorization techniques to suggest items for users based on past interactions.

---

## 📂 Deliverable
A Jupyter Notebook showcasing:
- Dataset loading
- Model training using Surprise SVD
- Evaluation metrics like RMSE/MAE
- Cosine similarity matrix for manual recommendation
- Heatmap visualization of user similarity

---

## 📦 Dataset
- Dataset: [MovieTweetings ratings.csv](https://github.com/sidooms/MovieTweetings)
- Columns used: `userId`, `movieId`, `rating`

---

## ⚙️ Tools Used
| Component         | Library        |
|------------------|----------------|
| Collaborative Filtering | `Surprise` |
| Matrix Similarity | `cosine_similarity` from `sklearn` |
| Visualization     | `seaborn`, `matplotlib` |
| Data Handling     | `pandas` |

---

## 🧠 Algorithms Used
- **SVD** (Singular Value Decomposition) – from `surprise` package
- **Cosine Similarity** – User-user matrix from rating pivot

---

## 🚀 How to Run

1. Open:
   ```
   main_advanced.ipynb
   ```

2. Run all cells in order. It will:
   - Train recommendation model
   - Display evaluation metrics
   - Visualize similarity matrix

---

## 📊 Output
- User similarity heatmap
- RMSE / MAE scores from cross-validation
- Printout of sample recommendations

---

## 📁 Files
| File                  | Description                            |
|-----------------------|----------------------------------------|
| `main_advanced.ipynb` | Full notebook with model + visualization |
| `similarity_heatmap.png` | Heatmap of user similarity          |

---

## 👨‍💻 Author
**Agatheeswaran R**  
Codetech ML Internship – Final Task 4
