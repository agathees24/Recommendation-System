# TASK 4 – RECOMMENDATION SYSTEM USING COLLABORATIVE FILTERING

"""
Author: Agatheeswaran R
Task: Codetech ML Internship – Task 4
Description: Collaborative Filtering for recommendation system using Surprise SVD and cosine similarity.
"""

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# 📥 Load MovieTweetings Ratings Dataset
ratings = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv')
ratings.rename(columns={"user_id": "userId", "book_id": "movieId"}, inplace=True)
ratings = ratings[['userId', 'movieId', 'rating']].dropna()

# ✅ Use Surprise for Collaborative Filtering
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings, reader)

# Apply SVD (Singular Value Decomposition)
model = SVD()
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# 📊 Create Cosine Similarity Matrix for User-User similarity
pivot = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Use only a small subset to avoid memory issues
subset_size = 1000  # You can reduce this if needed
pivot_small = pivot.iloc[:subset_size, :]

# Compute cosine similarity on the small subset
similarity_matrix = cosine_similarity(pivot_small)

# Visualize the similarity matrix (user-user)
sns.heatmap(similarity_matrix[:10, :10], cmap='viridis')
plt.title("User Similarity Heatmap (first 1000 users)")
plt.savefig("similarity_heatmap.png")
plt.show()

# Save model if needed (optional step)
# joblib.dump(model, "recommendation_model.pkl")

print("✅ Recommendation system evaluation complete!")
