import os
import pandas as pd

#data_path = "/Desktop/IMDB Movie Review Sentiment/train_data"
data_path = "/Desktop/IMDB Movie Review Sentiment/test_data"
reviews = []
labels = []

# data link  : https://ai.stanford.edu/~amaas/data/sentiment/

# Iterate clearly over 'pos' and 'neg' folders
for label in ['pos', 'neg']:
    folder_path = os.path.join(data_path, label)  # fix: correct path construction
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Open each text file
        with open(file_path, 'r', encoding='utf-8') as file:
            review_text = file.read()
            reviews.append(review_text)
            
            # Attach label clearly: pos=1, neg=0
            labels.append(1 if label == 'pos' else 0)

# Creating a DataFrame clearly
df = pd.DataFrame({'review': reviews, 'label': labels})

# Verify DataFrame clearly
print(df.head())
print(f"Number of reviews: {len(df)}")
print(f"Columns in DataFrame: {df.columns.tolist()}")

#df.to_csv("imdb_train_dataset.csv", index=False)
df.to_csv("imdb_test_dataset.csv", index=False)

