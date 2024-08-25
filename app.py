import pandas as pd
import streamlit as st
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data files
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Set of stop words to remove
stop_words = set(stopwords.words('english'))

# Custom stop words that are common in ingredient lists but not useful
custom_stop_words = set(['chopped', 'sliced', 'fresh', 'dried', 'ground', 'whole', 'finely', 'minced'])

# Synonym mapping (customize this list based on your data)
synonym_map = {
    'bell pepper': 'bell pepper',
    'capsicum': 'bell pepper',
    'cilantro': 'coriander',
    'garlic powder': 'garlic',
    'onions': 'onion',
    'tomatoes': 'tomato',
    'olive oil': 'oil',
    'black pepper': 'pepper',
}

def normalize_ingredient(ingredient):
    # Step 1: Convert to lowercase
    ingredient = ingredient.lower()

    # Step 2: Remove punctuation and special characters
    ingredient = re.sub(r'[^a-zA-Z\s]', '', ingredient)

    # Step 3: Tokenize the ingredient string
    tokens = ingredient.split()

    # Step 4: Remove stop words and custom stop words
    tokens = [word for word in tokens if word not in stop_words and word not in custom_stop_words]

    # Step 5: Lemmatize the remaining words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Step 6: Reconstruct the ingredient string
    normalized_ingredient = ' '.join(tokens)

    # Step 7: Apply synonym mapping
    for key, value in synonym_map.items():
        if key in normalized_ingredient:
            normalized_ingredient = normalized_ingredient.replace(key, value)

    return normalized_ingredient

def find_similar_recipes(user_ingredients, df, tfidf_matrix, tfidf_vectorizer, top_n=5):
    """
    Find top N recipes based on similarity to user's ingredients.
    """
    # Normalize the user's ingredients
    normalized_user_ingredients = normalize_ingredient(user_ingredients)

    # Transform the user's ingredients into a TF-IDF vector
    user_tfidf_vector = tfidf_vectorizer.transform([normalized_user_ingredients])

    # Compute cosine similarity between user's ingredients and all recipes
    cosine_similarities = cosine_similarity(user_tfidf_vector, tfidf_matrix).flatten()

    # Get the top N similar recipe indices
    top_recipe_indices = cosine_similarities.argsort()[-top_n:][::-1]

    # Return the top N recipes
    return df.iloc[top_recipe_indices]

# Load the dataset
df = pd.read_csv('recipes_normalized.csv')

# Ensure the dataset has 'title' and 'link' columns
if not {'title', 'link'}.issubset(df.columns):
    st.error("The dataset must contain 'title' and 'link' columns.")
    st.stop()

# Normalize the ingredient column
df['normalized_ingredients'] = df['ingredients'].apply(normalize_ingredient)

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the ingredient data into TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(df['normalized_ingredients'])

# Streamlit app title
st.title('Recipe Finder')

# User input
user_ingredients = st.text_input("Enter the ingredients you have (separated by commas):")

if user_ingredients:
    # Find top 5 or 10 similar recipes
    top_n = st.slider('Select the number of top recipes to display:', 5, 10, 5)
    top_recipes = find_similar_recipes(user_ingredients, df, tfidf_matrix, tfidf_vectorizer, top_n=top_n)

    # Create a new DataFrame for displaying in Streamlit
    display_df = top_recipes[['title', 'link']].copy()

    # Ensure URLs are properly formatted
    display_df['link'] = display_df['link'].apply(lambda x: x if x.startswith(('http://', 'https://')) else f'http://{x}')

    # Convert links to clickable titles
    display_df['title'] = display_df.apply(lambda row: f'<a href="{row["link"]}" target="_blank">{row["title"]}</a>', axis=1)

    # Use Streamlit's markdown function to render the dataframe with clickable links
    st.write("### Top Recipes Similar to Your Ingredients:")
    st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
