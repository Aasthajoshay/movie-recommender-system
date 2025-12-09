# MovieFlick: Movie Recommender System

A content-based movie recommendation system built with Python and Streamlit that suggests similar movies based on movie tags and descriptions.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Step-by-Step Code Explanation](#step-by-step-code-explanation)
- [Running the Project](#running-the-project)
- [Usage](#usage)

## 🎬 Overview

This project implements a content-based recommendation system that analyzes movie tags and descriptions to find similar movies. When you select a movie, the system recommends 5 movies that are most similar based on their content features.

## ✨ Features

- Interactive web interface using Streamlit
- Content-based movie recommendations
- Cosine similarity algorithm for finding similar movies
- Pre-computed similarity matrix for fast recommendations

## 📦 Requirements

- Python 3.7+
- pandas
- streamlit
- scikit-learn
- pickle (built-in)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Aasthajoshay/movie-recommender-system.git
cd movie-recommender-system
```

2. Install required packages:
```bash
pip install streamlit pandas scikit-learn
```

## 📁 Project Structure

```
movie-recommender-system/
│
├── app.py                      # Main Streamlit application
├── build_similarity.py         # Script to build similarity matrix
├── movie_dict.pkl             # Movie data dictionary (pickle file)
├── movies.pkl                  # Movie DataFrame (pickle file)
├── similarity.pkl              # Pre-computed similarity matrix
├── movie-recommender-system.ipynb  # Jupyter notebook (optional)
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## 🧠 How It Works

1. **Data Preparation**: Movie data is stored in pickle files containing movie titles, tags, and other metadata.

2. **Similarity Matrix Creation**: The `build_similarity.py` script:
   - Loads movie data
   - Extracts tags/descriptions for each movie
   - Converts text to numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency)
   - Computes cosine similarity between all movie pairs
   - Saves the similarity matrix to `similarity.pkl`

3. **Recommendation**: When a user selects a movie:
   - The system finds the selected movie's index
   - Retrieves similarity scores for all movies
   - Sorts movies by similarity (highest first)
   - Returns the top 5 most similar movies (excluding the selected movie itself)

## 📝 Step-by-Step Code Explanation

### 1. `build_similarity.py` - Building the Similarity Matrix

This script creates the similarity matrix that powers the recommendation system.

#### Line-by-Line Explanation:

```python
import os
import pickle
import pandas as pd
```
**Lines 1-3**: Import necessary libraries
- `os`: For checking file existence
- `pickle`: For saving/loading Python objects
- `pandas`: For data manipulation

```python
def build_similarity():
    """Builds a cosine similarity matrix from the `tags` column and saves it to `similarity.pkl`."""
```
**Line 5**: Define the main function that builds the similarity matrix

```python
    if os.path.exists('movies.pkl'):
        with open('movies.pkl', 'rb') as f:
            movies = pickle.load(f)
    elif os.path.exists('movie_dict.pkl'):
        with open('movie_dict.pkl', 'rb') as f:
            movies = pd.DataFrame(pickle.load(f))
    else:
        raise FileNotFoundError('Neither movies.pkl nor movie_dict.pkl found in the workspace')
```
**Lines 8-15**: Load movie data
- First tries to load `movies.pkl` (if it's already a DataFrame)
- If not found, tries `movie_dict.pkl` and converts it to a DataFrame
- Raises an error if neither file exists

```python
    if not hasattr(movies, 'columns') or 'tags' not in movies.columns:
        raise ValueError('The movies data must contain a "tags" column')
```
**Lines 17-18**: Validate that the data has a 'tags' column
- Checks if the DataFrame has columns and specifically a 'tags' column
- Raises an error if 'tags' column is missing

```python
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as e:
        raise ImportError('scikit-learn is required to build the similarity matrix. Install it with "pip install scikit-learn"') from e
```
**Lines 20-24**: Import scikit-learn modules
- `TfidfVectorizer`: Converts text to numerical vectors using TF-IDF
- `cosine_similarity`: Computes similarity between vectors
- Handles import errors gracefully

```python
    tags = movies['tags'].astype(str).values
```
**Line 27**: Extract tags column
- Converts tags to string type (handles any non-string values)
- Gets values as a NumPy array

```python
    vectorizer = TfidfVectorizer(stop_words='english')
```
**Line 28**: Initialize TF-IDF vectorizer
- `stop_words='english'`: Removes common English words (the, is, a, etc.) to focus on meaningful words

```python
    vectors = vectorizer.fit_transform(tags)
```
**Line 29**: Convert text to numerical vectors
- `fit_transform()`: Learns vocabulary from tags and converts each movie's tags to a vector
- Returns a sparse matrix where each row is a movie and each column is a word

```python
    similarity = cosine_similarity(vectors)
```
**Line 30**: Compute similarity matrix
- Calculates cosine similarity between all pairs of movies
- Result is a square matrix where `similarity[i][j]` = similarity between movie i and movie j
- Values range from 0 (no similarity) to 1 (identical)

```python
    with open('similarity.pkl', 'wb') as f:
        pickle.dump(similarity, f)
```
**Lines 32-33**: Save similarity matrix
- Opens `similarity.pkl` in write-binary mode
- Saves the similarity matrix using pickle for fast loading later

```python
    print('Created similarity.pkl with shape', getattr(similarity, 'shape', None))
```
**Line 35**: Print confirmation message with matrix dimensions

```python
if __name__ == '__main__':
    build_similarity()
```
**Lines 38-39**: Run the function when script is executed directly
- Allows the script to be run standalone: `python build_similarity.py`

---

### 2. `app.py` - Streamlit Web Application

This is the main application file that creates the interactive web interface.

#### Line-by-Line Explanation:

```python
import pickle
import streamlit as st
import requests
import pandas as pd
```
**Lines 1-4**: Import required libraries
- `pickle`: Load saved data files
- `streamlit`: Create web interface
- `requests`: (Currently unused, but can be used for fetching movie posters from APIs)
- `pandas`: Work with DataFrame

```python
def recommend(movie):
```
**Line 8**: Define recommendation function
- Takes a movie title as input
- Returns a list of 5 recommended movie titles

```python
    movie_index = movies[movies['title'] == movie].index[0]
```
**Line 9**: Find the index of the selected movie
- Filters DataFrame to find rows where title matches the selected movie
- Gets the first matching index (`.index[0]`)
- This index corresponds to the row in both `movies` DataFrame and `similarity` matrix

```python
    distances = similarity[movie_index]
```
**Line 10**: Get similarity scores for the selected movie
- Retrieves the entire row from similarity matrix
- `distances` is an array where each element is the similarity score between the selected movie and every other movie

```python
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
```
**Line 11**: Sort movies by similarity and get top 5
- `enumerate(distances)`: Creates pairs of (index, similarity_score)
- `list(...)`: Converts to list
- `sorted(..., reverse=True, key=lambda x:x[1])`: Sorts by similarity score (second element) in descending order
- `[1:6]`: Skips index 0 (the movie itself) and takes next 5 movies (indices 1-5)

```python
    recommended_movies=[]
```
**Line 12**: Initialize empty list for recommendations

```python
    for i in movies_list:
        # fetch the movie poster
        recommended_movies.append(movies.iloc[i[0]].title)
```
**Lines 13-15**: Extract movie titles
- Loops through top 5 similar movies
- `i[0]` is the movie index from the sorted list
- `movies.iloc[i[0]].title` gets the title of the movie at that index
- Appends title to recommendations list

```python
    return recommended_movies
```
**Line 17**: Return the list of 5 recommended movie titles

```python
movies_dict= pickle.load(open('movie_dict.pkl','rb'))
```
**Line 21**: Load movie dictionary from pickle file
- Opens `movie_dict.pkl` in read-binary mode
- Loads the dictionary containing movie data

```python
movies=pd.DataFrame(movies_dict)
```
**Line 22**: Convert dictionary to DataFrame
- Creates a pandas DataFrame for easier manipulation
- Each key becomes a column, each value becomes a row

```python
similarity = pickle.load(open('similarity.pkl','rb'))
```
**Line 23**: Load pre-computed similarity matrix
- Loads the similarity matrix created by `build_similarity.py`
- This is a 2D NumPy array where each cell represents similarity between two movies

```python
st.header('MovieFlick: Movie Recommender System')
```
**Line 24**: Display main heading
- Creates a header in the Streamlit web interface

```python
movie_list = movies['title'].values
```
**Line 27**: Extract all movie titles
- Gets all unique movie titles as a NumPy array
- This will be used for the dropdown menu

```python
selected_movie_name=st.selectbox(
    'Type or select a movie from the dropdown',movie_list)
```
**Lines 28-29**: Create dropdown selector
- `st.selectbox()`: Creates a dropdown menu in Streamlit
- First argument: Label text displayed above the dropdown
- Second argument: List of options (all movie titles)
- User's selection is stored in `selected_movie_name`

```python
if st.button('Show Recommendation'):
```
**Line 31**: Create recommendation button
- `st.button()`: Creates a clickable button
- Code inside the `if` block runs only when button is clicked

```python
   recommendations=recommend(selected_movie_name)
```
**Line 32**: Get recommendations
- Calls the `recommend()` function with selected movie
- Stores the list of 5 recommended movies

```python
   for i in recommendations:
       st.write(i)
```
**Lines 33-34**: Display recommendations
- Loops through each recommended movie
- `st.write()`: Displays text in the Streamlit interface
- Each recommendation appears as a separate line

## 🚀 Running the Project

### Step 1: Build the Similarity Matrix (First Time Only)

If `similarity.pkl` doesn't exist, you need to build it first:

```bash
python build_similarity.py
```

This will:
- Load movie data from `movies.pkl` or `movie_dict.pkl`
- Create TF-IDF vectors from movie tags
- Compute cosine similarity matrix
- Save to `similarity.pkl`

**Expected Output:**
```
Created similarity.pkl with shape (4806, 4806)
```

### Step 2: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will:
- Start a local web server
- Open automatically in your default browser
- Display at `http://localhost:8501`

## 💡 Usage

1. **Select a Movie**: Use the dropdown menu to select a movie you like
2. **Get Recommendations**: Click the "Show Recommendation" button
3. **View Results**: The system will display 5 movies similar to your selection

## 🔍 How Recommendations Work

The system uses **Cosine Similarity** with **TF-IDF Vectorization**:

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**:
   - Converts movie tags/descriptions into numerical vectors
   - Weights words by how important they are (common words get lower weights)
   - Creates a mathematical representation of each movie's content

2. **Cosine Similarity**:
   - Measures the angle between two movie vectors
   - Values close to 1 = very similar movies
   - Values close to 0 = dissimilar movies
   - More accurate than simple keyword matching

3. **Recommendation Process**:
   - Finds movies with highest similarity scores
   - Excludes the selected movie itself
   - Returns top 5 matches

## 📊 Example

If you select **"The Dark Knight"**, the system might recommend:
- Batman Begins
- The Dark Knight Rises
- Inception
- Interstellar
- The Prestige

These recommendations are based on similar tags, genres, and content features.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📄 License

This project is open source and available for educational purposes.

---

**Note**: Make sure `similarity.pkl` exists before running `app.py`. If it doesn't exist, run `build_similarity.py` first.

