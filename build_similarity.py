import os
import pickle
import pandas as pd

def build_similarity():
    """Builds a cosine similarity matrix from the `tags` column and saves it to `similarity.pkl`."""
    # Prefer a DataFrame pickle if present
    if os.path.exists('movies.pkl'):
        with open('movies.pkl', 'rb') as f:
            movies = pickle.load(f)
    elif os.path.exists('movie_dict.pkl'):
        with open('movie_dict.pkl', 'rb') as f:
            movies = pd.DataFrame(pickle.load(f))
    else:
        raise FileNotFoundError('Neither movies.pkl nor movie_dict.pkl found in the workspace')

    if not hasattr(movies, 'columns') or 'tags' not in movies.columns:
        raise ValueError('The movies data must contain a "tags" column')

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as e:
        raise ImportError('scikit-learn is required to build the similarity matrix. Install it with "pip install scikit-learn"') from e

    # Ensure tags are strings
    tags = movies['tags'].astype(str).values
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(tags)
    similarity = cosine_similarity(vectors)

    with open('similarity.pkl', 'wb') as f:
        pickle.dump(similarity, f)

    print('Created similarity.pkl with shape', getattr(similarity, 'shape', None))


if __name__ == '__main__':
    build_similarity()
