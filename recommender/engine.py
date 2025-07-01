import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

class MovieRecommender:
    def __init__(self, movies_path, credits_path):
        """Initialize the MovieRecommender with movie and credit data."""
        self.movies = pd.read_csv(movies_path)
        self.credits = pd.read_csv(credits_path)
        self.cv = None
        self.vectors = None
        self.similarity = None
        self.indices = None
        self._merge_and_prepare()

    def _merge_and_prepare(self):
        """Merge and preprocess the movie and credit data."""
        # Merge datasets on title
        self.movies = self.movies.merge(self.credits, on='title', how='left')

        # Keep only necessary columns
        self.movies = self.movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

        # Parse JSON-like columns
        for feature in ['genres', 'keywords', 'cast', 'crew']:
            self.movies[feature] = self.movies[feature].apply(self._parse_json)

        # Extract top 3 actors
        self.movies['cast'] = self.movies['cast'].apply(
            lambda x: [i['name'].strip().lower() for i in x[:3]] if isinstance(x, list) else [])

        # Extract director(s) from crew
        self.movies['crew'] = self.movies['crew'].apply(
            lambda x: [i['name'].strip().lower() for i in x if isinstance(i, dict) and i['job'] == 'Director'])

        # Convert genres & keywords to lists of names
        self.movies['genres'] = self.movies['genres'].apply(
            lambda x: [i['name'].strip().lower() for i in x] if isinstance(x, list) else [])
        self.movies['keywords'] = self.movies['keywords'].apply(
            lambda x: [i['name'].strip().lower() for i in x] if isinstance(x, list) else [])

        # Fill missing overviews with empty string
        self.movies['overview'] = self.movies['overview'].fillna('').str.strip().str.lower()

        # Create a 'tags' column with all relevant information
        self.movies['tags'] = self.movies.apply(
            lambda row: ' '.join(row['overview'].split()) + ' ' +
                       ' '.join(row['genres']) + ' ' +
                       ' '.join(row['keywords']) + ' ' +
                       ' '.join(row['cast']) + ' ' +
                       ' '.join(row['crew']),
            axis=1
        )

        # Clean up and drop unnecessary columns
        self.movies = self.movies[['movie_id', 'title', 'tags']].drop_duplicates(subset='title')

        # Vectorize the 'tags' text using TF-IDF for better results
        self.cv = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.vectors = self.cv.fit_transform(self.movies['tags'])
        self.similarity = cosine_similarity(self.vectors)

        # Create title index with fuzzy matching support
        self.indices = pd.Series(self.movies.index, index=self.movies['title'].str.lower()).drop_duplicates()

    def _parse_json(self, val):
        """Safely parse JSON-like strings into Python objects."""
        if pd.isna(val):
            return []
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []

    def _find_closest_title(self, title):
        """Find the closest matching title using fuzzy matching."""
        title = str(title).lower()
        if title in self.indices:
            return title
        
        # Use fuzzy matching to find the closest title
        matches = process.extractOne(title, self.indices.index)
        if matches and matches[1] > 80:  # similarity threshold
            return matches[0]
        return None

    def recommend(self, title, top_n=5):
        """
        Recommend similar movies based on title.
        
        Args:
            title (str): Movie title to find recommendations for
            top_n (int): Number of recommendations to return
            
        Returns:
            list: Recommended movie titles
            str: Error message if title not found (None if successful)
        """
        closest_title = self._find_closest_title(title)
        if not closest_title:
            return [], f"Movie '{title}' not found in database. Please check the title."

        idx = self.indices[closest_title]
        distances = sorted(
            list(enumerate(self.similarity[idx])),
            key=lambda x: x[1],
            reverse=True
        )[1:top_n + 1]  # Skip the first item (itself)
        
        movie_indices = [i[0] for i in distances]
        return self.movies['title'].iloc[movie_indices].tolist(), None