from flask import Flask, render_template, request
from recommender.engine import MovieRecommender
from dotenv import load_dotenv
import os
import requests
from functools import lru_cache
import time

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = Flask(__name__)

# Initialize recommender with error handling
try:
    recommender = MovieRecommender('data/tmdb_5000_movies.csv', 'data/tmdb_5000_credits.csv')
except Exception as e:
    print(f"Failed to initialize recommender: {e}")
    raise

# Cache posters to reduce API calls (expires after 1 hour)
@lru_cache(maxsize=500)
def fetch_poster(movie_title):
    try:
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": movie_title,
            "include_adult": "false"
        }
        
        # Add delay to avoid hitting rate limits
        time.sleep(0.2)  # TMDB allows ~50 requests/second
        
        response = requests.get(search_url, params=params, timeout=5)
        response.raise_for_status()
        
        results = response.json().get("results", [])
        if results:
            # Find the most popular match
            top_result = max(results, key=lambda x: x.get('popularity', 0))
            poster_path = top_result.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster for {movie_title}: {e}")
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        # Cache movie list to avoid repeated sorting
        if not hasattr(index, 'movies'):
            index.movies = sorted(recommender.movies['title'].tolist())
        
        if request.method == 'POST':
            movie_title = request.form.get('movie', '').strip()
            if not movie_title:
                return render_template('index.html', 
                                   movies=index.movies,
                                   error="Please enter a movie title")
            
            recommendations, error = recommender.recommend(movie_title)
            if error:
                return render_template('index.html',
                                    movies=index.movies,
                                    error=error)
            
            # Fetch posters in parallel would be better for production
            rec_with_posters = []
            for title in recommendations:
                poster = fetch_poster(title)
                rec_with_posters.append({
                    "title": title,
                    "poster": poster or "/static/placeholder.jpg"  # Fallback image
                })

            return render_template('recommendations.html',
                                movie=movie_title,
                                recommendations=rec_with_posters)

        return render_template('index.html', movies=index.movies)

    except Exception as e:
        print(f"Error in index route: {e}")
        return render_template('error.html'), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)