import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.parse import parse_qs
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)

# Sentiment analyzer initialization
sia = SentimentIntensityAnalyzer()

# Load reviews from CSV file
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

# Valid locations list
valid_locations = [
    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
    "El Paso, Texas", "Escondido, California", "Fresno, California",
    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
    "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
    "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            return self.handle_get(environ, start_response)
        elif environ["REQUEST_METHOD"] == "POST":
            return self.handle_post(environ, start_response)
        else:
            start_response("405 Method Not Allowed", [("Content-Type", "application/json")])
            return [json.dumps({"error": "Method not allowed"}).encode("utf-8")]

    def handle_get(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        # Parse query parameters
        query_params = parse_qs(environ.get("QUERY_STRING", ""))
        location_filter = query_params.get("location", [None])[0]
        start_date_filter = query_params.get("start_date", [None])[0]
        end_date_filter = query_params.get("end_date", [None])[0]

        # Filter reviews based on query parameters
        filtered_reviews = reviews
        if location_filter:
            filtered_reviews = [
                review for review in filtered_reviews
                if review["Location"] == location_filter
            ]
        if start_date_filter:
            filtered_reviews = [
                review for review in filtered_reviews
                if review["Timestamp"] >= start_date_filter
            ]
        if end_date_filter:
            filtered_reviews = [
                review for review in filtered_reviews
                if review["Timestamp"] <= end_date_filter
            ]

        # Perform sentiment analysis and add sentiment data to reviews
        for review in filtered_reviews:
            review["sentiment"] = self.analyze_sentiment(review["ReviewBody"])

        # Sort reviews by compound sentiment score in descending order
        sorted_reviews = sorted(
            filtered_reviews,
            key=lambda x: x["sentiment"]["compound"],
            reverse=True,
        )

        # Convert the reviews to a JSON byte string
        response_body = json.dumps(sorted_reviews, indent=2).encode("utf-8")
        start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
        ])
        return [response_body]

    def handle_post(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        try:
            # Get the size of the request body
            request_body_size = int(environ.get('CONTENT_LENGTH', 0))
            request_body = environ['wsgi.input'].read(request_body_size)
            post_params = parse_qs(request_body.decode('utf-8'))

            # Extract and validate required parameters
            location = post_params.get("Location", [None])[0]
            review_body = post_params.get("ReviewBody", [None])[0]

            if not location or not review_body:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [json.dumps({"error": "Location and ReviewBody are required."}).encode("utf-8")]

            if location not in valid_locations:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [json.dumps({"error": "Invalid location."}).encode("utf-8")]

            # Create a new review entry
            new_review = {
                "ReviewId": str(uuid.uuid4()),
                "Location": location,
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "ReviewBody": review_body,
            }
            reviews.append(new_review)

            # Return the new review as a JSON response
            response_body = json.dumps(new_review, indent=2).encode("utf-8")
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        except Exception as e:
            start_response("500 Internal Server Error", [("Content-Type", "application/json")])
            return [json.dumps({"error": str(e)}).encode("utf-8")]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
