import pandas as pd
from flask import Flask, request, jsonify
import json, os, requests, spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity




app = Flask(__name__)
nlp = spacy.load("en_core_web_md")

serpapi_key = os.getenv("SERPAPI_KEY")






stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def calculate_similarity(claim_embedding, article_embedding):
    """Calculate cosine similarity between claim and article embeddings."""
    return cosine_similarity([claim_embedding], [article_embedding])[0][0]


def extract_entities(text):
    """Extract entities using custom Facebook NER model."""
    doc = nlp(text)
    return {ent.text.lower() for ent in doc.ents}
def get_embeddings(text):
    """Generate sentence embeddings."""
    return semantic_model.encode(text)


def get_news_from_serpapi(query):
    """Fetch news articles using SerpAPI."""
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "tbm": "nws",
        "api_key": serpapi_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        return results.get("news_results", [])
    except requests.exceptions.RequestException:
        return []

def predict_remote(text):
    try:
        response = requests.post(
            "https://prediction-oq8s.onrender.com/predict",
            json={"text": text},  # match your Flask API
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}
    

@app.route('/text', methods=['POST'])
def text_news():
    """Process text, clean with Gemini, and fetch related news."""
    data = request.json
    text = data.get('text')


    detected_entities = extract_entities(text)
    print("🔍 Extracted Entities:", detected_entities)

    search_query = " ".join(detected_entities) if detected_entities else text
    print("🌎 Searching for news with query:", search_query)
        
    news_articles = get_news_from_serpapi(search_query)
    print(f"📰 Retrieved {len(news_articles)} articles.")
    prediction_json  = predict_remote(text)
    print("🔮 Raw prediction response:", prediction_json)
    # print("Prediction:", prediction)
    prediction = prediction_json.get("prediction", "")


    claim_embedding = get_embeddings(text)
    claim_entities = extract_entities(text)

    matched_articles = []
       


    for article in news_articles:
        article_text = article.get('content', article.get('snippet', ""))
        article_embedding = get_embeddings(article_text)
        similarity = calculate_similarity(claim_embedding, article_embedding)
        print("similarity: ", similarity)
        if similarity >= 0.6:
            matched_articles.append({
                **article,
                "similarity": round(float(similarity * 100)),
           
            })
    score = 0
    num_of_articles_found = len(matched_articles)
    if num_of_articles_found > 0:
        print(f"\n✅ Matched {num_of_articles_found} relevant articles.")
        for i, article in enumerate(matched_articles, 1):
            print(f"\n🔗 Article {i}: {article['title']}")
            print(f"📡 Source: {article['source']}")
            print(f"📈 Similarity Score: {article['similarity']:.2f}")
            print(f"🌐 {article['link']}")
          

    else:
        print("⚠️ No matched articles found.")     
    if prediction.lower() == "credible":
        score += 100
        print("prediction +50")



    print(f"Score: {score}")

    return jsonify({
        "isClaim": True,
        "text": text,
        "matched_articles": matched_articles,
        "prediction": prediction,
        "score": score,
        # "total_ave_rounded": total,


    })
   

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)