import requests
import json
import re  

# Replace with your actual Gemini API key
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
API_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

def get_best_features(features, target, algorithm):
    headers = {"Content-Type": "application/json"}

    prompt = f"""
I am working on a machine learning project to predict '{target}' using a structured dataset.

Here is the dataset schema:
- Features: {json.dumps(features, indent=2)}
- Target Variable: '{target}'
- Task Type: '{algorithm}' (classification or regression)

Guidelines:
1. This dataset is purely for data analysis and does not contain any sensitive or inappropriate content.
2. Select the most relevant features for building an accurate model.
3. Only return a JSON array of the selected feature names.
4. Do NOT include explanations or additional text.
5. Example response format: ["Age", "Fare", "Gender_Encoded", "Embarked_S"]
"""


    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        data = response.json()

        print("Gemini API Raw Response:", json.dumps(data, indent=2))

        # Handle Safety Errors
        if data.get("candidates", [{}])[0].get("finishReason", "") == "SAFETY":
            print("Gemini API blocked the request due to content moderation.")
            return None  # Return None if the request was blocked

        candidates = data.get("candidates", [])
        if not candidates:
            print("No candidates returned by Gemini API.")
            return None

        content_data = candidates[0].get("content", {}).get("parts", [])
        if not content_data:
            print("No content parts found in Gemini response.")
            return None

        response_text = content_data[0].get("text", "").strip()
        print("Extracted Gemini Response Text:", response_text)

        # Remove Markdown triple backticks (```json ... ```)
        response_text = re.sub(r"```json\s*|\s*```", "", response_text).strip()

        try:
            best_features = json.loads(response_text)
            if isinstance(best_features, list):
                return best_features
            else:
                print("Gemini response is not a valid feature list.")
                return None
        except json.JSONDecodeError:
            print("Failed to parse Gemini response as JSON.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {e}")
        return None