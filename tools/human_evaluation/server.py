import os
import json
from flask import Flask, request, jsonify, send_from_directory
import time
from flask_cors import CORS

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

RESULTS_FILE = "survey_results.jsonl"
RESULTS_SINGLE_FILE = "survey_submit_{}.json"


@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")


@app.route("/submit", methods=["POST"])
def handle_submission():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    submission_data = request.get_json()

    try:
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(submission_data, ensure_ascii=False) + "\n")
        with open(RESULTS_SINGLE_FILE.format(submission_data["userId"]), "w") as f:
            json.dump(submission_data, f, indent=2)

        print(
            f"Successfully saved submission from userId: {submission_data.get('userId')}"
        )
        return jsonify({"status": "success", "message": "Submission received"}), 200
    except Exception as e:
        print(f"Error saving submission: {e}")
        return jsonify({"status": "error", "message": "Could not save submission"}), 500


if __name__ == "__main__":
    # For production, use a professional WSGI server like Gunicorn or uWSGI
    app.run(host="0.0.0.0", port=5000, debug=True)
