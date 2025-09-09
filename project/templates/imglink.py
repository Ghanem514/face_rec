import os
import pickle
import numpy as np
import json
from flask import Flask, render_template, request, Response
import cv2
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import csv
import smtplib
from email.message import EmailMessage

app = Flask(__name__)

STATIC_IMAGE_FOLDER = os.path.join("static", "face_images")
EMBEDDINGS_FILE = "embeddingst1.pkl"
CSV_FILE = "user_data.csv"
EMAIL_SENDER = "ghanemgh43@gmail.com"
EMAIL_PASSWORD = "hngi ixss ickr jnga"
TOP_K = 10

# --- Helper to normalize filenames (fix underscores/dashes) ---
def normalize_name(name):
    return name.replace("_", "-").lower()

# Load face embeddings
with open(EMBEDDINGS_FILE, "rb") as f:
    data = pickle.load(f)
embedding_list = data["embeddings"]
name_list = data["filenames"]

# Load image links
with open("image_links.json", "r") as f:
    image_url_map = json.load(f)

# Normalize image_url_map keys so lookups always work
normalized_image_url_map = {normalize_name(k): v for k, v in image_url_map.items()}

# FaceAnalysis
face_app = FaceAnalysis(name='antelopev2')
face_app.prepare(ctx_id=0)

camera = cv2.VideoCapture(0)

def save_user_data(name, phone, email):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Name', 'Phone', 'Email'])
        writer.writerow([name, phone, email])

def send_email_with_links(receiver_email, selected_filenames):
    msg = EmailMessage()
    msg["Subject"] = "Your Selected Face Matches"
    msg["From"] = EMAIL_SENDER
    msg["To"] = receiver_email

    html_content = "<h2>Here are the matched faces you selected:</h2><ul>"

    for filename in selected_filenames:
        normalized_filename = normalize_name(filename)
        image_url = normalized_image_url_map.get(normalized_filename, "#")
        html_content += f"<li><b>{filename}</b><br><a href='{image_url}' target='_blank'>View Image</a></li><br>"

    html_content += "</ul>"
    msg.add_alternative(html_content, subtype='html')

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print(f"✅ Email sent to {receiver_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

@app.route("/")
def index():
    return render_template("index.html")

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        cv2.putText(frame, "Scanning...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/recognize", methods=["POST"])
def recognize():
    user_name = request.form.get("name")
    user_phone = request.form.get("phone")
    user_email = request.form.get("email")

    save_user_data(user_name, user_phone, user_email)

    success, frame = camera.read()
    if not success:
        return "Failed to capture image."

    faces = face_app.get(frame)
    if not faces:
        return "No face detected."

    matches = []

    for face in faces:
        face_embedding = face.embedding
        similarities = cosine_similarity([face_embedding], embedding_list)[0]

        for idx, score in enumerate(similarities):
            if score >= 0.60:
                filename = os.path.basename(name_list[idx])
                normalized_filename = normalize_name(filename)
                if normalized_filename in normalized_image_url_map:
                    matches.append({
                        "name": filename,
                        "score": round(float(score), 2),
                        "email": user_email
                    })

    if matches:
        matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:TOP_K]
        return render_template("matches.html", matches=matches, email=user_email)
    else:
        return "No match found."

@app.route("/send_selected", methods=["POST"])
def send_selected():
    selected = request.form.getlist("selected_matches")
    user_email = request.form.get("email")
    if selected and user_email:
        send_email_with_links(user_email, selected)
        return f"✅ Email sent to {user_email} with {len(selected)} selected matches."
    return "❌ No matches selected or missing email."

if __name__ == "__main__":
    app.run(debug=True)
