import os
import pickle
import numpy as np
import json
import base64
from flask import Flask, render_template, request, session, redirect, url_for
import cv2
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.message import EmailMessage
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # Needed for session

# --- Directories and files ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_FILE_1 = os.path.join(BASE_DIR, "embhack.pkl")
EMBEDDINGS_FILE_2 = os.path.join(BASE_DIR, "embhack1.pkl")
EMBEDDINGS_FILE_3 = os.path.join(BASE_DIR, "embhack2.pkl")
EMBEDDINGS_FILE_4 = os.path.join(BASE_DIR, "embhack3.pkl")
EMBEDDINGS_FILE_5 = os.path.join(BASE_DIR, "embhack4.pkl")
EMBEDDINGS_FILE_6 = os.path.join(BASE_DIR, "embhack5.pkl")

IMAGE_LINKS_FILE = os.path.join(BASE_DIR, "image_links.json")

EMAIL_SENDER = "techarabi717@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TOP_K = 100

# --- MongoDB connection ---

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["face_rec"]
users_collection = db["user"]

# --- Helper to normalize filenames ---
def normalize_name(name):
    return name.replace("_", "-").lower()

# --- Load embeddings ---
embedding_list = []
name_list = []

EMBEDDINGS_FILES = [
    EMBEDDINGS_FILE_1,
    EMBEDDINGS_FILE_2,
    EMBEDDINGS_FILE_3,
    EMBEDDINGS_FILE_4,
    EMBEDDINGS_FILE_5,
    EMBEDDINGS_FILE_6, 
]

for file in EMBEDDINGS_FILES:
    if os.path.exists(file):
        with open(file, "rb") as f:
            data = pickle.load(f)
            embedding_list.extend(data["embeddings"])
            name_list.extend(data["filenames"])
            print(f"📂 Loaded {len(data['embeddings'])} embeddings from {os.path.basename(file)}")
    else:
        print(f"⚠️ File not found: {file}")

print(f"✅ Total loaded: {len(embedding_list)} embeddings from {len(name_list)} images")

# --- Load image links ---
with open(IMAGE_LINKS_FILE, "r") as f:
    image_url_map = json.load(f)
normalized_image_url_map = {normalize_name(k): v for k, v in image_url_map.items()}

# --- Initialize FaceAnalysis ---
face_app = FaceAnalysis(name='antelopev2')
face_app.prepare(ctx_id=0)

# --- MongoDB helpers ---
def save_user_with_matches(name, phone, email, matches):
    """
    Save user info and matched images into MongoDB using URLs from JSON.
    """
    matches_with_urls = []
    for m in matches:
        normalized_filename = normalize_name(m['name'])
        image_url = normalized_image_url_map.get(normalized_filename)
        if image_url:
            matches_with_urls.append({
                "name": m["name"],
                "score": m["score"],
                "url": image_url
            })

    user_doc = {
        "name": name,
        "phone": phone,
        "email": email,
        "matched_images": [m["url"] for m in matches_with_urls],  # save URLs
        "sent_images": []
    }
    result = users_collection.insert_one(user_doc)
    print(f"✅ Saved user {name} with id {result.inserted_id}")
    return result.inserted_id

def update_sent_images(user_id, sent_images):
    urls = [normalized_image_url_map.get(normalize_name(f), f) for f in sent_images]
    users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"sent_images": urls}}
    )

# --- Email ---
def send_email_with_links(receiver_email, selected_filenames):
    msg = EmailMessage()
    msg["Subject"] = "Your Face Recognition Matches"
    msg["From"] = EMAIL_SENDER
    msg["To"] = receiver_email

    html_content = "<h2>Hello,\n\nHere are the face matches you selected:\n</h2><ul>"
    for filename in selected_filenames:
        normalized_filename = normalize_name(filename)
        image_url = normalized_image_url_map.get(normalized_filename, "#")
        html_content += f"<li><b>{filename}</b>: <a href='{image_url}' target='_blank'>View Image</a></li><br>"
    html_content += "</ul>"

    msg.add_alternative(html_content, subtype='html')

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"✅ Email sent to {receiver_email}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

# --- Language switching ---
@app.route("/switch_lang/<lang>")
def switch_lang(lang):
    if lang in ["en", "ar"]:
        session['lang'] = lang
    return redirect(request.referrer or url_for('index'))

def get_lang():
    return session.get('lang', 'en')

# --- Routes ---
@app.route("/")
def index():
    lang = get_lang()
    return render_template("index.html", lang=lang)

@app.route("/recognize", methods=["POST"])
def recognize():
    import traceback
    lang = get_lang()
    user_name = request.form.get("name", "").strip()
    user_phone = request.form.get("phone", "").strip()
    user_email = request.form.get("email", "").strip()

    # Get image from form (base64 string)
    image_data = request.form.get("image") or request.form.get("captured_image")
    if not image_data or not image_data.startswith("data:image"):
        return render_template("no_matches.html", lang=lang, message="❌ No image received.")

    # --- 1) Decode base64 image ---
    try:
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return render_template("no_matches.html", lang=lang, message="❌ Failed to decode image.")
    except Exception as e:
        tb = traceback.format_exc()
        return f"<h2>Error decoding image</h2><pre>{tb}</pre>", 400

    # --- 2) Detect faces ---
    faces = face_app.get(frame)
    if not faces:
        return render_template("no_matches.html", lang=lang, message="❌ No face detected.")

    # --- 3) Build embedding matrix ---
    if not embedding_list or len(embedding_list) == 0:
        return "<h2>No stored embeddings loaded.</h2>", 500

    try:
        embedding_matrix = np.vstack([np.asarray(e).reshape(1, -1) for e in embedding_list])
        embedding_matrix_norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_matrix_norms[embedding_matrix_norms == 0] = 1.0
        embedding_matrix_normalized = embedding_matrix / embedding_matrix_norms
    except Exception as e:
        tb = traceback.format_exc()
        return f"<h2>Error building embedding matrix</h2><pre>{tb}</pre>", 500

    # --- 4) Compare each detected face ---
    THRESHOLD = 0.60
    all_matches = []

    for fi, face in enumerate(faces):
        try:
            face_emb = np.asarray(face.embedding).reshape(1, -1)
            if face_emb.shape[1] != embedding_matrix_normalized.shape[1]:
                return f"<h2>Embedding dimension mismatch</h2><p>Detected: {face_emb.shape[1]}, Stored: {embedding_matrix_normalized.shape[1]}</p>", 500

            face_emb_norm = np.linalg.norm(face_emb)
            if face_emb_norm == 0:
                face_emb_norm = 1.0
            face_emb_normalized = face_emb / face_emb_norm

            # Cosine similarity
            sims = embedding_matrix_normalized.dot(face_emb_normalized.T).flatten()

            # Get indices where score >= threshold
            match_indices = np.where(sims >= THRESHOLD)[0]

            for idx in match_indices:
                raw_filename = os.path.basename(name_list[int(idx)])
                filename_no_ext = os.path.splitext(raw_filename)[0]
                normalized_filename_key = normalize_name(filename_no_ext)
                image_url = normalized_image_url_map.get(normalized_filename_key)

                all_matches.append({
                    "name": raw_filename,
                    "score": round(float(sims[int(idx)]), 2),  # 2-digit score
                    "url": image_url
                })
        except Exception as e:
            print(f"[WARN] Failed face {fi}: {e}")
            continue

    if all_matches:
        # Remove duplicates and sort by score
        unique = { (m["name"], m.get("url")): m for m in all_matches }
        matches_sorted = sorted(unique.values(), key=lambda x: x["score"], reverse=True)[:TOP_K]

        # Save user and session
        try:
            user_id = save_user_with_matches(user_name, user_phone, user_email, matches_sorted)
            session["user_id"] = str(user_id)
        except Exception as e:
            print(f"[WARN] save_user_with_matches failed: {e}")

        return render_template("matches.html", matches=matches_sorted, email=user_email, message=None, lang=lang)

    # No matches found
    return render_template("no_matches.html", lang=lang, message="❌ No matches found above threshold.")


@app.route("/send_selected", methods=["POST"])
def send_selected():
    lang = get_lang()
    selected = request.form.getlist("selected_matches")
    user_email = request.form.get("email")

    if selected and user_email:
        user_id = session.get("user_id")
        if user_id:
            update_sent_images(user_id, selected)

        success = send_email_with_links(user_email, selected)

        message = "✅ Email sent successfully!" if success else "❌ Failed to send email."
        if lang == "ar":
            message = "✅ تم إرسال البريد بنجاح!" if success else "❌ فشل إرسال البريد."
        return render_template("confirmation.html", message=message, lang=lang)
    else:
        return "❌ No matches selected or missing email."

# --- Main ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
