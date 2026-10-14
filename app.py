"""
University Placement Prediction System - Flask Web Application
Provides end-to-end ML lifecycle serving, dataset profiling, interactive visualizations (M1-M6),
model accuracy comparisons, dedicated model output views, and live student placement inference.
"""
import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import ml_engine

app = Flask(__name__)
app.secret_key = "placement-prediction-system-key-2026"

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
UPLOAD_FILE_PATH = os.path.join(UPLOAD_FOLDER, "uploaded_dataset.csv")

@app.route("/")
def home():
    models_summary = ml_engine.load_models_summary()
    return render_template("home.html", models_summary=models_summary)

@app.route("/dataset", methods=["GET", "POST"])
def dataset():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "reset":
            if os.path.exists(UPLOAD_FILE_PATH):
                try:
                    os.remove(UPLOAD_FILE_PATH)
                    flash("Reset to default 50,000 university benchmark dataset successfully.", "success")
                except Exception as e:
                    flash(f"Error resetting dataset: {e}", "danger")
            return redirect(url_for("dataset"))

        if action == "upload":
            file = request.files.get("dataset_file")
            if not file or file.filename == "":
                flash("Please choose a valid CSV file to upload.", "danger")
                return redirect(url_for("dataset"))

            if not file.filename.lower().endswith(".csv"):
                flash("Unsupported file format. Please upload a .csv file.", "danger")
                return redirect(url_for("dataset"))

            try:
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                file.save(UPLOAD_FILE_PATH)
                flash(f"Dataset '{file.filename}' uploaded and active successfully!", "success")
            except Exception as e:
                flash(f"Failed to process uploaded file: {e}", "danger")
            return redirect(url_for("dataset"))

    profile = ml_engine.get_dataset_profile()
    if profile is None:
        flash("Could not locate default dataset. Please verify dataset/placement_predict_50K_Raw.csv.", "warning")
        profile = {
            'filename': 'None',
            'is_custom_upload': False,
            'total_rows': 0,
            'total_cols': 0,
            'missing_total': 0,
            'duplicate_total': 0,
            'target_dist': {},
            'columns': [],
            'column_summaries': [],
            'preview_data': []
        }
    return render_template("dataset.html", profile=profile)

@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html")

@app.route("/visualization")
def visualization():
    return render_template("visualization.html")

@app.route("/models")
def models():
    models_summary = ml_engine.load_models_summary()
    return render_template("models.html", models_summary=models_summary)

@app.route("/models/<model_id>")
def model_detail(model_id):
    models_summary = ml_engine.load_models_summary()
    if model_id not in models_summary:
        flash(f"Model '{model_id}' not found in registry.", "danger")
        return redirect(url_for("models"))
    model = models_summary[model_id]
    return render_template("model_detail.html", model=model)

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    models_summary = ml_engine.load_models_summary()
    selected_model = request.args.get("model", "gradient_boosting")
    result = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()
        selected_model = form_data.get("model_id", "gradient_boosting")
        try:
            result = ml_engine.predict_student(form_data, model_id=selected_model)
            flash(f"Prediction generated successfully using {result['model_name']}!", "success")
        except Exception as e:
            flash(f"Inference error: {e}", "danger")

    return render_template(
        "prediction.html",
        models_summary=models_summary,
        selected_model=selected_model,
        result=result,
        form_data=form_data
    )

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "clear":
            ml_engine.clear_prediction_history()
            flash("Prediction audit telemetry cleared.", "success")
            return redirect(url_for("dashboard"))

    analytics = ml_engine.get_dashboard_analytics()
    return render_template("dashboard.html", analytics=analytics)

@app.route("/reports")
def reports():
    return render_template("reports.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# ============================================================
# REST API Endpoints for Integration & Benchmarking
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True, silent=True) or request.form.to_dict()
    model_id = payload.get("model_id", "gradient_boosting")
    try:
        res = ml_engine.predict_student(payload, model_id=model_id)
        return jsonify({
            "status": "success",
            "prediction": res["status"],
            "is_placed": res["is_placed"],
            "probability_pct": res["probability"],
            "salary_package": res["salary_package"],
            "model_used": res["model_name"]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/models", methods=["GET"])
def api_models():
    summary = ml_engine.load_models_summary()
    return jsonify(summary)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 PlacementPredict server running at http://127.0.0.1:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)
