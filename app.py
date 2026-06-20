"""
app.py  —  Flask backend for Satellite Anomaly Detection
"""

import os, ast, io, base64, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, confusion_matrix, precision_recall_curve,
    average_precision_score,
)
import torch
from torch.utils.data import TensorDataset, DataLoader
from flask import Flask, render_template, request, jsonify
from model import PatchTST

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
TRAIN_DIR  = "data/train"
TEST_DIR   = "data/test"
LABEL_CSV  = "data/labeled_anomalies.csv"
MODEL_DIR  = "models"
RESULT_DIR = "results"
SEQ_LEN, PRED_LEN, PATCH_LEN, D_MODEL = 96, 24, 16, 96

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load labels once at startup ────────────────────────────────────────────
labels_df = pd.read_csv(LABEL_CSV, header=None)
anomaly_map = {}
for _, row in labels_df.iloc[1:].iterrows():
    cid = str(row[0]).strip()
    try:
        seqs = ast.literal_eval(row[2])
        anomaly_map[cid] = seqs if isinstance(seqs, list) else []
    except Exception:
        anomaly_map[cid] = []

# ── Helpers ────────────────────────────────────────────────────────────────
def point_labels_fn(cid, length):
    lbl = np.zeros(length, dtype=int)
    for s, e in anomaly_map.get(cid, []):
        lbl[max(0,s):min(length,e)] = 1
    return lbl

def make_windows(series, stride=1):
    total = SEQ_LEN + PRED_LEN
    X, Y = [], []
    for i in range(0, len(series) - total + 1, stride):
        X.append(series[i:i+SEQ_LEN])
        Y.append(series[i+SEQ_LEN:i+total])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def window_labels_fn(pt_lbl, stride=1):
    total = SEQ_LEN + PRED_LEN
    return np.array([
        int(pt_lbl[i+SEQ_LEN:i+total].any())
        for i in range(0, len(pt_lbl) - total + 1, stride)
    ], dtype=int)

def load_model(cid):
    path = os.path.join(MODEL_DIR, f"{cid}.pt")
    if not os.path.exists(path): return None
    m = PatchTST(seq_len=SEQ_LEN, pred_len=PRED_LEN,
                 patch_len=PATCH_LEN, d_model=D_MODEL).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

def get_scores(model, X, Y, batch=256):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    ld = DataLoader(ds, batch_size=batch, shuffle=False)
    scores = []
    with torch.no_grad():
        for xb, yb in ld:
            err = (model(xb.to(device)) - yb.to(device)).abs()
            scores.extend(err.max(dim=1)[0].cpu().numpy())
    return np.array(scores)

def best_threshold(scores, labels):
    prec, rec, thr = precision_recall_curve(labels, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    idx = int(np.argmax(f1))
    return float(thr[idx]) if idx < len(thr) else float(thr[-1])

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64

def get_channel_list():
    models = glob.glob(os.path.join(MODEL_DIR, "*.pt"))
    return sorted([os.path.basename(f).replace(".pt","") for f in models])

# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    channels = get_channel_list()
    return render_template("index.html", channels=channels)

@app.route("/api/dashboard")
def api_dashboard():
    path = os.path.join(RESULT_DIR, "all_results.csv")
    if not os.path.exists(path):
        return jsonify({"error": "Run pipeline.py first"}), 404
    df = pd.read_csv(path)
    summary = {
        "total":    int(len(df)),
        "mean_auc": round(float(df["auc_roc"].mean()), 4),
        "mean_f1":  round(float(df["f1"].mean()), 4),
        "mean_acc": round(float(df["accuracy"].mean()), 4),
        "high_auc": int((df["auc_roc"] >= 0.90).sum()),
        "high_f1":  int((df["f1"] >= 0.70).sum()),
    }
    records = df[["series","auc_roc","auc_pr","f1","accuracy"]].round(4).to_dict("records")
    return jsonify({"summary": summary, "records": records})

@app.route("/api/channel/<cid>")
def api_channel(cid):
    cid = cid.upper()
    train_path = os.path.join(TRAIN_DIR, f"{cid}.npy")
    test_path  = os.path.join(TEST_DIR,  f"{cid}.npy")
    if not os.path.exists(train_path):
        return jsonify({"error": f"Channel {cid} not found"}), 404
    model = load_model(cid)
    if model is None:
        return jsonify({"error": f"No model for {cid}"}), 404

    train_raw = np.load(train_path)
    test_raw  = np.load(test_path)
    scaler    = StandardScaler().fit(train_raw[:, 0:1])
    te_series = scaler.transform(test_raw[:, 0:1]).ravel()
    X, Y      = make_windows(te_series)
    pt_lbl    = point_labels_fn(cid, len(te_series))
    y_lbl     = window_labels_fn(pt_lbl)

    if y_lbl.sum() == 0:
        return jsonify({"error": f"No anomalies in test set for {cid}"}), 200

    scores = get_scores(model, X, Y)
    thr    = best_threshold(scores, y_lbl)
    preds  = (scores >= thr).astype(int)

    auc = float(roc_auc_score(y_lbl, scores))
    f1  = float(f1_score(y_lbl, preds, zero_division=0))
    pre = float(precision_score(y_lbl, preds, zero_division=0))
    rec = float(recall_score(y_lbl, preds, zero_division=0))
    ap  = float(average_precision_score(y_lbl, scores))
    cm  = confusion_matrix(y_lbl, preds).tolist()

    # ── Plot 1: Anomaly Score Timeline ────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(scores, lw=1.2, color="#3B82F6", label="Anomaly Score")
    ax.axhline(thr, color="#EF4444", linestyle="--", lw=2,
               label=f"Threshold={thr:.3f}")
    ax.fill_between(range(len(scores)), 0, scores.max(),
                    where=(y_lbl==1), alpha=0.25, color="#EF4444",
                    label="True Anomaly")
    ax.fill_between(range(len(scores)), 0, scores.max(),
                    where=(preds==1), alpha=0.15, color="#F59E0B",
                    label="Predicted")
    ax.set_title(f"Channel {cid} — Anomaly Scores", fontweight="bold")
    ax.set_xlabel("Window Index"); ax.set_ylabel("Score")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.patch.set_facecolor("#0F172A"); ax.set_facecolor("#1E293B")
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white"); ax.title.set_color("white")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    ax.legend(facecolor="#1E293B", labelcolor="white", fontsize=9)
    timeline_b64 = fig_to_b64(fig)

    # ── Plot 2: ROC Curve ─────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_lbl, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2.5, color="#10B981", label=f"AUC={auc:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#10B981")
    ax.plot([0,1],[0,1],"--", color="#64748B", lw=1)
    ax.set_title("ROC Curve", fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(); ax.grid(alpha=0.3)
    fig.patch.set_facecolor("#0F172A"); ax.set_facecolor("#1E293B")
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_color("white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1E293B", labelcolor="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    roc_b64 = fig_to_b64(fig)

    # ── Plot 3: Confusion Matrix ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(confusion_matrix(y_lbl, preds), annot=True, fmt="d",
                cmap="Blues", ax=ax, cbar=False,
                xticklabels=["Normal","Anomaly"],
                yticklabels=["Normal","Anomaly"])
    ax.set_title("Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.patch.set_facecolor("#0F172A"); ax.set_facecolor("#1E293B")
    ax.title.set_color("white"); ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white"); ax.tick_params(colors="white")
    cm_b64 = fig_to_b64(fig)

    return jsonify({
        "channel": cid,
        "metrics": {
            "auc_roc":   round(auc, 4),
            "auc_pr":    round(ap,  4),
            "f1":        round(f1,  4),
            "precision": round(pre, 4),
            "recall":    round(rec, 4),
            "threshold": round(thr, 4),
            "n_windows": int(len(scores)),
            "n_anomaly": int(y_lbl.sum()),
        },
        "plots": {
            "timeline": timeline_b64,
            "roc":      roc_b64,
            "cm":       cm_b64,
        }
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    cid   = request.form.get("channel", "").upper()
    model = load_model(cid)
    if model is None:
        return jsonify({"error": f"No model for {cid}"}), 404

    file = request.files["file"]
    try:
        data = np.load(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    series = (data[:, 0] if data.ndim > 1 else data).astype(np.float32)
    series = (series - series.mean()) / (series.std() + 1e-8)

    if len(series) < SEQ_LEN + PRED_LEN:
        return jsonify({"error": f"Need ≥ {SEQ_LEN+PRED_LEN} time steps"}), 400

    X, Y   = make_windows(series)
    scores = get_scores(model, X, Y)
    thr    = float(np.percentile(scores, 95))
    preds  = (scores >= thr).astype(int)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(scores, lw=1.2, color="#3B82F6", label="Anomaly Score")
    ax.axhline(thr, color="#EF4444", linestyle="--", lw=2,
               label=f"Threshold={thr:.3f}")
    ax.fill_between(range(len(scores)), 0, scores.max(),
                    where=(preds==1), alpha=0.25, color="#F59E0B",
                    label="Predicted Anomaly")
    ax.set_title(f"Uploaded File — {cid} Model", fontweight="bold")
    ax.set_xlabel("Window"); ax.set_ylabel("Score")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.patch.set_facecolor("#0F172A"); ax.set_facecolor("#1E293B")
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white"); ax.title.set_color("white")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    ax.legend(facecolor="#1E293B", labelcolor="white", fontsize=9)
    plot_b64 = fig_to_b64(fig)

    top_idx = np.argsort(scores)[::-1][:10].tolist()
    return jsonify({
        "n_windows":  int(len(scores)),
        "n_anomaly":  int(preds.sum()),
        "pct_anomaly": round(float(preds.mean()*100), 2),
        "threshold":  round(thr, 5),
        "max_score":  round(float(scores.max()), 5),
        "top_windows": [{"index": i, "score": round(float(scores[i]),5)}
                        for i in top_idx],
        "plot": plot_b64,
    })

@app.route("/api/channels")
def api_channels():
    return jsonify({"channels": get_channel_list()})

if __name__ == "__main__":
    app.run(debug=True, port=5000)