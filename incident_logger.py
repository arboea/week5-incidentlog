import json
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from lime.lime_text import LimeTextExplainer


def generate_sample_events(n=20, seed=42):
    random.seed(seed)
    base_time = datetime.now()
    locations = ["Warehouse A", "Assembly Line 1", "Loading Dock", "Office Area", "Rooftop"]
    templates = [
        "Worker slipped on wet floor and fell",
        "Forklift collided with pallet and knocked over materials",
        "Employee cut hand while operating box cutter",
        "Chemical spill near storage shelf",
        "Worker reported dizziness and shortness of breath",
        "Loose wiring caused small electrical spark",
        "Worker struck head on low beam",
        "Ladder slipped during maintenance task",
        "Fire alarm triggered by overheated motor",
        "Box fell from shelf hitting employee's foot",
    ]

    events = []
    for i in range(n):
        desc = random.choice(templates)
        # add modifiers to produce more varied text
        if random.random() < 0.3:
            desc += "; water on floor from leaking pipe"
        if random.random() < 0.2:
            desc += "; witnessed by multiple coworkers"
        if random.random() < 0.15:
            desc += "; employee transported to hospital"

        event = {
            "id": str(uuid.uuid4()),
            "timestamp": (base_time - timedelta(minutes=random.randint(0, 60 * 24))).isoformat(),
            "location": random.choice(locations),
            "description": desc,
            "witness_count": random.choice([0, 1, 2, 3, 5]),
            "injured": random.random() < 0.3,
        }
        events.append(event)
    return events


def label_severity(description, injured, witness_count):
    # simple heuristic to create labels for training
    desc = description.lower()
    if "hospital" in desc or injured or "dizziness" in desc or "fire" in desc or "chemical" in desc:
        return "critical"
    if "cut" in desc or "hit" in desc or witness_count >= 3:
        return "major"
    return "minor"


def train_text_classifier(events):
    texts = [e["description"] for e in events]
    labels = [label_severity(e["description"], e["injured"], e["witness_count"]) for e in events]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
    clf = LogisticRegression(max_iter=1000)
    pipeline = make_pipeline(vectorizer, clf)
    pipeline.fit(texts, labels)
    # derive class names from the trained classifier (final estimator)
    try:
        final_clf = pipeline.named_steps["logisticregression"]
        classes = final_clf.classes_.tolist()
    except Exception:
        # fallback: inspect predict_proba columns by making a dummy call
        classes = list(set(labels))
    return pipeline, classes


def explain_with_lime(pipeline, class_names, text, explainer, num_features=5):
    # LIME expects a function that accepts a list of strings and returns probability arrays
    prob_fn = pipeline.predict_proba
    exp = explainer.explain_instance(text, prob_fn, num_features=num_features)
    # Get list of (feature, weight) for top features for the predicted class
    # LIME returns tuples per class — use predicted class
    pred = pipeline.predict([text])[0]
    # LIME local_exp keys correspond to class indices used by predict_proba.
    # Determine the index of the predicted label from the trained classifier's classes_
    label_index = None
    try:
        final_clf = pipeline.named_steps.get("logisticregression")
        if final_clf is not None:
            label_index = int(list(final_clf.classes_).index(pred))
    except Exception:
        label_index = None

    if label_index is None or label_index not in exp.local_exp:
        # fall back to an available label index from LIME's explanation
        available = sorted(list(exp.local_exp.keys()))
        label_index = available[0] if available else 0

    feature_list = exp.as_list(label=label_index)
    # convert to simple dict list
    explanation = [{"feature": f, "weight": float(w)} for f, w in feature_list]
    return pred, explanation, exp


def generate_incident_log(events, pipeline, class_names):
    explainer = LimeTextExplainer(class_names=class_names)
    log_entries = []
    for e in events:
        text = e["description"]
        pred, explanation, _ = explain_with_lime(pipeline, class_names, text, explainer)
        proba = pipeline.predict_proba([text])[0].tolist()

        # build human readable summary using top explanation features
        top_feats = [f["feature"].split("=")[0] if "=" in f["feature"] else f["feature"] for f in explanation]
        summary = (
            f"Incident at {e['location']} on {e['timestamp']}: classified as {pred}. "
            f"Key contributing phrases: {', '.join(top_feats)}."
        )

        entry = {
            "id": e["id"],
            "timestamp": e["timestamp"],
            "location": e["location"],
            "description": text,
            "witness_count": e["witness_count"],
            "injured": e["injured"],
            "predicted_severity": pred,
            "predicted_proba": {class_names[i]: proba[i] for i in range(len(class_names))},
            "explanation": explanation,
            "summary": summary,
        }
        log_entries.append(entry)
    return log_entries


def main():
    # generate sample data
    events = generate_sample_events(n=25)

    # create a training dataset from the generated events using heuristic labels
    # duplicate and perturb examples to create more training samples
    train_events = []
    for e in events:
        for _ in range(3):
            ev = e.copy()
            # small perturbation
            if random.random() < 0.1:
                ev["description"] += "; additional note: slippery"
            train_events.append(ev)

    pipeline, class_names = train_text_classifier(train_events)

    # generate incident log with LIME explanations and summaries
    log = generate_incident_log(events, pipeline, class_names)

    out_file = "incident_log.json"
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)

    # print a short human-readable summary for each event
    for entry in log:
        print(entry["summary"])

    print(f"\nGenerated {len(log)} incidents and wrote to {out_file}")


if __name__ == "__main__":
    main()
