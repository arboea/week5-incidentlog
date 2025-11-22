import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt


ROOT_CAUSE_KEYWORDS = {
    "Slip/Trip (wet floor)": ["slip", "slipped", "wet", "floor", "pipe"],
    "Falling objects": ["box", "shelf", "fell", "fall", "foot", "hitting"],
    "Cuts/Sharp object": ["cut", "cutter", "box cutter", "hand"],
    "Chemical spill": ["chemical", "spill", "storage"],
    "Equipment / Forklift": ["forklift", "pallet", "collided", "knocked"],
    "Fire/Overheat": ["fire", "overheated", "alarm", "triggered"],
    "Maintenance/Fall": ["ladder", "maintenance", "task"],
    "Electrical": ["wiring", "electrical", "spark"],
    "Struck by/against": ["head", "beam", "struck"],
}


def match_root_cause(token):
    t = token.lower()
    matched = []
    for rc, keywords in ROOT_CAUSE_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                matched.append(rc)
                break
    return matched


def main():
    in_file = "incident_log.json"
    out_dir = "visualizations"
    os.makedirs(out_dir, exist_ok=True)

    with open(in_file, "r", encoding="utf-8") as fh:
        incidents = json.load(fh)

    # Aggregate absolute explanation weights per root cause
    rc_scores = defaultdict(float)
    rc_counts = defaultdict(int)

    for inc in incidents:
        explanation = inc.get("explanation", [])
        # use top features (they're already ranked) — consider all provided
        for feat in explanation:
            token = feat.get("feature", "").strip()
            weight = abs(feat.get("weight", 0.0))
            # normalize tokens like 'and' or punctuation
            if len(token) == 0 or token.lower() in {"and", "the", "of", "in", "to", "by", "with", "from"}:
                continue
            matched = match_root_cause(token)
            if not matched:
                # try splitting token on non-alpha
                subtoks = [t for t in token.replace("_", " ").split() if len(t) > 2]
                for st in subtoks:
                    m2 = match_root_cause(st)
                    if m2:
                        matched = m2
                        break

            if matched:
                for rc in matched:
                    rc_scores[rc] += weight
                    rc_counts[rc] += 1
            else:
                rc_scores["Other"] += weight
                rc_counts["Other"] += 1

    # Prepare data for plotting: sort by score
    items = sorted(rc_scores.items(), key=lambda x: x[1], reverse=True)
    labels = [it[0] for it in items]
    scores = [it[1] for it in items]
    counts = [rc_counts[l] for l in labels]

    # Plot a horizontal bar chart of aggregated LIME importance per root cause
    plt.figure(figsize=(8, max(3, 0.5 * len(labels))))
    bars = plt.barh(labels, scores, color="C0")
    plt.xlabel("Aggregated LIME importance (sum of |weights|)")
    plt.title("Likely Root Causes — aggregated from LIME explanations")
    plt.gca().invert_yaxis()

    # annotate counts next to bars
    for i, b in enumerate(bars):
        plt.text(b.get_width() + max(scores) * 0.01, b.get_y() + b.get_height() / 2, f"count={counts[i]}", va="center")

    out_png = os.path.join(out_dir, "root_causes.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    # write a simple HTML page to display the visual
    index_html = os.path.join(out_dir, "root_causes.html")
    with open(index_html, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>Root Cause Summary</title></head><body>\n")
        fh.write("<h1>Likely Root Causes (from LIME)</h1>\n")
        fh.write(f"<p>Visual aggregated from {len(incidents)} incidents. Image shows summed absolute LIME weights per root cause; annotated with counts.</p>\n")
        fh.write(f"<img src='root_causes.png' alt='Root causes chart' style='max-width:100%;height:auto;'>\n")
        fh.write("</body></html>\n")

    print(f"Wrote root-cause visual to {out_png} and HTML page {index_html}")


if __name__ == "__main__":
    main()
