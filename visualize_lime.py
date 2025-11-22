import json
import os
from incident_logger import train_text_classifier
from lime.lime_text import LimeTextExplainer


def main():
    in_file = "incident_log.json"
    out_dir = "visualizations"
    os.makedirs(out_dir, exist_ok=True)

    with open(in_file, "r", encoding="utf-8") as fh:
        incidents = json.load(fh)

    # Build a small training set from the incident descriptions
    train_events = []
    for inc in incidents:
        ev = {
            "description": inc.get("description", ""),
            "injured": inc.get("injured", False),
            "witness_count": inc.get("witness_count", 0),
        }
        # duplicate to give the classifier more examples
        for _ in range(4):
            train_events.append(ev.copy())

    pipeline, class_names = train_text_classifier(train_events)
    explainer = LimeTextExplainer(class_names=class_names)

    index_lines = [
        "<html>",
        "<head><meta charset='utf-8'><title>Incident LIME Visualizations</title></head>",
        "<body>",
        "<h1>Incident LIME Visualizations</h1>",
        "<ul>",
    ]

    for inc in incidents:
        text = inc.get("description", "")
        exp = explainer.explain_instance(text, pipeline.predict_proba, num_features=6)
        html = exp.as_html()

        fname = f"incident_{inc['id']}.html"
        path = os.path.join(out_dir, fname)

        # Save the LIME HTML output for the incident
        with open(path, "w", encoding="utf-8") as fh:
            # Write a small header then the LIME HTML (LIME returns a full HTML document)
            fh.write(f"<!-- Incident: {inc['id']} -->\n")
            fh.write(f"<h2>Incident {inc['id']}</h2>\n")
            fh.write(f"<p><strong>Location:</strong> {inc.get('location','')}<br>")
            fh.write(f"<strong>Time:</strong> {inc.get('timestamp','')}<br>")
            fh.write(f"<strong>Predicted severity:</strong> {inc.get('predicted_severity','')}<br></p>\n")
            fh.write(html)

        index_lines.append(
            f"<li><a href=\"{fname}\">{inc.get('timestamp','')} — {inc.get('location','')} — {inc.get('predicted_severity','')}</a></li>"
        )

    index_lines.extend(["</ul>", "</body>", "</html>"])
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(index_lines))

    print(f"Wrote {len(incidents)} visualizations into '{out_dir}/' (open {out_dir}/index.html)")


if __name__ == "__main__":
    main()
