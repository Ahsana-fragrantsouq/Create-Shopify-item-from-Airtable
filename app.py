from flask import Flask
from description_agent import generate_description_from_three_note_strings
from webhook_handlers import handle_airtable_webhook
from create_shopify_item import create_shopify_bp
import logging

# ✅ Initialize Flask app first
app = Flask(__name__)

# ✅ Then register blueprints
app.register_blueprint(create_shopify_bp)


@app.route("/generate", methods=["POST"])
def generate_description():
    from flask import request, jsonify
    data = request.get_json()
    perfume_name = data.get("perfume_name")
    brand_name = data.get("brand_name")
    top_notes = data.get("top_notes")
    middle_notes = data.get("middle_notes")
    base_notes = data.get("base_notes")
        
    if not perfume_name:
        return jsonify({"error": "perfume_name is required"}), 400
    description_html = generate_description_from_three_note_strings(perfume_name, brand_name,top_notes,middle_notes,base_notes)
    return jsonify({"description": description_html})


@app.route("/airtable-webhook", methods=["POST"])
def airtable_webhook_route():
    return handle_airtable_webhook()


@app.route("/", methods=["GET"])
def home():
    return "Unified Shopify + Description API running!", 200


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.logger.setLevel(logging.INFO)

    app.run(host="0.0.0.0", port=8080, debug=True)
