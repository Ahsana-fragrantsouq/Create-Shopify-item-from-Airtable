from flask import Flask
from description_agent import generate_description_from_three_note_strings, generate_description_for_api, CONFIG
from webhook_handlers import handle_airtable_webhook
from create_shopify_item import create_shopify_bp
import logging

# ✅ Initialize Flask app first
app = Flask(__name__)

# ✅ Then register blueprints
app.register_blueprint(create_shopify_bp)



# ---------- Your Flask API Endpoint (to be used in your app) ----------

@app.route("/generate", methods=["POST"])
def generate_description():
    from flask import request, jsonify

    data = request.get_json(silent=True) or {}
    print("DEBUG incoming JSON =", data)

    # Required field
    perfume_name = data.get("perfume_name")
    if not perfume_name:
        return jsonify({"error": "perfume_name is required"}), 400

    # Optional fields
    brand_name = data.get("brand_name")
    top_notes = data.get("top_notes")
    middle_notes = data.get("middle_notes")
    base_notes = data.get("base_notes")
    gift_items_list = data.get("gift_items_list")  # list of strings

    model = data.get("model", CONFIG["default_model"])

    # -----------------------------
    # GIFT SET vs PERFUME ROUTER
    # -----------------------------
    if "gift set" in perfume_name.lower():
        if not gift_items_list or not isinstance(gift_items_list, list):
            return jsonify({
                "error": "gift_items_list (array) is required for gift set products"
            }), 400

        result = generate_gift_set_description(
            product_name=perfume_name,
            brand_name=brand_name,
            set_items=gift_items_list,
            model=model
        )
    else:
        result = generate_description_for_api(
            perfume_name=perfume_name,
            brand_name=brand_name,
            top_notes=top_notes,
            middle_notes=middle_notes,
            base_notes=base_notes,
            model=model
        )

    # -----------------------------
    # RESPONSE
    # -----------------------------
    if result.get("success"):
        return jsonify({
            "success": True,
            "description": result.get("description") or result.get("description_html"),
            "metadata": {
                "perfume_name": perfume_name,
                "brand_name": brand_name,
                "length": result.get("length"),
                "model_used": result.get("model_used")
            }
        })
    else:
        return jsonify({
            "success": False,
            "error": result.get("error"),
            "fallback_description": result.get("fallback_description", "")
        }), 500


"""
# def generate_description():
#     from flask import request, jsonify
    
#     data = request.get_json(silent=True) or {}
#     print("DEBUG incoming JSON =", data)

#     # Required field
#     perfume_name = data.get("perfume_name")
#     if not perfume_name:
#         return jsonify({"error": "perfume_name is required"}), 400
    
#     # Optional fields with defaults
#     brand_name = data.get("brand_name")
#     top_notes = data.get("top_notes")
#     middle_notes = data.get("middle_notes")
#     base_notes = data.get("base_notes")
    
#     # Optional: model parameter
#     model = data.get("model", CONFIG["default_model"])
    
#     # Generate description
#     result = generate_description_for_api(
#         perfume_name=perfume_name,
#         brand_name=brand_name,
#         top_notes=top_notes,
#         middle_notes=middle_notes,
#         base_notes=base_notes,
#         model=model
#     )
    
#     if result["success"]:
#         return jsonify({
#             "success": True,
#             "description": result["description"],
#             "metadata": {
#                 "perfume_name": result["perfume_name"],
#                 "brand_name": result["brand_name"],
#                 "length": result["length"],
#                 "model_used": result["model_used"]
#             }
#         })
#     else:
#         return jsonify({
#             "success": False,
#             "error": result["error"],
#             "fallback_description": result.get("fallback_description", "")
#         }), 500



# @app.route("/generate", methods=["POST"])
# def generate_description():
#     from flask import request, jsonify
#     data = request.get_json()
#     perfume_name = data.get("perfume_name")
#     brand_name = data.get("brand_name")
#     top_notes = data.get("top_notes")
#     middle_notes = data.get("middle_notes")
#     base_notes = data.get("base_notes")
        
#     if not perfume_name:
#         return jsonify({"error": "perfume_name is required"}), 400
#     description_html = generate_description_from_three_note_strings(perfume_name, brand_name,top_notes,middle_notes,base_notes)
#     return jsonify({"description": description_html})
"""


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
