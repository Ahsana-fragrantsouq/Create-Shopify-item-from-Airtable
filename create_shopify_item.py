import os
import requests
from flask import Blueprint, request, jsonify
from airtable import Airtable
import shopify
from typing import Dict, Optional
from shopify_utils import _to_number, shopify_graphql

import unicodedata
import re

# ---------------------------
# 🔐 SHOPIFY CLIENT CREDS
# ---------------------------
SHOPIFY_CLIENT_ID = os.getenv("SHOPIFY_CLIENT_ID")
SHOPIFY_CLIENT_SECRET = os.getenv("SHOPIFY_CLIENT_SECRET")

# ---------------------------
# CONFIGURATION
# ---------------------------
SHOP = os.environ.get("SHOPIFY_SHOP")
TOKEN = os.environ.get("SHOPIFY_API_TOKEN")
API_VERSION = os.getenv("SHOPIFY_API_VERSION", "2025-01")

AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")
AIRTABLE_TABLE_NAME = "French Inventories"



def convert_title_to_image_name(product_title):
    product_name_clean = unicodedata.normalize('NFKD', product_title)
    product_name_clean = product_name_clean.encode('ascii', 'ignore').decode('ascii')

    brand_replacements = {
        "Victor&Rolf": "Viktor_Rolf",
        "Van Cleef & Arpels": "Van_Cleef_Arpels",
        "Dolce & Gabbana": "Dolce_Gabbana",
        "Yves Saint Laurent": "Yves_Saint_Laurent",
        "Jean Paul Gaultier": "Jean_Paul_Gaultier",
        "Salvatore Ferragamo": "Salvatore_Ferragamo",
        "Tonino Lamborghini": "Tonino_Lamborghini"
    }

    for old, new in brand_replacements.items():
        product_name_clean = product_name_clean.replace(old, new)

    product_name_clean = product_name_clean.replace("'", "_")
    product_name_clean = product_name_clean.replace("&", "and")
    product_name_clean = re.sub(r'([A-Za-z])(\d)', r'\1_\2', product_name_clean)
    product_name_clean = re.sub(r'(\d)([A-Za-z])', r'\1_\2', product_name_clean)
    product_name_clean = re.sub(r'[®™°º"+]+', '', product_name_clean)
    product_name_clean = re.sub(r'[.,:/()\-]', ' ', product_name_clean)
    product_name_clean = re.sub(r'\s+', ' ', product_name_clean).strip()
    product_name_clean = product_name_clean.replace(" ", "_")
    product_name_clean = re.sub(r'_+', '_', product_name_clean)
    product_name_clean = re.sub(r'^[^A-Za]*', '', product_name_clean)
    product_name_clean = product_name_clean.strip('_')

    return product_name_clean


create_shopify_bp = Blueprint("create_shopify_bp", __name__)


# ---------------------------
# 🔐 ACCESS TOKEN FETCH
# ---------------------------
def get_shopify_access_token():
    print("🔐 Requesting Shopify access token...", flush=True)

    if not SHOPIFY_CLIENT_ID or not SHOPIFY_CLIENT_SECRET or not SHOP:
        raise Exception("Missing SHOPIFY_CLIENT_ID / SHOPIFY_CLIENT_SECRET / SHOPIFY_SHOP")

    url = f"https://{SHOP}/admin/oauth/access_token"

    payload = {
        "client_id": SHOPIFY_CLIENT_ID,
        "client_secret": SHOPIFY_CLIENT_SECRET,
        "grant_type": "client_credentials"
    }

    resp = requests.post(url, json=payload)
    print(f"🔁 Token raw response: {resp.text}", flush=True)

    data = resp.json()
    token = data.get("access_token")

    if not token:
        raise Exception("Token generation failed")

    print("✅ Shopify access token received", flush=True)
    return token


# ---------------------------
# SHOPIFY SESSION MANAGEMENT
# ---------------------------
def setup_shopify_session():
    try:
        global TOKEN

        if not TOKEN:
            TOKEN = get_shopify_access_token()

        shopify.Session.setup(api_key="dummy", secret="dummy")
        session = shopify.Session(f"https://{SHOP}", API_VERSION, TOKEN)
        shopify.ShopifyResource.activate_session(session)
        return True

    except Exception as e:
        print(f"⚠️ Shopify session setup failed: {e}", flush=True)
        return False


def clear_shopify_session():
    try:
        shopify.ShopifyResource.clear_session()
    except Exception as e:
        print(f"⚠️ Error clearing Shopify session: {e}", flush=True)


# ---------------------------
# IMAGE SEARCHER
# ---------------------------
class ImageSearcher:
    @staticmethod
    def search_by_product_name(product_name: str, limit=10, exact_match=False, cursor=None) -> Dict:
        if not product_name:
            return {"success": False, "images": []}

        try:
            if not setup_shopify_session():
                return {"success": False, "images": []}

            product_name_clean = convert_title_to_image_name(product_name)
            search_pattern = f'filename:{product_name_clean}*'

            query = f"""
            query {{
              files(first: {limit}, query: "{search_pattern} AND media_type:IMAGE") {{
                edges {{
                  node {{
                    ... on MediaImage {{
                      image {{ url }}
                    }}
                  }}
                }}
              }}
            }}
            """

            gql = shopify.GraphQL()
            result = gql.execute(query)

            if isinstance(result, str):
                import json
                result = json.loads(result)

            images = [
                edge["node"]["image"]["url"]
                for edge in result.get("data", {}).get("files", {}).get("edges", [])
                if edge.get("node", {}).get("image")
            ]

            print(f"✅ Found {len(images)} images", flush=True)
            return {"success": True, "images": images, "count": len(images)}

        except Exception as e:
            print(f"⚠️ Image search error: {e}", flush=True)
            return {"success": False, "images": []}
        finally:
            clear_shopify_session()


# ---------------------------
# HELPERS
# ---------------------------
def _json_headers():
    return {"Content-Type": "application/json", "X-Shopify-Access-Token": TOKEN}


def _rest_url(path):
    return f"https://{SHOP}/admin/api/{API_VERSION}/{path}"


def _graphql_url():
    return f"https://{SHOP}/admin/api/{API_VERSION}/graphql.json"


# ---------------------------
# MAIN ROUTE
# ---------------------------
@create_shopify_bp.route("/create-shopify-item", methods=["POST"])
def create_shopify_item():
    try:
        data = request.get_json(force=True)
        record = data.get("fields", {})
        record_id = data.get("record_id")

        qty = _to_number(record.get("Qty given in shopify", 0))
        status = "active" if qty > 0 else "draft"

        product_data = {
            "product": {
                "title": record.get("Product Name", ""),
                "status": status,
                "variants": [{
                    "price": str(_to_number(record.get("UAE Price", 0))),
                    "inventory_management": "shopify"
                }]
            }
        }

        resp = requests.post(_rest_url("products.json"), headers=_json_headers(), json=product_data)

        if resp.status_code != 201:
            return jsonify({"error": resp.text}), resp.status_code

        product = resp.json()["product"]
        print(f"✅ Product created: {product['id']}", flush=True)

        return jsonify({"success": True, "shopify_id": product["id"]}), 201

    except Exception as e:
        print(f"❌ Unexpected error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500
