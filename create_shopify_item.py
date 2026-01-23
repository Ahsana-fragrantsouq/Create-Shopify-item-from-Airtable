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
# 🔐 SHOPIFY TOKEN FETCHER (WITH LOGS)
# ---------------------------
def getShopifyAccessToken():
    print("🔐 Fetching Shopify Admin API token...", flush=True)

    token = os.getenv("SHOPIFY_API_TOKEN")

    if not token:
        print("❌ SHOPIFY_API_TOKEN missing", flush=True)
        raise Exception("SHOPIFY_API_TOKEN not set")

    print(f"✅ Token loaded (masked): {token[:6]}********", flush=True)
    return token


# ---------------------------
# IMAGE NAME NORMALIZER
# ---------------------------
def convert_title_to_image_name(product_title):
    print(f"🧹 Normalizing product title: {product_title}", flush=True)

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

    product_name_clean = product_name_clean.replace("'", "_").replace("&", "and")
    product_name_clean = re.sub(r'([A-Za-z])(\d)', r'\1_\2', product_name_clean)
    product_name_clean = re.sub(r'(\d)([A-Za-z])', r'\1_\2', product_name_clean)
    product_name_clean = re.sub(r'[®™°º"+.,:/()\-]', ' ', product_name_clean)

    product_name_clean = re.sub(r'\s+', ' ', product_name_clean).strip().replace(" ", "_")
    product_name_clean = re.sub(r'_+', '_', product_name_clean)

    print(f"🧼 Clean filename: {product_name_clean}", flush=True)
    return product_name_clean


create_shopify_bp = Blueprint("create_shopify_bp", __name__)

# ---------------------------
# CONFIG
# ---------------------------
SHOP = os.environ.get("SHOPIFY_SHOP")
TOKEN = getShopifyAccessToken()
API_VERSION = os.getenv("SHOPIFY_API_VERSION", "2025-01")

AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")
AIRTABLE_TABLE_NAME = "French Inventories"

print(f"🏪 Shopify shop: {SHOP}", flush=True)
print(f"📦 API version: {API_VERSION}", flush=True)

airtable = None
if AIRTABLE_BASE_ID and AIRTABLE_API_KEY:
    airtable = Airtable(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, AIRTABLE_API_KEY)
    print("✅ Airtable initialized", flush=True)
else:
    print("⚠️ Airtable credentials missing", flush=True)

# ---------------------------
# SHOPIFY SESSION
# ---------------------------
def setup_shopify_session():
    try:
        print("🔄 Activating Shopify session...", flush=True)
        shopify.Session.setup(api_key="dummy", secret="dummy")
        session = shopify.Session(f"https://{SHOP}", API_VERSION, TOKEN)
        shopify.ShopifyResource.activate_session(session)
        print("✅ Shopify session active", flush=True)
        return True
    except Exception as e:
        print(f"❌ Shopify session failed: {e}", flush=True)
        return False


def clear_shopify_session():
    shopify.ShopifyResource.clear_session()
    print("🧹 Shopify session cleared", flush=True)


# ---------------------------
# IMAGE SEARCH
# ---------------------------
class ImageSearcher:
    @staticmethod
    def search_by_product_name(product_name: str, limit: int = 10, cursor: Optional[str] = None) -> Dict:
        print(f"🔍 Starting image search for: {product_name}", flush=True)

        try:
            if not setup_shopify_session():
                return {"success": False, "error": "Session failed", "images": []}

            clean_name = convert_title_to_image_name(product_name)
            search_pattern = f'filename:{clean_name}*'
            after = f', after: "{cursor}"' if cursor else ""

            query = f"""
            query {{
              files(first: {limit}{after}, query: "{search_pattern} AND media_type:IMAGE") {{
                edges {{
                  node {{
                    ... on MediaImage {{
                      id
                      image {{ url }}
                    }}
                  }}
                }}
              }}
            }}
            """

            print(f"📤 GraphQL query running...", flush=True)
            gql = shopify.GraphQL()
            result = gql.execute(query)

            if isinstance(result, str):
                import json
                result = json.loads(result)

            if "errors" in result:
                print(f"❌ GraphQL error: {result['errors']}", flush=True)
                return {"success": False, "error": result["errors"], "images": []}

            images = [e["node"] for e in result["data"]["files"]["edges"]]
            print(f"✅ {len(images)} images found", flush=True)

            return {"success": True, "images": images, "count": len(images)}

        except Exception as e:
            print(f"❌ Image search exception: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e), "images": []}

        finally:
            clear_shopify_session()


# ---------------------------
# HELPERS
# ---------------------------
def _json_headers():
    return {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": TOKEN
    }

def _rest_url(path: str):
    return f"https://{SHOP}/admin/api/{API_VERSION}/{path}"

def _graphql_url():
    return f"https://{SHOP}/admin/api/{API_VERSION}/graphql.json"
