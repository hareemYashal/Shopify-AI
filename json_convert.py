import json

def convert_to_jsonl(input_file, output_file):
    # Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    products = data.get("products", [])

    with open(output_file, 'w', encoding='utf-8') as out:
        for p in products:
            # Extract main fields
            product_id = str(p.get("id"))
            title = p.get("title", "").strip()
            text = p.get("body_html", "")
            text = text.replace("<p>", "").replace("</p>", "").replace("<br>", " ").replace("\u003C/p\u003E", "").strip()
            price = None
            in_stock = False
            category = p.get("product_type", "")
            tags = p.get("tags", [])
            url = f"/products/{p.get('handle', '')}"

            # Get first image
            image = None
            if p.get("images"):
                image = p["images"][0].get("src")

            # Variant info for price + availability
            if p.get("variants"):
                v = p["variants"][0]
                try:
                    price = float(v.get("price", "0").replace("$", ""))
                except:
                    price = 0.0
                in_stock = bool(v.get("available", False))

            # Build the flattened object
            item = {
                "product_id": product_id,
                "title": title,
                "text": text,
                "price": price,
                "url": url,
                "image": image,
                "in_stock": in_stock,
                "category": category,
                "tags": tags
            }

            # Write to JSONL
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Converted {len(products)} products to {output_file}")


if __name__ == "__main__":
    convert_to_jsonl("new_file.json", "catalog.jsonl")
