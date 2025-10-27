import os
from chroma_utils import get_chroma_client

def inspect_and_manage_chroma_db(persist_directory="./chroma_db"):
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        print(f"📁 Created directory: {persist_directory}")

    print(f"🔍 Inspecting ChromaDB at: {persist_directory}")

    # Initialize Chroma client using shared utility
    client = get_chroma_client(persist_directory)

    # List existing collections
    collections = client.list_collections()
    if not collections:
        print("⚠️ No collections found in this Chroma database.")
    else:
        print(f"✅ Found {len(collections)} collection(s):\n")
        for col in collections:
            name = col.name
            print(f"📦 Collection: {name}")
            try:
                collection = client.get_collection(name)
                count = collection.count()
                print(f"   • Total items: {count}")
                data = collection.get(limit=3, include=["metadatas", "documents", "ids"])
                print(f"   • Sample IDs: {data.get('ids', [])}")
                if data.get("metadatas"):
                    keys = list(data["metadatas"][0].keys())
                    print(f"   • Sample metadata keys: {keys}")
                print("-" * 50)
            except Exception as e:
                print(f"   ⚠️ Error inspecting collection '{name}': {e}")
                print("-" * 50)

    # Ask user if they want to create a new collection
    create_new = input("\n➕ Do you want to create a new collection? (y/n): ").strip().lower()
    if create_new == "y":
        new_name = input("🆕 Enter new collection name: ").strip()
        if new_name:
            try:
                new_collection = client.get_or_create_collection(name=new_name)
                print(f"✅ Collection '{new_name}' created successfully!")
            except Exception as e:
                print(f"❌ Failed to create collection '{new_name}': {e}")
        else:
            print("⚠️ Collection name cannot be empty.")
    else:
        print("👋 No new collection created.")

if __name__ == "__main__":
    inspect_and_manage_chroma_db("./chroma_db")
