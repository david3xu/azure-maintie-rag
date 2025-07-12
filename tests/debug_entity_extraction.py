from src.knowledge.data_transformer import MaintIEDataTransformer
import json
from config.settings import settings

def debug_entity_extraction():
    """Debug the entity extraction process step by step"""

    transformer = MaintIEDataTransformer()

    # Step 1: Check if raw data loads
    print("🔄 Step 1: Loading raw data...")
    try:
        # Directly call the internal method as it's what the transformer uses
        raw_gold_data = transformer._load_json_file(settings.raw_data_dir / settings.gold_data_filename)
        raw_silver_data = transformer._load_json_file(settings.raw_data_dir / settings.silver_data_filename)
        raw_data = raw_gold_data + raw_silver_data # Combine for simpler debugging

        print(f"✅ Raw data loaded: {len(raw_data)} items (combined gold and silver)")

        # Sample first item
        if raw_data:
            first_item = raw_data[0]
            print(f"📊 First item structure: {json.dumps(first_item, indent=2)[:500]}...") # Increased limit to 500 for more context
    except Exception as e:
        print(f"❌ Raw data loading failed: {e}")
        return

    # Step 2: Check entity extraction from sample
    print("\n🔄 Step 2: Testing entity extraction on sample...")
    try:
        if not raw_data:
            print("⚠️ No raw data to process. Exiting entity extraction debug.")
            return

        sample_item = raw_data[0]
        doc_id = sample_item.get("id", "sample_doc_0") # Use a consistent doc_id for debugging
        doc_text = sample_item.get("text", "")
        doc_tokens = sample_item.get("tokens", [])

        # Try to extract entities from first item
        if 'entities' in sample_item:
            entities_data = sample_item['entities']
            print(f"📊 Found {len(entities_data)} entities in sample item")

            for i, entity_data in enumerate(entities_data[:5]):  # Show first 5 entities for detailed look
                print(f"   Processing entity {i}: {entity_data}")
                # Directly call _create_entity with all necessary arguments
                entity = transformer._create_entity(entity_data, doc_id, doc_text, doc_tokens, 1.0) # Use 1.0 for confidence_base for debugging
                if entity:
                    print(f"   ✅ Created entity: ID={entity.entity_id}, Text='{entity.text}', Type={entity.entity_type.value}")
                else:
                    print(f"   ❌ Failed to create entity from data: {entity_data}")
        else:
            print("❌ No 'entities' key found in sample item")
            print(f"📊 Available keys: {list(sample_item.keys())}")

    except Exception as e:
        print(f"❌ Entity extraction test failed: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for detailed error

if __name__ == "__main__":
    debug_entity_extraction()
