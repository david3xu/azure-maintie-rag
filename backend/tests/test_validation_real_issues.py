def test_entity_extraction_with_real_data():
    """Test entity extraction with realistic data structure"""
    from collections import defaultdict

    # Mock data similar to real MaintIE structure
    mock_documents = {
        "doc_3255": {
            "title": "Pump Maintenance Report",
            "text": "The centrifugal pump bearing failure was analyzed during routine maintenance",
            "doc_id": "doc_3255"
        },
        "doc_5579": {
            "title": "Equipment Status",
            "text": "Pump operational status shows bearing wear patterns",
            "doc_id": "doc_5579"
        }
    }

    mock_entities = {
        "entity_pump_001": {"text": "pump", "entity_id": "entity_pump_001"},
        "entity_bearing_001": {"text": "bearing", "entity_id": "entity_bearing_001"},
        "entity_failure_001": {"text": "failure", "entity_id": "entity_failure_001"},
        "entity_maintenance_001": {"text": "maintenance", "entity_id": "entity_maintenance_001"}
    }

    # Test entity extraction logic
    def extract_entities_enhanced(doc_data, entities):
        import re
        full_text = f"{doc_data['title']} {doc_data['text']}".lower()
        found_entities = []

        for entity_data in entities.values():
            entity_text = entity_data["text"].lower().strip()
            if re.search(r'\b' + re.escape(entity_text) + r'\b', full_text):
                found_entities.append(entity_data["text"])

        return found_entities

    # Test each document
    for doc_id, doc_data in mock_documents.items():
        entities = extract_entities_enhanced(doc_data, mock_entities)
        print(f"Document {doc_id}: {entities}")

        # Validate results
        assert len(entities) > 0, f"Document {doc_id} should have entities but got empty list"

        # Check specific expected entities
        if doc_id == "doc_3255":
            assert "pump" in entities, f"Document {doc_id} should contain 'pump'"
            assert "bearing" in entities, f"Document {doc_id} should contain 'bearing'"
            assert "failure" in entities, f"Document {doc_id} should contain 'failure'"
            assert "maintenance" in entities, f"Document {doc_id} should contain 'maintenance'"

        if doc_id == "doc_5579":
            assert "pump" in entities, f"Document {doc_id} should contain 'pump'"
            assert "bearing" in entities, f"Document {doc_id} should contain 'bearing'"

    print("✅ Enhanced entity extraction test passed")



