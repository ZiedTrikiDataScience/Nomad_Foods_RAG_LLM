import json
from prefect import task, flow

@task(name="Reading_The_New_Entries" , log_prints=True)
def read_new_faq_entries(file_path="new_faq_data.json"):
    """
    Reads new FAQ entries from a JSON file.
    Args:
        file_path (str): Path to the JSON file with new FAQ entries.
    Returns:
        list: A list of new FAQ entries.
    """
    print(f"Reading new FAQ entries from '{file_path}'...")
    with open(file_path, 'r') as file:
        new_faq_data = json.load(file)
    print(f"Successfully read {len(new_faq_data['faq_data'])} new FAQ entries.")
    return new_faq_data["faq_data"]

@task(name="Updating_Faq_File" , log_prints=True)
def update_faq_file(new_faq_entries, source_file="faq_data.json"):
    """
    Updates the existing FAQ file with new entries from new_faq_entries.
    Args:
        new_faq_entries (list): List of FAQ entries with 'category', 'question', and 'answer'.
        source_file (str): Path to the existing JSON file containing FAQ data.
    """
    print(f"Updating existing FAQ file '{source_file}' with new entries...")
    # Load existing FAQ data
    with open(source_file, 'r') as file:
        existing_data = json.load(file)

    # Process each new FAQ entry
    for entry in new_faq_entries:
        category_found = False
        print(f"Processing entry: Category '{entry['category']}', Question '{entry['question']}'")
        
        # Check if the category exists in the existing data
        for existing_category in existing_data["faq_data"]:
            if existing_category["category"] == entry["category"]:
                # Append the new question and answer to the existing category
                existing_category["questions"].append({
                    "question": entry["question"],
                    "answer": entry["answer"]
                })
                print(f"Added to existing category '{entry['category']}'.")
                category_found = True
                break
        
        # If category is not found, add a new category entry
        if not category_found:
            existing_data["faq_data"].append({
                "category": entry["category"],
                "questions": [{
                    "question": entry["question"],
                    "answer": entry["answer"]
                }]
            })
            print(f"Created new category '{entry['category']}' and added the question.")

    # Save the updated FAQ data back to the source file
    with open(source_file, 'w') as file:
        json.dump(existing_data, file, indent=4)
    print(f"Successfully updated '{source_file}' with new FAQ entries.")

@flow(name="New_Faq_Ingestion_Flow", log_prints=True)
def faq_update_flow():
    # Read new entries from a JSON file
    new_faq_entries = read_new_faq_entries()
    # Update the main FAQ file with these entries
    update_faq_file(new_faq_entries)

# Run the flow
if __name__ == "__main__":
    faq_update_flow()
