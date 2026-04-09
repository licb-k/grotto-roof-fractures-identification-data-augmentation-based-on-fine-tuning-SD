import json

# Original JSON file path
json_file = "diffusion_fine-tuning/data1/train.json"

# Old path prefix
old_prefix = "path/to/diffusion_fine-tuning"

# New path prefix
new_prefix = "path/to/diffusion_fine-tuning"

# Read the JSON file
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Replace the path in the "image" field
for item in data:
    if "image" in item and isinstance(item["image"], str):
        item["image"] = item["image"].replace(old_prefix, new_prefix)

# Save the modified JSON file
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Path replacement completed")
