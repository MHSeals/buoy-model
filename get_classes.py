from ultralytics import YOLO

# Load the model
model = YOLO("runs/detect/v20/weights/best.pt")

# Get class names and IDs
class_names = model.names
num_classes = len(class_names)

print(f"Total classes: {num_classes}\n")
print("Class ID | Class Name")
print("-" * 40)

for class_id, class_name in class_names.items():
    print(f"{class_id:8} | {class_name}")

# Also print as a dictionary for easy copy-paste
print("\n" + "="*40)
print("As dictionary:")
print(class_names)

# Print as a list (ordered by ID)
print("\n" + "="*40)
print("As list (ordered by ID):")
class_list = [class_names[i] for i in sorted(class_names.keys())]
print(class_list)
