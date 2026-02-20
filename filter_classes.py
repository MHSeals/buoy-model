"""
Preprocessing steps for YOLO-format datasets:

remap_classes:
  - Renames classes in-place (ORIGINAL_NAME -> NEW_NAME).
  - The target name must already exist in the dataset (supports merging).
  - Source classes are removed from the class list; their annotations are
    re-labelled as the target class.
  - data.yaml is updated to reflect the new class list.

filter_classes:
  - Removes annotations whose class is not in `keep_classes`.
  - Remaining class IDs are remapped to be consecutive starting from 0.
  - If a label file becomes empty after filtering, both the label file and its
    corresponding image are deleted.
  - data.yaml is updated to reflect the new class list.

Run remap_classes before filter_classes so renamed classes are correctly
recognised by the filter step.
"""

import os
import yaml
from tqdm import tqdm


def remap_classes(dataset_path: str, remap: dict[str, str]) -> None:
    """
    Rename/merge classes in a YOLO-format dataset in-place.

    Args:
        dataset_path: Path to the dataset root containing data.yaml.
        remap: Mapping of {original_class_name: target_class_name}.
               The target class must already exist in the dataset.
               After remapping the original class is removed from the
               class list and all its annotations are relabelled as the
               target class.
    """
    yaml_path = os.path.join(dataset_path, "data.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    original_names: list[str] = data["names"]

    # Validate sources and targets
    for src, tgt in remap.items():
        if src not in original_names:
            raise ValueError(f"Remap source '{src}' not found in dataset classes: {original_names}")
        if tgt not in original_names:
            raise ValueError(
                f"Remap target '{tgt}' not found in dataset classes: {original_names}. "
                "The target class must already exist in the dataset."
            )

    print(f"Remapping classes: {remap}")

    # New names list: remove all source classes
    new_names = [n for n in original_names if n not in remap]

    # Build old_id -> new_id mapping
    old_id_to_new_id: dict[int, int] = {}
    for old_id, name in enumerate(original_names):
        target_name = remap.get(name, name)  # resolve through remap if present
        old_id_to_new_id[old_id] = new_names.index(target_name)

    splits = ["train", "test", "val"]

    for split in splits:
        labels_dir = os.path.join(dataset_path, split, "labels")
        if not os.path.isdir(labels_dir):
            continue

        label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
        remapped_count = 0

        for label_file in tqdm(label_files, desc=f"Remapping {split}"):
            label_path = os.path.join(labels_dir, label_file)

            with open(label_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                old_class_id = int(parts[0])
                new_class_id = old_id_to_new_id[old_class_id]
                if new_class_id != old_class_id:
                    remapped_count += 1
                parts[0] = str(new_class_id)
                new_lines.append(" ".join(parts))

            with open(label_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")

        print(f"  {split}: remapped {remapped_count} annotation(s)")

    # Update data.yaml
    data["names"] = new_names
    data["nc"] = len(new_names)

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"Updated data.yaml after remap: nc={len(new_names)}, names={new_names}")


def filter_classes(dataset_path: str, keep_classes: list[str]) -> None:
    """
    Filter a YOLO-format dataset in-place, keeping only the listed classes.

    Args:
        dataset_path: Absolute or relative path to the dataset root (the folder
                      that contains data.yaml and train/test/val sub-directories).
        keep_classes: List of class name strings to retain.
    """
    yaml_path = os.path.join(dataset_path, "data.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    original_names: list[str] = data["names"]
    print(f"Original classes ({len(original_names)}): {original_names}")

    # Validate that all requested classes actually exist in the dataset
    missing = [c for c in keep_classes if c not in original_names]
    if missing:
        raise ValueError(f"The following classes were not found in the dataset: {missing}")

    # Build old_id -> new_id mapping (only for kept classes)
    old_id_to_new_id: dict[int, int] = {}
    new_id = 0
    for old_id, name in enumerate(original_names):
        if name in keep_classes:
            old_id_to_new_id[old_id] = new_id
            new_id += 1

    # Preserve the order defined by keep_classes
    new_names = [name for name in original_names if name in keep_classes]
    print(f"Keeping classes ({len(new_names)}): {new_names}")

    splits = ["train", "test", "val"]
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    for split in splits:
        labels_dir = os.path.join(dataset_path, split, "labels")
        images_dir = os.path.join(dataset_path, split, "images")

        if not os.path.isdir(labels_dir):
            continue

        label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
        removed_images = 0
        filtered_annotations = 0

        for label_file in tqdm(label_files, desc=f"Filtering {split}"):
            label_path = os.path.join(labels_dir, label_file)

            with open(label_path, "r") as f:
                lines = f.readlines()

            kept_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                old_class_id = int(parts[0])
                if old_class_id in old_id_to_new_id:
                    parts[0] = str(old_id_to_new_id[old_class_id])
                    kept_lines.append(" ".join(parts))
                else:
                    filtered_annotations += 1

            if not kept_lines:
                # Remove empty label file and its corresponding image
                os.remove(label_path)
                stem = os.path.splitext(label_file)[0]
                for ext in image_extensions:
                    img_path = os.path.join(images_dir, stem + ext)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        break
                removed_images += 1
            else:
                with open(label_path, "w") as f:
                    f.write("\n".join(kept_lines) + "\n")

        print(
            f"  {split}: removed {filtered_annotations} annotations, "
            f"deleted {removed_images} image(s) with no remaining labels"
        )

    # Update data.yaml
    data["names"] = new_names
    data["nc"] = len(new_names)

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"Updated data.yaml: nc={len(new_names)}, names={new_names}")
