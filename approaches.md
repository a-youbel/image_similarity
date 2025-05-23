# Possible approaches

The aim is to be able to detect identical or very similar images of different resolutions.

Different approaches are possible, list made with the help of ChatGPT

## Perceptual hashing

Perceptual hashing algorithms generate a fingerprint of an image such that visually similar images (even at different sizes or slight variations) will have similar hashes.

Perceptual hashing is a technique for summarizing the **visual appearance of an image** into a compact "fingerprint" (usually a short hash like `f0a4c3d8...`) such that **visually similar images have similar hashes**, even if their resolution, format, or minor features differ.

---

### 📸 Why Perceptual Hashing?

Unlike cryptographic hashes (e.g., MD5, SHA-1) where any small change results in a completely different hash, **perceptual hashes are designed to change gradually** with the image content.

Perceptual hashing is:

* Fast and lightweight
* Good for detecting resized, compressed, or slightly edited images
* Widely used in reverse image search, content deduplication, and moderation

---

### 🔑 Main Perceptual Hashing Methods

All these methods convert images into a fixed-size fingerprint. You can compute the Hamming distance (number of differing bits) between two hashes to assess similarity.

#### 1. **Average Hash (aHash)**

* Simplest method
* Converts image to grayscale and resizes to small fixed size (e.g., 8×8)
* Hash = 1 if pixel > average brightness, else 0

🟢 Good: Very fast
🔴 Bad: Not robust to contrast changes or complex transformations

#### 2. **Difference Hash (dHash)**

* Measures **gradient direction** (difference between adjacent pixels)
* More robust than aHash to brightness and contrast changes

🟢 Good: Efficient, better than aHash
🔴 Bad: Still can struggle with certain transformations

#### 3. **Perceptual Hash (pHash)**

* Uses **Discrete Cosine Transform (DCT)** to focus on low-frequency image features
* Captures overall structure and shapes

🟢 Good: Most robust among classic methods
🔴 Slightly slower

#### 4. **Wavelet Hash (wHash)**

* Uses wavelet transformation to capture both frequency and spatial features
* Best for capturing high-level image similarity

🟢 Good: Strong against noise and transformation
🔴 Less widely used

---

### 🧪 Example Using `ImageHash` (Python)

```python
from PIL import Image
import imagehash

img1 = Image.open('image1.jpg')
img2 = Image.open('image2.jpg')

hash1 = imagehash.phash(img1)  # or dhash, ahash, whash
hash2 = imagehash.phash(img2)

print(f"Hash1: {hash1}")
print(f"Hash2: {hash2}")
print(f"Hamming distance: {hash1 - hash2}")

if hash1 - hash2 < 5:
    print("Images are likely identical or very similar.")
```

### 🧮 Hash Comparison Rule of Thumb

| Hamming Distance | Similarity         |
| ---------------- | ------------------ |
| 0–5              | Likely identical   |
| 6–10             | Possibly similar   |
| 11+              | Probably different |

---

### 📂 Comparing Many Images

* Use a hash-based **lookup dictionary** or **LSH (Locality-Sensitive Hashing)** for fast comparison.
* For batch deduplication, compute hashes for all images and group ones with similar hashes.

---

Here's a ready-to-run **Python script** that uses **perceptual hashing (pHash)** to find **near-duplicate images in a folder**, even if they are at different resolutions or slightly modified.

---

### 🐍 Script: Find Near-Duplicate Images Using pHash

```python
import os
from PIL import Image
import imagehash
from collections import defaultdict
from itertools import combinations

# --- CONFIGURATION ---
IMAGE_FOLDER = "images"         # Change to your image folder path
HASH_FUNC = imagehash.phash     # Options: phash, ahash, dhash, whash
HASH_THRESHOLD = 5              # Hamming distance threshold for similarity

# --- COLLECT IMAGES & COMPUTE HASHES ---
hashes = {}
print("Scanning images and computing hashes...")

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        try:
            path = os.path.join(IMAGE_FOLDER, filename)
            img = Image.open(path)
            img_hash = HASH_FUNC(img)
            hashes[filename] = img_hash
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# --- FIND NEAR-DUPLICATES ---
print("\nFinding near-duplicate images:")
duplicates = defaultdict(list)

for file1, file2 in combinations(hashes.keys(), 2):
    hash1 = hashes[file1]
    hash2 = hashes[file2]
    dist = hash1 - hash2  # Hamming distance

    if dist <= HASH_THRESHOLD:
        duplicates[file1].append((file2, dist))

# --- DISPLAY RESULTS ---
if duplicates:
    for file, dupes in duplicates.items():
        print(f"\n{file} has near-duplicates:")
        for dupe, dist in dupes:
            print(f"  - {dupe} (distance: {dist})")
else:
    print("No duplicates found.")
```

---

### ✅ How to Use

1. Save the script to a file, e.g., `find_duplicates.py`.
2. Place all your images in a folder (e.g., `images/`).
3. Run:

```bash
python find_duplicates.py
```

---

### 🛠 Optional Improvements

* Save duplicate groups to a file
* Auto-delete or move duplicates
* Add support for multiple hash functions


## Feature matching

For more complex cases (e.g., when images are resized, cropped, or slightly modified):

Feature matching is a **classical computer vision technique** used to detect **identical or similar images**, even if they've been **resized, rotated, cropped, or distorted**. It works by identifying and comparing **distinctive patterns (features)** in two images.

---

### 🔍 What Is Feature Matching?

Feature matching finds **corresponding points** (called **keypoints**) between two images using:

1. **Feature detection** – find interesting points (corners, edges, textures).
2. **Feature description** – describe the local patch around each point.
3. **Feature matching** – compare descriptors to find likely matches.

---

### 🧠 Common Feature Detectors & Descriptors

| Name      | Description                       | Pros                                    | Cons                              |
| --------- | --------------------------------- | --------------------------------------- | --------------------------------- |
| **SIFT**  | Scale-Invariant Feature Transform | Very accurate, scale/rotation invariant | Patent expired (now open-source)  |
| **SURF**  | Speeded-Up Robust Features        | Faster than SIFT, robust                | Still under some licensing issues |
| **ORB**   | Oriented FAST and Rotated BRIEF   | Free, fast, decent accuracy             | Slightly less robust              |
| **AKAZE** | Accelerated-KAZE                  | Good for non-linear scale spaces        | Newer, not as widely used         |

---

### 🔧 Example: Image Matching with ORB (OpenCV)

```python
import cv2

# Load images in grayscale
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# Create ORB detector
orb = cv2.ORB_create()

# Detect and compute features
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match descriptors using Brute Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance (lower = better)
matches = sorted(matches, key=lambda x: x.distance)

# Calculate match ratio
match_ratio = len(matches) / max(len(kp1), len(kp2))
print(f"Match ratio: {match_ratio:.2f}")

# Draw top 50 matches
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
cv2.imshow("Matches", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### 📈 How to Interpret Results

* **High number of good matches** → Likely the same or similar images
* **Low number or scattered matches** → Different images

Use thresholds (e.g., match ratio > 0.3) to decide if images are duplicates.

---

### ✅ Feature Matching Is Good For:

* Detecting **exact matches at different scales or angles**
* Finding **sub-images** or **partial matches**
* Robust against **resolution changes**

---

### ❌ Limitations

* Slower than perceptual hashing for large datasets
* Can be sensitive to heavy filters or large occlusions
* Doesn’t scale well without indexing (for >1k images)

---

Here’s a **complete Python script** that uses **ORB feature matching** with OpenCV to detect **visually similar or identical image pairs** in a folder—even if they have different resolutions, are cropped, or rotated.

---

### 📁 Script: Detect Similar Images in a Folder with ORB

```python
import cv2
import os
from itertools import combinations

# --- CONFIGURATION ---
IMAGE_FOLDER = "images"       # Change this to your image folder
MATCH_THRESHOLD = 0.3         # Match ratio threshold (0.0 to 1.0)
TOP_MATCHES_TO_DRAW = 20      # Optional: Number of matches to visualize

# --- ORB FEATURE MATCHING SETUP ---
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def load_and_compute_features(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load {path}")
    kp, des = orb.detectAndCompute(img, None)
    return img, kp, des

# --- LOAD IMAGES AND FEATURES ---
images = {}
features = {}
print("Scanning images...")
for file in os.listdir(IMAGE_FOLDER):
    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        filepath = os.path.join(IMAGE_FOLDER, file)
        try:
            img, kp, des = load_and_compute_features(filepath)
            images[file] = img
            features[file] = (kp, des)
        except Exception as e:
            print(f"Skipped {file}: {e}")

# --- COMPARE IMAGE PAIRS ---
print("\nComparing image pairs...")
for (file1, file2) in combinations(features.keys(), 2):
    kp1, des1 = features[file1]
    kp2, des2 = features[file2]
    if des1 is None or des2 is None:
        continue

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    match_ratio = len(matches) / max(len(kp1), len(kp2))
    if match_ratio >= MATCH_THRESHOLD:
        print(f"\nMatch found between '{file1}' and '{file2}' (match ratio: {match_ratio:.2f})")

        # OPTIONAL: Show matches visually
        img_match = cv2.drawMatches(images[file1], kp1, images[file2], kp2, matches[:TOP_MATCHES_TO_DRAW], None, flags=2)
        cv2.imshow(f"{file1} ↔ {file2}", img_match)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
```

---

### ✅ How to Use

1. Save this script as `orb_matcher.py`
2. Place all your images in a folder, e.g. `images/`
3. Run the script:

```bash
python orb_matcher.py
```

It will:

* Print out matching image pairs
* Display visual matches (keypoints between images)
* Allow you to adjust the sensitivity with `MATCH_THRESHOLD`

---

### 🚀 More?

* **SIFT** instead of ORB? (More accurate but requires OpenCV with contrib modules)
* Batching it with perceptual hashing or deep learning to filter candidates first?


## Deep Learning Embeddings

Deep learning embeddings are a powerful and flexible way to detect visually identical or similar images, even if they're resized, cropped, color-adjusted, or slightly modified. 

An image embedding is a vector (a list of numbers) that captures the semantic content of an image, generated by passing the image through a pretrained deep neural network (like ResNet, CLIP, or Inception). These embeddings can then be compared using a distance metric like cosine similarity or Euclidean distance.

If two images are visually the same, even at different resolutions, their embeddings will be very close in the high-dimensional embedding space.

- Resolution invariant: Neural networks preprocess images to a standard size, so resolution differences are normalized.
- Robust to minor edits: Embeddings can still match images with slight crops, filters, or overlays.
- Scalable: You can use techniques like FAISS or Annoy to compare millions of image embeddings efficiently.

### Common deep learning models for Embeddings

| Model            | Description                                 | Usage Example                                        |
| ---------------- | ------------------------------------------- | ---------------------------------------------------- |
| **ResNet50**     | Classic CNN; great general-purpose features | `torchvision.models.resnet50(pretrained=True)`       |
| **CLIP**         | Joint vision-language model from OpenAI     | `openai/clip-vit-base-patch32` or via `transformers` |
| **EfficientNet** | Lightweight and accurate CNN                | `tf.keras.applications.EfficientNetB0`               |

Possible to use FAISS for large datasets (not the case for now)

```python
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = resnet50(pretrained=True)
model.eval()

# Remove final classification layer to get embeddings
model = torch.nn.Sequential(*list(model.children())[:-1])

# Transform to prepare the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().numpy()
    return embedding

# Get embeddings
emb1 = get_embedding("image1.jpg")
emb2 = get_embedding("image2.jpg")

# Compare embeddings
similarity = cosine_similarity([emb1], [emb2])[0][0]
print(f"Cosine similarity: {similarity:.4f}")


```
