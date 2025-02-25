# AI and Machine Learning Concepts

## 1. Supervised Learning
A type of ML where the model learns from **labeled data** (input-output pairs).
  - Training a ad filter using labels as **compliant** or **"non compliant"**.
### Text - NLP
 - AI analyzes ad copy, captions, and metadata to detect prohibited words, hate speech, misleading claims, etc.
 - Techniques like Named Entity Recognition (NER) and Sentiment Analysis help in categorizing content.
### Image & Video - CNN

## 2. Unsupervised Learning
The model learns from unlabeled data, finding patterns and relationships on its own.
 - Customer segmentation—grouping users based on behavior without predefined labels.
## 3.  Human-in-the-Loop (HITL) AI
A system where humans assist in AI training, reviewing or correcting labels to improve model accuracy.
### Metrics
Accuracy: Percentage of correctly classified data points.
Precision & Recall: Important for imbalanced datasets (TP, FP, TN, FN).
F1 Score: Balance between precision and recall.
## 4. Active Learning
Instead of labeling all data, the AI model chooses the most useful data points to be labeled, reducing labeling effort.
Used in situations where labeling is expensive, like medical image analysis.

# AI models training workflow
## 1. Data collection & Labeling
-  historical ad data
1. text
2. image and video
3. metadata
- compliant/non-compliant ads label
## 2. Data Cleaning
text/images/videos
### Remove noise 
irrelevant words, duplicate ads
### Tokenize text
```python
import re

text_ad = "🚀 Buy NOW!!! Get a HUGE discount: https://example.com 🎉🔥"

# Remove URLs, emojis, and special characters
clean_text = re.sub(r"http\S+", "", text_ad)  # Remove URLs
clean_text = re.sub(r"[^\w\s]", "", clean_text)  # Remove special characters

print("Cleaned Text:", clean_text)
```
#### Named Entity Recognition
```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("I'm going to New York City for work.")
tokens = [token.text for token in doc]

print(tokens)
```
### Resize and normalize images & videos 
```python
import cv2
import numpy as np

image = cv2.imread("ad.jpg")
normalized_image = image / 255.0  # Scale pixel values to range [0,1]
```
#### scaling pixel values (0-1)
#### Resize dimensions
Resizing media to consistent dimensions for uniform processing.
#### Standardize RGB
Use libraries like:
pandas & numpy (data handling)
scikit-learn (preprocessing, feature engineering)
opencv, PIL (image processing)
### Challenges 
#### Incomplete Data
Missing values that need to be addressed.
#### Noisy Data
Irrelevant or erroneous information that can mislead the model.
#### Inconsistent Data:
#### Imbalanced Data
Uneven distribution of classes, leading to biased models.

## 3. Model Training
Train models (e.g., Logistic Regression, BERT)
### Text - NLP
nltk, spaCy, transformers for ad text processing
```python
import spacy

# Load pre-trained NLP model
nlp = spacy.load("en_core_web_sm")

text_ad = "Buy weight-loss pills now! Limited offer!"
doc = nlp(text_ad)

# Keyword-based filtering
banned_keywords = {"weight-loss", "pills", "limited offer"}
detected_keywords = {token.text.lower() for token in doc if token.text.lower() in banned_keywords}

if detected_keywords:
    print("Flagged for moderation:", detected_keywords)
else:
    print("Ad is compliant.")
```
### Image & Video -CNN
image: computer Vision and image recognition models 
video: frame-by-fame, audio analysis
### Hybrid models
combining both for video content
## 4. Model Evaluation & Tuning
### Precision and Recall
### F1 - score

![image-20250225212856152](/Users/zhoulingwei/Library/Application Support/typora-user-images/image-20250225212856152.png)

### Confidence Threasholding
Metrics: Precision, Recall, F1-score (sklearn.metrics)
Confidence Thresholding: Adjusting thresholds to balance accuracy vs. moderation costs
## 5.  Human-in-the-Loop (HITL) & Active Learning
1. send low-confidence ads to moderators
2. correction feedback to be new training data

# According to What Kind of Features Does AI Label Content?
## Textual Features
Keywords, phrases, and language patterns.
## Visual Features
Objects, scenes, and activities detected in images or videos.
## Audio Features
Spoken words, tone, and sound patterns.
## Behavioral Features
User interaction metrics like shares, comments, and views.
## Metadata:
Author Information, Creation Date and Time, Geographical Location, Device Information
As described above, information about the content's origin, creator, and context.
# Moderation Labels
## Compliant
Adheres to all platform policies and guidelines.
## Non-Compliant
Violates one or more policies.
## Sensitive
May be appropriate but contains content that could be upsetting or controversial.
## Pending Review
