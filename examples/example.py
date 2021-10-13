import cv2
from PIL import Image

from clip_search.search import ClipSearch

clipsearch = ClipSearch(model_name="RN50")

img_paths = ["resources/rock.jpg", "resources/smallapple.jpg"]
img1 = cv2.imread(img_paths[0])
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = Image.open(img_paths[1])
images = [img1, img2]

texts = [
    "an image of an apple",
    "an image of an orange",
    "an image of a muscular man",
    "an image of a pretty lady",
    "an image of WWE WRESTLER THE ROCK",
]

img_feats = clipsearch.encode_images(images)
text_feats = clipsearch.encode_texts(texts)

print()
# query with only the 1st image
print(f"Querying texts with 1st image: {img_paths[0]}")
probs_img_query = clipsearch.query_with_feats(img_feats[0], text_feats)

for text, prob in sorted(zip(texts, probs_img_query), key=lambda x: x[1], reverse=True):
    print(f"{prob*100:.0f}%: {text}")

print()
# query with only the 1st text
print(f"Querying images with 1st text: {texts[0]}")
probs_text_query = clipsearch.query_with_feats(img_feats, text_feats[0])

for imgpath, prob in sorted(
    zip(img_paths, probs_text_query), key=lambda x: x[1], reverse=True
):
    print(f"{prob*100:.0f}%: {imgpath}")
