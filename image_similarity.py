import torch
# CLIP (Contrastive Language-Image Pre-Training) from the openAI 
import open_clip
import cv2
# util module is used for working with embeddings
from sentence_transformers import util
from PIL import Image

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create the model and preprocess function
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained="laion2b_s32b_b82k")
model.to(device)

def imageEncoder(img):
    """
        This function takes image in numpy array, convert to 'rgb' format.
        The preprocess function is obtained from the open_clip.create_model_and_transforms 
        function and includes steps such as resizing, normalization, .unsqueeze(0) adds an extra 
        dimension to the tensor, making it a batch of size 1. The encode_image method outputs a feature vector.
    """

    img = Image.fromarray(img).convert('RGB')
    img = preprocess(img).unsqueeze(0).to(device)
    img = model.encode_image(img)
    return img


def generateScore(image1, image2):
    """
    pytorch_cos_sim computes the cosine similarity between the two tensors img1 and img2.
    float(cos_scores[0][0])*100 converts the extracted score to a percentage.

    """

    image1 = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(image1)
    img2 = imageEncoder(image2)

    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0])*100, 2)
    
    return score


image1 = "/Users/apple/Downloads/zillow_images/32fb530f721f28213c730a10b06604da-cc_ft_384.jpg"
image2 = "/Users/apple/Downloads/zillow_images/84a5c8a912092be2efef26fe1a9d7085-uncropped_scaled_within_1344_1008.jpg"

print(f"similarity Score: ", round(generateScore(image1, image2), 2))
