from PIL import Image
import torchvision.transforms as transform
import torch

def load_image(image_path, transform):
    image = Image.open(image_path) 
    image = transform(image).unsqueeze(0)
    return image

def to_img(img_tensor):
    img = img_tensor.cpu().clone()
    img = img.squeeze(0)
    img = transform.ToPILImage()(img)
    return img

def compute_gram_matrix(feature_map):
    batch, c, w, h = feature_map.shape
    feature = feature_map.view(c, w*h)
    gram_matrix = torch.mm(feature, feature.T)
    # Normalize
    # gram_matrix = gram_matrix/(c*w*h)
    return gram_matrix