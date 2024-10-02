from model import VGG
import utils

import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

def run_style_transfer(num_steps, display_step, content_path, style_path):
    # Define hyperparameters
    learning_rate = 1e-2
    style_weight = 1
    content_weight = 1e-5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 512
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    style_img = utils.load_image(style_path, transform=transform).to(device)
    content_img = utils.load_image(content_path, transform=transform).to(device)
    output_image = content_img.clone()
    noise = torch.rand_like(output_image, device=device) * 0.8
    output_image = output_image + noise
    output_image = output_image.requires_grad_(True).to(device)

    model = VGG()
    model.to(device)
    optimizer = optim.Adam(params=[output_image], lr=learning_rate)

    model.eval()
    for step in range(num_steps):
        generated_features = model(output_image)
        content_features = model(content_img)
        style_features = model(style_img)

        content_loss = style_loss = 0

        for gen_feature, content_feature, style_feature in zip(generated_features, content_features, style_features):
            # Content loss
            content_loss += F.mse_loss(content_feature, gen_feature)

            # Style loss
            G_gram_matrix = utils.compute_gram_matrix(gen_feature)
            S_gram_matrix = utils.compute_gram_matrix(style_feature)
            style_loss += F.mse_loss(S_gram_matrix, G_gram_matrix)

        total_loss = style_weight*style_loss + content_weight*content_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (step+1)%display_step == 0:
            # save_image(output_image, f"static/generated_output.jpg")
            print(f"Step: {step+1}. total_loss: {total_loss}, content_loss: {content_loss}, style_loss: {style_loss}")
    save_image(output_image, f"static/generated_output.png")
    return utils.to_img(output_image)

# if __name__ == "__main__":
#     run_style_transfer()