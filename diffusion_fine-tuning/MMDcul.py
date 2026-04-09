import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel
from pathlib import Path
from PIL import Image
from torchvision import transforms, models

# Load ResNet50
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the final fully connected layer
resnet.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(640),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract image features
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = resnet(image).view(-1).numpy()  # Convert to a 1D feature vector
    return features

# Load images from a folder and extract features
def load_images_from_folder(folder_path):
    images = []
    for filename in Path(folder_path).glob("*.png"):  # Assume image files are in PNG format
        features = extract_features(filename)
        images.append(features)
    return np.array(images)

# Compute the Gaussian kernel
def compute_rbf_kernel(X, Y, gamma=1000.0):
    return rbf_kernel(X, Y, gamma=gamma)

# Compute the MMD value
def compute_mmd(X, Y, gamma=1000.0):
    n = X.shape[0]  # Size of the real dataset
    m = Y.shape[0]  # Size of the generated dataset
    
    # Compute kernel matrices
    K_XX = compute_rbf_kernel(X, X, gamma)
    K_XY = compute_rbf_kernel(X, Y, gamma)
    K_YY = compute_rbf_kernel(Y, Y, gamma)

    print(f"Max value of X_np: {np.max(X)}")
    print(f"Min value of X_np: {np.min(X)}")

    print(f"Max value of Y_np: {np.max(Y)}")
    print(f"Min value of Y_np: {np.min(Y)}")

    # Print the first 20 values of X_np and Y_np
    print(f"First 20 values of X_np: {['{:.3f}'.format(x) for x in X.flatten()[:20]]}")
    print(f"First 20 values of Y_np: {['{:.3f}'.format(x) for x in Y.flatten()[:20]]}")
    print(f"K_XX (first 20 elements): {['{:.3f}'.format(x) for x in K_XX.flatten()[:20]]}")
    print(f"K_XY (first 20 elements): {['{:.3f}'.format(x) for x in K_XY.flatten()[:20]]}")
    print(f"K_YY (first 20 elements): {['{:.3f}'.format(x) for x in K_YY.flatten()[:20]]}")
    print(f"K_XX shape: {K_XX.shape}, total elements: {K_XX.size}")
    print(f"K_XY shape: {K_XY.shape}, total elements: {K_XY.size}")
    print(f"K_YY shape: {K_YY.shape}, total elements: {K_YY.size}")
    
    # Compute the MMD value according to the formula
    term1 = np.sum(K_XX) / (n**2)
    term2 = 2 * np.sum(K_XY) / (n * m)
    term3 = np.sum(K_YY) / (m**2)
    
    mmd_value = term1 - term2 + term3
    return mmd_value

# Folder paths (please modify them according to your actual paths)
real_data_folder = "path/to/diffusion_fine-tuning/data1"
generated_data_folder = "path/to/diffusion_image"

# Extract features
real_features = load_images_from_folder(real_data_folder)
generated_features = load_images_from_folder(generated_data_folder)

# Compute the MMD value
mmd_value = compute_mmd(real_features, generated_features, gamma=0.01)
print(f"MMD between real and generated data: {mmd_value}")