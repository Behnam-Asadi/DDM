import os
import getopt
import torch
from torchvision import transforms
from torchvision.transforms import Compose, Lambda
from datasets import load_dataset
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
import sys


def parse_arguments(argv):
    generated_images_path = "../../output/cifar10/x0"  # Default path
    try:
        opts, args = getopt.getopt(argv, "hp:", ["path="])
    except getopt.GetoptError:
        print('evaluate.py -p <path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('evaluate.py -p <path>')
            sys.exit()
        elif opt in ("-p", "--path"):
            generated_images_path = arg
    return generated_images_path

def transform():
    return Compose([
        transforms.ToTensor(),
    ])

def load_cifar10(transform, device):
    dataset = load_dataset("cifar10")
    transformed_images = [transform(image['img']).to(device) for image in dataset['test']]
    return torch.stack(transformed_images)

def load_generated_images(path, transform, device):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
    images = [transform(Image.open(p).convert('RGB')).to(device) for p in image_paths]
    return torch.stack(images)

def evaluate_generated_images(generated_images_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    transform_func = transform()
    fake_images = load_generated_images(generated_images_path, transform_func, device)
    real_images = load_cifar10(transform_func, device)
    print("Real images shape:", real_images.shape)
    print("Fake images shape:", fake_images.shape)

    fid = FrechetInceptionDistance(normalize=True).to(device)
    batch_size = 512
    # Update FID with fake images in batches
    for i in range(0, len(fake_images), batch_size):
        batch_fake_images = fake_images[i:i + batch_size]
        fid.update(batch_fake_images, real=False)

    # Update FID with real images in batches
    for i in range(0, len(real_images), batch_size):
        batch_real_images = real_images[i:i + batch_size]
        fid.update(batch_real_images, real=True)

    fid_score = float(fid.compute())
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    path = parse_arguments(sys.argv[1:])
    evaluate_generated_images(path)
