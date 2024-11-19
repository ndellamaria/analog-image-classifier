import os
import cv2
import numpy as np
from PIL import Image
import random
from pathlib import Path

def apply_blur(image):
    """
    Args:
        image: OpenCV image array
    Returns:
        tuple: (augmented image, label)
    """
    intensity_sigma = random.uniform(7, 10)
    kernel_size = random.randrange(73, 101, 2)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), intensity_sigma), 'blurry'


def apply_light_exposure(image):
    # Create gradient mask
    height, width = image.shape[:2]
    gradient = np.zeros((height, width), dtype=np.float32)
    for i in range(width):
        gradient[:, i] = i / width
    
    # Expand gradient to 3 channels
    gradient = cv2.merge([gradient, gradient, gradient])
    
    # Create yellow-orange light leak color
    light_color = np.full_like(image, [50, 200, 255], dtype=np.float32)  # BGR
    
    # Combine original image with light leak
    intensity = random.uniform(.5, 1.0)
    light_leak = cv2.addWeighted(
        image.astype(np.float32), 1.0,
        gradient * light_color, intensity,
        0
    )
    
    # Add slight overexposure
    light_leak = cv2.convertScaleAbs(light_leak, alpha=1.2, beta=10)
    
    return light_leak, 'light_exposure'

def apply_over_exposed(image):
    """
    Args:
        image: OpenCV image array
    Returns:
        tuple: (augmented image, label)
    """
    intensity = random.uniform(50,100)
    return cv2.convertScaleAbs(image, alpha=1.5, beta=intensity), 'over_exposed'

def apply_under_exposed(img):

    grain_intensity = np.random.uniform(25, 45)
    brightness_factor=random.uniform(0.1, 0.7)
    blue_tint=1.2

    # Convert to float32 for processing
    img = img.astype(np.float32) / 255.0
    
    # Add grain
    noise = np.random.normal(0, grain_intensity/255.0, img.shape)
    img = np.clip(img + noise, 0, 1)
    
    # Adjust brightness
    img = img * brightness_factor
    
    # Add slight blue tint (adjust B channel in BGR)
    img[:,:,0] = np.clip(img[:,:,0] * blue_tint, 0, 1)
    
    # Add vignette effect
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    img = img * mask[:,:,np.newaxis]
    
    # Convert back to uint8
    img = (img * 255).astype(np.uint8)
    
    return cv2.convertScaleAbs(img), 'under_exposed'

def process_images(input_dir, output_dir):
    """
    Process images with augmentations and save with labels.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create labels file
    with open(os.path.join(output_dir, 'labels.txt'), 'w') as label_file:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)
                cv_image = cv2.imread(image_path)
                
                if cv_image is not None:
                    # Save raw image and label as 'good'
                    label = 'good'
                    output_filename = f'{label}_{filename}'
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, cv_image)
                    label_file.write(f'{output_filename},{label}\n')

                    # Define augmentations
                    augmentations = [
                        (apply_blur, 'blurry'),
                        (apply_light_exposure, 'light_exposure'),
                        (apply_over_exposed, 'over_exposed'),
                        (apply_under_exposed, 'under_exposed')
                    ]

                    # Apply augmentations and save images
                    for augmentation, label in augmentations:
                        augmented_image, label = augmentation(cv_image)
                        output_filename = f'{label}_{filename}'
                        output_path = os.path.join(output_dir, output_filename)
                        cv2.imwrite(output_path, augmented_image)
                        label_file.write(f'{output_filename},{label}\n')


if __name__ == "__main__":
    input_directory = "raw_images"
    output_directory = "augmented_images"
    
    process_images(input_directory, output_directory)