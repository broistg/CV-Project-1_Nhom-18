import numpy as np

def convolution2D(img, kernel):
  img_height, img_width = img.shape[:2]
  kernel_height, kernel_width = kernel.shape[:2]

  output_height = img_height - kernel_height + 1
  output_width = img_width - kernel_width + 1

  output_img = np.zeros((output_height, output_width))

  for i in range(output_height):
    for j in range(output_width):
      region = img[i:i+kernel_height, j:j+kernel_width]

      output_img[i, j] = np.sum(region * kernel)

  return output_img

def normalization(img):
  return np.uint8((img - np.min(img)) / (np.max(img) - np.min(img)) * 255)