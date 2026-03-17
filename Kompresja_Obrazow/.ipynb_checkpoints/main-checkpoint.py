import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import skimage.color as color
from scipy.fftpack import dct, idct

matrix_size = 8
path = "test_model.jpg"
image = io.imread(path)
image = color.rgb2gray(image)

def crop_image(image, crop_size):
  h, w = image.shape
  new_h = (h // crop_size) * crop_size
  new_w = (w // crop_size) * crop_size
  return image[:new_h, :new_w]

image = crop_image(image, matrix_size)

def splint_to_bloks(image, block_size):
  h, w = image.shape
  splited = np.reshape(image, (h // block_size, block_size, w // block_size, block_size))
  splited = np.transpose(splited,(0,2,1,3))
  return splited

img = splint_to_bloks(image, matrix_size)

#wyświetlenie fragmnetu obrazu z podziałem na bloki
for i in range(15):
  for j in range(15):
    plt.subplot(15, 15, i*15 + j + 1)
    plt.imshow(img[i,j], cmap='gray')
    plt.axis('off')
matplotlib.pyplot.imshow(img[15,0], cmap='gray')
matplotlib.pyplot.show()
print(type(image))
print(image.shape)

img_h = np.shape(img)[0]
img_w = np.shape(img)[1]

def dct_compression(img, img_h, img_w):
 with np.printoptions(edgeitems=8, precision=2, linewidth=1000):
  
  img_block = img[img_h, img_w]

  M = img_block.shape[0]
  N = img_block.shape[1]
  PI = np.pi

  alfa_p = np.full((M, 1), np.sqrt(2/M))
  alfa_q = np.full((1, N), np.sqrt(2/N))
  alfa_p[0, 0] = 1/np.sqrt(M)
  alfa_q[0, 0] = 1/np.sqrt(N)

  m=np.arange(M)
  p=np.arange(M)
  m=np.reshape(m, (1, -1))
  p=np.reshape(p, (-1, 1))
  T = np.cos((PI * (2*m+1) * p) / (2*M))

  T[0, :] /= np.sqrt(M)         
  T[1:, :] /= np.sqrt(M / 2)
 
  B = (T @ img_block @ T.T)
  
  mask_size = M//2
  mask = np.zeros((M,N))
  for i in range (0, mask_size):
    for j in range(mask_size-i):
      mask[i, j] = 1
  
  B_masked = B * mask

  A_compressed = T.T @ B_masked @ T
  return A_compressed

img_compressed = np.zeros_like(img)

for i in range(img_h):
   for j in range(img_w):
    block = dct_compression(img, i, j)
    img_compressed[i, j] = block 

def scipy_dct(image_input):
 with np.printoptions(edgeitems=8, precision=2, linewidth=1000):
  B = dct(dct(image_input.T, norm='ortho').T, norm='ortho')

  mask_size = matrix_size//2
  mask = np.zeros((matrix_size,matrix_size))
  for i in range (0, mask_size):
    for j in range(mask_size-i):
      mask[i, j] = 1
    
  B_compressed = B * mask

  A_compressed = idct(idct(B_compressed, norm='ortho').T, norm='ortho').T
  return A_compressed

img_compressed2 = np.zeros_like(img)
for i in range(img_h):
   for j in range(img_w):
    block2 = scipy_dct(img[i,j])
    img_compressed2[i, j] = block2

def to_image(image, block_size):
  h = image.shape[0]
  w = image.shape[1]
  temp = image.transpose(0, 2, 1, 3)
  combined = np.reshape(temp, (h * block_size, w * block_size))
  return combined

new_image = to_image(img_compressed, 8)
new_image2 = to_image(img_compressed2, 8)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Przed kompresją")
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Po kompresji")
plt.imshow(new_image, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Po kompresji 2")
plt.imshow(new_image2, cmap='gray')
plt.show()

