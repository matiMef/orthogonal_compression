import numpy as np

def numpy_fft(cropped_image, M):
  Transformation = np.fft.fft2(cropped_image)
  mask = np.abs(Transformation) >= M
  Transformation_compressed = Transformation * mask
  fft_image = np.real(np.fft.ifft2(Transformation_compressed))
  return fft_image