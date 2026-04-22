import numpy as np

def sft(cropped_image, keep_fraction=0.05):
  H, W = cropped_image.shape
  k = max(int(H * W * keep_fraction), 1)
  
  n_buckets = max(4 * k, 64)
  n_buckets = int(2 ** np.ceil(np.log2(n_buckets)))
  n_iterations = 3
  tolerance = 1e-6
  
  rng = np.random.default_rng(42)
  a = int(rng.integers(1, max(H // 2, 2)) * 2 + 1)
  b = int(rng.integers(0, H))
  row_perm = (a * np.arange(H) + b) % H
  col_perm = (a * np.arange(W) + b) % W
  permuted = cropped_image[np.ix_(row_perm, col_perm)]
  
  residual_spectrum = np.fft.fft2(permuted)
  found_freqs = {}
  
  per_iter = max(k // n_iterations, 1)
  stride_h = max(H // n_buckets, 1)
  stride_w = max(W // n_buckets, 1)
  
  for _ in range(n_iterations):
    if len(found_freqs) >= k: break
    
    bucket_spectrum = sft_hash_to_buckets(np.fft.ifft2(residual_spectrum).real, n_buckets)
    power = np.abs(bucket_spectrum) ** 2
    max_power = power.max()
    if max_power == 0: break
    threshold = tolerance * max_power
    
    candidates =[]
    active_buckets = np.argwhere(power > threshold)
    
    for bh, bw in active_buckets:
      fh_start, fw_start = int(bh) * stride_h, int(bw) * stride_w
      fh_range = range(fh_start, min(fh_start + stride_h, H))
      fw_range = range(fw_start, min(fw_start + stride_w, W))
      
      for fh in fh_range:
        for fw in fw_range:
          if (fh, fw) not in found_freqs:
            p = abs(residual_spectrum[fh, fw]) ** 2
            candidates.append((p, fh, fw))
            
    candidates.sort(key=lambda x: -x[0])
    new_count = 0
    for p, fh, fw in candidates:
      if len(found_freqs) >= k or new_count >= per_iter: break
      if (fh, fw) in found_freqs: continue
      
      found_freqs[(fh, fw)] = residual_spectrum[fh, fw]
      residual_spectrum[fh, fw] = 0.0 
      new_count += 1
      
  if len(found_freqs) < k:
    magnitudes = np.abs(residual_spectrum)
    flat_idx = np.argsort(magnitudes, axis=None)[::-1]
    for idx in flat_idx:
      if len(found_freqs) >= k: break
      fh, fw = np.unravel_index(idx, (H, W))
      if (fh, fw) not in found_freqs:
        found_freqs[(fh, fw)] = residual_spectrum[fh, fw]
        
  sparse_spectrum = np.zeros((H, W), dtype=complex)
  for (fh, fw), val in found_freqs.items():
    sparse_spectrum[fh, fw] = val
    
  permuted_recon = np.fft.ifft2(sparse_spectrum).real
  
  inv_row = np.argsort((a * np.arange(H) + b) % H)
  inv_col = np.argsort((a * np.arange(W) + b) % W)
  reconstructed = permuted_recon[np.ix_(inv_row, inv_col)]
  
  return np.clip(reconstructed, 0, 255)

def sft_hash_to_buckets(signal_2d, n_buckets):
  H, W = signal_2d.shape
  stride_h = max(H // n_buckets, 1)
  stride_w = max(W // n_buckets, 1)
  subsampled = signal_2d[::stride_h, ::stride_w]
  return np.fft.fft2(subsampled)