# Orthogonal Expansions in Image Processing and Compression

This project focuses on the implementation and comparative analysis of orthogonal algorithms for image compression, specifically DCT (Discrete Cosine Transform), FFT (Fast Fourier Transform), and SFT (Sparse Fourier Transform). The goal is to visualize compression effects and understand how orthogonal transforms reduce data in digital images.

## Objectives
* Implementation of DCT and SFT algorithms.
* Comparative analysis of compression efficiency and quality across different methods.
* Visualization of quantization and thresholding effects on test images.

## Theoretical Background
The project utilizes orthogonality properties to decompose images into independent frequency components:
* DCT: The foundation of the JPEG standard; it divides images into blocks (e.g., 8x8) and concentrates signal energy into low frequencies.
* FFT: Uses complex numbers to decompose an image into a sum of sinusoidal functions.
* SFT: An optimized version for sparse signals, allowing for rapid identification of dominant spectral components.

## Technologies and Libraries
The project is implemented in Python using the following libraries:
* NumPy: For matrix and numerical calculations.
* Scikit-image: For image loading and processing operations.
* SciPy: For reference implementations of transforms.
* Matplotlib: For data visualization and generating result plots.

## Project Structure
* algorithms/: Custom implementations of DCT, FFT, and SFT.
* utils/: Tools for timing, image preprocessing, and visualization.
* test_models/: Sample images used for testing.
* Orthogonal_Compression.ipynb: Main notebook containing the analysis and results.

## Potential Applications
1. Optimization of data transmission for web applications.
2. Preprocessing for Machine Learning models to decrease training time.
3. Efficient storage of digital media through lossy compression techniques.
