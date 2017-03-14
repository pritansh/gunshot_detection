# Gunshot Detection

#### python 2.7

### Install requisites

```python
pip install -r requirements.txt

```

### Initial setup

```python
python setup.py

```

### Features
- Chroma Stft (Default - 12 coeff)
- Melspectrogram (128 coeff)
- Mel Frequency Cepstral Coefficients (MFCC) (Default - 20 coeff)
- Root Mean Square Energy (RMSE) (1 coeff)
- Spectral Bandwidth (1 coeff)
- Spectral Centroid (1 coeff)
- Spectral Contrast (1 coeff)
- Spectral Rolloff (1 coeff)
- Poly features (Default - 1 order (order+1 coeff))
- Zero Crossing Rate (ZCR) (1 coeff)

### Vector Reduction
- Mean
- IQR
- Variance

### Feature Reduction (Default - 10 Features)
- Principal Component Analysis (PCA)
- Incremental PCA (IPCA)
- Kernel PCA (KPCA)
- Fast Independent Component Analysis (FICA)
- Truncated SVD (TSVD)
- Non-negative Matrix Factorization (NMF)
- Sparse PCA (SPCA)
- Canonical Correlation Analysis (CCA)
- Partial Least Square SVD(PLSSVD)
- Linear Discriminant Analysis (LDA) (classes-1 Features)

### Neural Network
- Multilayer Perceptron