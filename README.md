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
- Chromagram {c_stft} (Default - 12 coeff) (chroma_coeff)
- Constant Q-Chromagram {c_cqt} (Default - 12 coeff) (chroma_coeff)
- Chroma Energy Normalized {c_cens} (Default - 12 coeff) (chroma_coeff)
- Melspectrogram {mel_spec} (128 coeff)
- Mel Frequency Cepstral Coefficients {mfcc} (Default - 20 coeff) (mfcc_coeff)
- Root Mean Square Energy {rmse} (1 coeff)
- Spectral Bandwidth {bandwidth} (1 coeff)
- Spectral Centroid {centroid} (1 coeff)
- Spectral Contrast {contrast} (Default - 6 bands (bands+1 coeff)) (contrast_bands)
- Spectral Rolloff {rolloff} (1 coeff)
- Poly features {poly} (Default - 1 order (order+1 coeff)) (poly_order)
- Tonnetz {tonnetz} (6 coeff)
- Zero Crossing Rate {zcr} (1 coeff)

### Vector Reduction
- Mean {mean}
- IQR {iqr} (Default - 50 percentile)
- Variance {var}

### Feature Reduction (Default - 10 Features)
- Unsupervised
    - Principal Component Analysis {pca}
    - Incremental PCA {ipca}
    - Kernel PCA {kpca}
    - Fast Independent Component Analysis {fica}
    - Truncated SVD {tsvd}
    - Non-negative Matrix Factorization {nmf}
    - Sparse PCA {spca}
    - Canonical Correlation Analysis {cca}
    - Partial Least Square SVD {plssvd}
    - Gaussian Random Projection {grp}
    - Sparse Random Projection {srp}

- Supervised
    - Linear Discriminant Analysis {lda} (classes-1 Features)

### Neural Network
- Multilayer Perceptron

### Support Vector Machines
- Types
    - C SVM
    - Nu SVM

- Kernels
    - Linear {linear}
    - Polynomial {poly} (Default 3 degree) (poly_degree)
    - Radial Basis Function {rbf}
    - Sigmoid {sigmoid}