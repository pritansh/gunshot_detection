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
- Chromagram {c_stft} (Default - 12 coeff)
- Constant Q-Chromagram {c_cqt} (Default - 12 coeff)
- Chroma Energy Normalized {c_cens} (Default - 12 coeff)
- Melspectrogram {mel_spec} (128 coeff)
- Mel Frequency Cepstral Coefficients {mfcc} (Default - 20 coeff)
- Root Mean Square Energy {rmse} (1 coeff)
- Spectral Bandwidth {bandwidth} (1 coeff)
- Spectral Centroid {centroid} (1 coeff)
- Spectral Contrast {contrast} (1 coeff)
- Spectral Rolloff {rolloff} (1 coeff)
- Poly features {poly} (Default - 1 order (order+1 coeff))
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

- Supervised
    - Linear Discriminant Analysis {lda} (classes-1 Features)

### Neural Network
- Multilayer Perceptron