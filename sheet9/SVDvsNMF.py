import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import NMF
from numpy.linalg import svd

digits = load_digits()

x = digits['data'] / 255
y = digits['target']
mean_x = np.mean(x, axis=0)
centered_x = x - mean_x

#singular value decomposition
#rows of v are eigenvectors
#(usually columns are ev after complex conjugations but we only have real numbers)
u, s, v = svd(centered_x)
v += mean_x

#largest eigenvector in image format
ev = v[0]
ev.shape = (8, 8)

#non-negative matrix factorization
nmf = NMF(n_components=10)
nmf.fit(x)
h = nmf.components_

#first component in image format
c0 = h[0].reshape((8, 8))
plt.imshow(c0)
plt.show()

#compare first 6 eigenvectors with first 6 components
v6 = v[0:6]
h6 = h[0:6]

f , ax = plt.subplots(2)
plt.suptitle('first rows of matrixes V and H')
ax[0].imshow(v6)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('first six eigenvectors')
ax[1].imshow(h6)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('first six NMF components')
plt.show()

#all eigenvectors in image format
f, axes = plt.subplots(1, 6)
plt.suptitle('first six eigenvectors in image format')
for i, ax in enumerate(axes):
    ax.imshow(v[i].reshape((8, 8)))
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

#all NMF-components in image format
f, axes = plt.subplots(1, 6)
plt.suptitle('first six NMF-components in image format')
for i, ax in enumerate(axes):
    ax.imshow(h[i].reshape((8, 8)))
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()