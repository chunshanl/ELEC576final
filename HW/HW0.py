import numpy as np
import scipy.linalg
from scipy import signal

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

a = np.array([[1., 2., 3.], [4., 5., 6.]])
print(np.ndim(a))
print(np.size(a))

# use print for output if the code is run in pycharm
# no need to add print if it is run in ipython
np.shape(a)
a.shape[2 - 1]

b = a
c = a
d = a
a = np.block([[a, b], [c, d]])
a

a[-1]
a[1, 2]
a[1, :]
a[0:1, :]
a[-1:]
a[0:3][:, 2:4]
a[np.ix_([1, 3], [0, 2])]
a[0:3:2, :]
a[::2, :]
a[::-1, :]
a[np.r_[:len(a), 0]]
a.T
a.conj().T

a = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
b = a
a @ b
a * b
a / b
a ** 3
(a > 0.5)
np.nonzero(a > 0.5)
v = np.array([0.1, 1, 2])
a[:, np.nonzero(v > 0.5)[0]]
a[:, v.T > 0.5]
a[a < 0.5] = 0
a * (a > 0.5)
a[:] = 3

x = a
x.copy()
x[1, :].copy()
x.flatten()

np.arange(1., 11.)
np.arange(10.)
np.r_[:10.]
np.arange(1., 11.)[:, np.newaxis]
np.zeros((3, 4))
np.zeros((3, 4, 5))
np.ones((3, 4))
np.eye(3)

a = np.arange(1., 11.)
np.diag(a)
np.diag(a, 0)

np.random.rand(3, 4)
np.linspace(1, 3, 4)
np.mgrid[0:9., 0:6.]
np.meshgrid([1, 2, 4], [2, 4, 5])

np.tile(a, (2, 3))

a = np.eye(3)
b = a
a
b
np.column_stack((a, b))
np.vstack((a, b))
a.max()
a.max(0)
a.max(1)
np.maximum(a, b)

v = np.array([1., 2., 3.])
np.sqrt(v @ v)

np.logical_and(a, b)
np.logical_or(a, b)

np.linalg.inv(a)
np.linalg.pinv(a)
np.linalg.matrix_rank(a)
np.linalg.solve(a, b)
b = a
U, S, Vh = np.linalg.svd(a)
V = Vh.T
U
S
V
np.linalg.cholesky(a).T
D, V = np.linalg.eig(a)
D
V
D, V = scipy.linalg.eig(a, b)
D
V
Q, R = scipy.linalg.qr(a)
Q
R

a = np.arange(1., 11.)
a
np.fft.fft(a)
np.fft.ifft(a)

a = np.eye(3)
np.sort(a)
I = np.argsort(a[:, 0])
b = a[I, :]
b

a = np.arange(1., 11.)
np.sort(a)

X = np.random.rand(50, 2)
y = np.random.rand(50, 1)
np.linalg.lstsq(X, y, rcond=None)

y = np.random.rand(1, 50)
scipy.signal.resample(y, 40)

a = np.zeros((2, 3, 4))
np.unique(a)
a.squeeze()
