import numpy as np

ti = np.array([1, 2, 3, 4])
print(ti)
# array([1, 2, 3, 4])
print(ti.dtype)
# dtype('int64')
tf = np.array([1.5, 2.5, 3.5, 4.5])
print(tf.dtype)
# dtype('float64')
tf2d = np.array([[1.5, 2, 3], [4, 5, 6]])
print(tf2d)
t = np.array([1, 2, 3, 4, 5, 6])
print(t[2])
t = np.array([[1,2,3], [4,5,6]])
print(t[0]) # 1ère ligne
print(t[0, 1])
# Créer un tableau de 3 lignes et 4 colonnes rempli de zéros
tz2d = np.zeros((3,4))
print(tz2d)
# array([[ 0., 0., 0., 0.],
# [ 0., 0., 0., 0.],
# [ 0., 0., 0., 0.]])
# Créer un tableau de 3 lignes et 4 colonnes rempli de uns
tu2d = np.ones((3,4))
print(tu2d)
# array([[ 1., 1., 1., 1.],
# [ 1., 1., 1., 1.],
# [ 1., 1., 1., 1.]])
id2d = np.eye(5)
print(id2d)
# matrice identité d’ordre 5
tni2d = np.empty((3,4))
print(tni2d)
# matrice initialisée par des 1
ts1d = np.arange(0, 40, 5)
print(ts1d)
# array([ 0, 5, 10, 15, 20, 25, 30, 35])
ts1d2 = np.linspace(0, 35, 8)
print(ts1d2)
# array([ 0., 5., 10., 15., 20., 25., 30., 35.])
ta2d = np.random.rand(3,5)
print(ta2d)
tr = np.arange(20)
print(tr)
# array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
13, 14, 15, 16,
# 17, 18, 19])
# Réordonne le tableau en une matrice à 4 lignes et 5colonnes
tr = tr.reshape(4,5)
print(tr)
# array([[ 0, 1, 2, 3, 4],
# [ 5, 6, 7, 8, 9],
# [10, 11, 12, 13, 14],
# [15, 16, 17, 18, 19]])
# Réordonne le tableau en une matrice à 2 lignes et 10 colonnes
print(tr.reshape(2,10))
# array([[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
print(tr.reshape(20))
ta1d = np.random.rand(5)
print(ta1d)
# array([0.68431206, 0.22183178, 0.50668827, 0.40924377, 0.35185467])
ta3d = np.random.rand(2,3,5)
print(ta3d)
a = tr[:2]
print(a)
# array([0, 1])
a[0] = 3
print(tr)
