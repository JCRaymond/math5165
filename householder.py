
from math import sqrt

def norm2(v):
   return sum(map(lambda x: x*x, v))

def norm(v):
   return sqrt(norm2(v))

def Householder(u):
   res = [[0]*len(u) for i in range(len(u))]
   u_norm2 = norm2(u)

   for i in range(len(u)):
      res[i][i] = 1
      u_i = u[i]
      for j in range(len(u)):
         res[i][j] -= 2 * u_i * u[j] / u_norm2

   return res

def rot_to_idx(x, idx):
   return Householder([x[i] - (norm(x) if i == idx else 0) for i in range(len(x))])

def mul(mat, vec):
   res = [0]*len(mat)
   for i in range(len(mat)):
      for j in range(len(vec)):
         res[i] += mat[i][j]*vec[j]
   return res


x = [1,2,3]
A = rot_to_idx(x, 0)
y = mul(A, x)
print(x)
print(y)

