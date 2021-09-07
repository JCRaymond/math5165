#!/usr/bin/env python3.9

import numpy as np

def normalize(vec):
   return vec / np.linalg.norm(vec)

def power_eigenvec(A, epsilon=1e-6):
   if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
      return None

   n = A.shape[0]
   x = normalize([np.ones(n)]).T
   
   while True:
      x_old = x
      x = normalize(A * (A * x))

      dist = np.linalg.norm(x - x_old, ord=float('inf'))
      if dist < epsilon:
         return x

def power_e_val_vec(A, epsilon=1e-6):
   e_vec = power_eigenvec(A)
   val_vec = (A*e_vec)/e_vec
   e_val = max(val_vec.max(), val_vec.min(), key=abs)
   return e_val, e_vec


if __name__ == "__main__":
   A = np.matrix([
      [4, 0, 0],
      [0, 1, 0],
      [0, 0, -5]
   ])

   val, vec = power_e_val_vec(A)

   print(val, vec)

