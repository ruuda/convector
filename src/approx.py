#!/usr/bin/env python3

# The goal is to approximate acos(x) with a rational function f such that the
# worst absolute error is minimal. That is, pick the function that performs best
# in the worst case. Furthermore, I impose the following restrictions:
#
#  * f(0) = pi/2. This implies that the constant term is pi/2.
#  * f(1) = 0 and f(-1) = pi. This implies that (a + b + c) / (1 + d) = -pi/2.

from mpmath import mp, fabs, acos, isinf
from scipy.optimize import minimize

mp.prec = 64

def d(a, b, c):
    return -1 - 2 * (a + b + c) / mp.pi

def f(x, a, b, c):
    return mp.pi/2 + (a * x + b * x**3 + c * x**5) / (1 + d(a, b, c) * x**2)

def error(coefs, progress=True):
    (a, b, c) = coefs
    xs = (x / mp.mpf(4096) for x in range(-4096, 4097))
    err = max(fabs(acos(x) - f(x, a, b, c)) for x in xs)
    if progress:
        print('(a, b, c, d): ({}, {}, {}, {})'.format(a, b, c, d(a, b, c)))
        print('evaluated error: ', err)
        print()
    return float(err)

initial_guess = (-0.96967, 0.65129, 0.26605)
coefs = minimize(error, initial_guess).x
print('a:', coefs[0])
print('b:', coefs[1])
print('c:', coefs[2])
print('d:', d(*coefs))
print('max error:', error(coefs, progress=False))

# Output:
#
#     a: -0.94525109533
#     b: 0.585786288561
#     c: 0.315822007008
#     d: -0.97221613075101857
#     max error: 0.02240680886961469
