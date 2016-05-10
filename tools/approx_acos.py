#!/usr/bin/env python3

# Convector -- An interactive CPU path tracer
# Copyright 2016 Ruud van Asseldonk

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3. A copy
# of the License is available in the root of the repository.

# The goal is to approximate acos(x) with a rational function f such that the
# worst absolute error is minimal. That is, pick the function that performs best
# in the worst case. Furthermore, I impose the following restrictions:
#
#  * f(0) = pi/2. This implies that the constant term is pi/2.
#  * f(1) = 0 and f(-1) = pi. This implies that (a + b) / (1 + c + d) = -pi/2.

from mpmath import mp, fabs, acos
from scipy.optimize import minimize

mp.prec = 64

def d(a, b, c):
    return -1 - 2 * (a + b) / mp.pi - c

def f(x, a, b, c):
    return mp.pi/2 + (a * x + b * x**3) / (1 + c * x**2 + d(a, b, c) * x**4)

def error(coefs, progress=True):
    (a, b, c) = coefs
    xs = (x / mp.mpf(4096) for x in range(-4096, 4097))
    err = max(fabs(acos(x) - f(x, a, b, c)) for x in xs)
    if progress:
        print('(a, b, c, d): ({}, {}, {}, {})'.format(a, b, c, d(a, b, c)))
        print('evaluated error: ', err)
        print()
    return float(err)

initial_guess = (-0.9823, 0.9421, -1.1851)
coefs = minimize(error, initial_guess).x
print('a:', coefs[0])
print('b:', coefs[1])
print('c:', coefs[2])
print('d:', d(*coefs))
print('max error:', error(coefs, progress=False))

# Output:
#
#     a: -0.939115566365855,
#     b: 0.9217841528914573,
#     c: -1.2845906244690837,
#     d: 0.295624144969963174
#     max error:  0.0167244179117447796
