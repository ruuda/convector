#!/usr/bin/env python3

# Convector -- An interactive CPU path tracer
# Copyright 2016 Ruud van Asseldonk

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3. A copy
# of the License is available in the root of the repository.

# The goal is to approximate cos(x) with a polynomial f on the domain (-pi, pi),
# such that the worst absolute error is minimal. That is, pick the function that
# performs best in the worst case. Furthermore, I impose the following
# restrictions:
#
#  * f(0) = 1. This implies that the constant term is 1.
#  * f(pi) = -1 and f(-pi) = -1. This implies that
#    c = -(2 + a*pi^2 + b*pi^4) / pi^6.

from mpmath import mp, cos, fabs
from scipy.optimize import minimize

mp.prec = 64

def c(a, b):
    return - (2.0 + a * mp.pi**2 + b * mp.pi**4) / mp.pi**6

def f(x, a, b):
    return 1.0 + a * x**2 + b * x**4 + c(a, b) * x**6

def error(coefs, progress=True):
    (a, b) = coefs
    xs = (x * mp.pi / mp.mpf(4096) for x in range(-4096, 4097))
    err = max(fabs(cos(x) - f(x, a, b)) for x in xs)
    if progress:
        print('(a, b, c): ({}, {}, {})'.format(a, b, c(a, b)))
        print('evaluated error: ', err)
        print()
    return float(err)

initial_guess = (-0.4960, 0.03926)
coefs = minimize(error, initial_guess).x
print('a:', coefs[0])
print('b:', coefs[1])
print('c:', c(*coefs))
print('max error:', error(coefs, progress=False))

# Output:
#
#     a: -0.496000299455
#     b: 0.0392596924214
#     c: -0.000966231179636657107
#     max error: 0.0020164493561441203
