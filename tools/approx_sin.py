#!/usr/bin/env python3

# The goal is to approximate sin(x) with a polynomial f on the domain (-pi, pi),
# such that the worst absolute error is minimal. That is, pick the function that
# performs best in the worst case. Furthermore, I impose the following
# restrictions:
#
#  * f(0)  = 0. This implies that the constant term is 0.
#  * f(pi) = 0 and f(-pi) = 0. This implies that c = -(a*pi + b*pi^3) / pi^5.

from mpmath import mp, fabs, sin
from scipy.optimize import minimize

mp.prec = 64

def c(a, b):
    return - (a * mp.pi + b * mp.pi**3) / mp.pi**5

def f(x, a, b):
    return a * x + b * x**3 + c(a, b) * x**5

def error(coefs, progress=True):
    (a, b) = coefs
    xs = (x * mp.pi / mp.mpf(4096) for x in range(-4096, 4097))
    err = max(fabs(sin(x) - f(x, a, b)) for x in xs)
    if progress:
        print('(a, b, c): ({}, {}, {})'.format(a, b, c(a, b)))
        print('evaluated error: ', err)
        print()
    return float(err)

initial_guess = (0.9820, -0.1522)
coefs = minimize(error, initial_guess).x
print('a:', coefs[0])
print('b:', coefs[1])
print('c:', c(*coefs))
print('max error:', error(coefs, progress=False))

# Output:
#
#     a: 0.982012145975
#     b: -0.152178468117
#     c: 0.00533758325004438232
#     max error: 0.008109495819698682
