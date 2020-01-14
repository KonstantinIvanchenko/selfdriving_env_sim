#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Author: Konstantin Ivanchenko
# Date: December 25, 2019

import math
import numpy as np
from scipy.interpolate import interp1d


class CubicSplineParam(object):
    """
    Parametrized cubic spline class.
    :param
    points : []
        List of point objects
    interp_granul : int
        Sets the total amount of output points interpolated with the fit spline
    """
    def __init__(self, points, interp_granul):
        self.points = np.array([[points[i].x for i in range(len(points))],
                                [points[i].y for i in range(len(points))],
                                [points[i].s for i in range(len(points))]]).T

        temp = np.diff(self.points, axis=0 )

        self.d = np.cumsum(np.sqrt(np.sum( np.diff(self.points, axis=0  )**2, axis=1 )))
        self.d = np.insert(self.d, 0, 0)/self.d[-1]
        self.alpha = np.linspace(0, 1, interp_granul)
        self.method = 'cubic'
        try:
            self.interpolator = interp1d(self.d, self.points, kind=self.method, axis=0)
        except Exception as e:
            print(e, " 1dSpline interpolation error")

    """
    Interpolates with the fit spline.
    :param : None
    :return : []
        Array of interpolated points
    """
    def interpolate(self):
        interp_points = self.interpolator(self.alpha)
        return interp_points

    """
    Interpolates with the fit spline as split X,Y arrays
    :param : None
    :return : [], []
        Array of points X
        Array of points Y
    """
    def interpolate_wp(self):
        new_points = self.interpolate().T
        return new_points[0], new_points[1], new_points[2]


class CubicsplineT1(object):
    def __init__(self, points):
        self.p = points
        self.splines = []

        self.get_splines()

    def hi(self, i):
        return self.p[i].x-self.p[i-1].x

    def vi(self, i):
        return self.p[i].y-self.p[i-1].y

    def initialize_splines(self):
        np1=len(self.p)
        n=np1-1

        #X=self.p[:].x
        X = [self.p[i].x for i in range(len(self.p))]
        #Y=self.p[:].y
        Y = [self.p[i].y for i in range(len(self.p))]
        #a=Y[:]
        a = Y[:]
        b = [0.0] * (n)
        d = [0.0] * (n)
        h = [X[i+1]-X[i] for i in range(n)]
        alpha = [0.0]*n
        for i in range(1, n):
            alpha[i] = 3 / h[i] * (a[i + 1] - a[i]) - 3 / h[i - 1] * (a[i] - a[i - 1])
        c = [0.0] * np1
        L = [0.0] * np1
        u = [0.0] * np1
        z = [0.0] * np1
        L[0] = 1.0;
        u[0] = z[0] = 0.0
        for i in range(1, n):
            L[i] = 2 * (X[i + 1] - X[i - 1]) - h[i - 1] * u[i - 1]
            u[i] = h[i] / L[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / L[i]
        L[n] = 1.0;
        z[n] = c[n] = 0.0
        for j in range(n - 1, -1, -1):
            c[j] = z[j] - u[j] * c[j + 1]
            b[j] = (a[j + 1] - a[j]) / h[j] - (h[j] * (c[j + 1] + 2 * c[j])) / 3
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])
        for i in range(n):
            self.splines.append((a[i], b[i], c[i], d[i], X[i]))

    def get_splines(self):
        self.initialize_splines()

    def get_y(self, x, spl_i):
        xi = x - self.splines[spl_i][4]
        a = self.splines[spl_i][0]
        b = self.splines[spl_i][1]
        c = self.splines[spl_i][2]
        d = self.splines[spl_i][3]
        y = d * xi ** 3 + c * xi ** 2 + b * xi + a
        return y



class CubicsplineT2(object):
    def __init__(self, points):
        self.p = points
        self.A = list()
        self.B = list()
        self.C = list()  #
        self.F = list()
        self.h = list() # defined for i=0,..,n-1; h1=x1-x0
        self.v = list()

        self.a = list()
        self.b = list()
        self.c = list()
        self.d = list()
        self.alfa = list()
        self.beta = list()
        self.get_splines()

    def hi(self, i):
        return self.p[i].x-self.p[i-1].x

    def vi(self, i):
        return self.p[i].y-self.p[i-1].y

    def initialize_splines(self):
        self.A.append(0) # insert dummies
        self.C.append(0)
        self.B.append(0)
        self.F.append(0)

        # Total amount of alpha and beta shall be n-1
        self.alfa.append(0) # alpha[0] = 0
        self.beta.append(0) # alpha[1] = 0
        #####self.c.append(0)    # c[0] = 0
        # Total amount of alpha and beta shall be n-1
        for i in range(0, len(self.p), 1):
            self.a.append(self.p[i].y)

        for i in range(1, len(self.p)-1, 1):
            hi = self.hi(i)
            hi_n = self.hi(i+1)
            vi = self.vi(i)
            vi_n = self.vi(i+1)
            self.h.append(hi)


            self.A.append(hi) # start from A[1]
            self.C.append(2*(hi+hi_n)) # start from C[1]
            self.B.append(hi_n)
            self.F.append(6 * (vi_n / hi_n - vi / hi))

            # i as it starts from 1
            prop_coef_div = self.A[i]*self.alfa[i-1]+self.C[i]
            self.alfa.append(-self.B[i]/prop_coef_div)
            self.beta.append((self.F[i]-self.A[i]*self.beta[i-1])/prop_coef_div)

        # back propagate alfa and beta coefficients
        # total amount of correct alfa and beta coefs is n-2.
        # But actual count n-1 (except alpa0=0 and beta0=0)
        num = len(self.p)

        # append the last h for future use of splines
        self.h.append(self.hi(num-1))

        # start from the last spline n-1
        cn_m_1 = (self.F[num-2] - self.A[num-2]*self.beta[num-2])\
                / (self.C[num-2] + self.A[num-2]*self.alfa[num-2])
        self.c.append(cn_m_1)

        for i in range(num-2, 0, -1):
            ci = self.alfa[i]*self.c[0]+self.beta[i]
            self.c.insert(0, ci)  # append from left

        # the c0 shall be 0
        #self.c.insert(0, 0)

        # total amount of c elements is n-1
        # generate di and bi elements
        for i in range(num-1, 0, -1):
            hi = self.hi(i)
            vi = self.vi(i)

            di = (self.c[i]-self.c[i-1])/hi
            bi = (2*self.c[i] + self.c[i-1])*hi/6 + vi/hi
            self.d.append(di)
            self.b.append(bi)

    def get_splines(self):
        self.initialize_splines()

    def get_y(self, x, spl_i):

        xi = x - self.p[spl_i].x  # ##self.h[spl_i]
        return (xi**3)*self.d[spl_i] + (xi**2)*self.c[spl_i] + xi*self.b[spl_i] + self.a[spl_i]

