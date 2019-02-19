import numpy as np
import math
import operator
import itertools
from scipy import interpolate
import scipy
import scipy.optimize
from math import sqrt
import random
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
#import pylab as pl
#import statsmodels.api as sm
#from phasespace import *

class Bin:
    def __init__(self):
        self.n = 0
        self.y = 0.
        self.y2 = 0.
        self.e2 = 0.

    def add(self, y, e=0.):
        self.n += 1
        self.y += y
        self.y2 += y * y
        self.e2 += e * e

    def gety(self, sample=False):
        if self.n == 0:
            return float('nan')
        if sample:
            return random.gauss(*self.getres())
        return self.y / self.n

    def gete(self):
        if self.n > 1:
            return math.sqrt((self.y2 - self.y ** 2 / self.n) / (self.n - 1) + self.e2 / self.n ** 2)
        if self.n > 0:
            return math.sqrt(self.e2) / self.n
        return float('nan')

    def getres(self):
        return (self.gety(), self.gete())

    def __str__(self):
        return str(str(self.n) + ' ' + str(self.getres()))

class Grid:
    def __init__(self, method, dim, xmin, xmax, nbin, data=[], binned=False):
        self.dim = dim
        self.xmin = np.array(xmin, dtype=float)
        self.xmax = np.array(xmax, dtype=float)
        self.nbin = np.array(nbin)
        self.binned = binned
        self.method = method
        self.lowclip = 1e-20 # error and 1/error clipped to [lowclip,highclip], this improves the numerical stability of the fit
        self.highclip = 1e20

        self.dx = np.abs((self.xmin - self.xmax).astype(float) / self.nbin)

        if (not binned):
            self.nbin = self.nbin + 1

        self.data = {}
        self.deg = {}  # degree of interpolation poynomial
        self.pol = {}  # coefficients of interpolation polynomial
        for k in np.ndindex(tuple(self.nbin)):
            self.data[k] = Bin()

        if (binned):
            if (data):
                self.addPoint(data)
        else:
            nneighbor = 5
            nsamples = 1

            def linfunc(x, a, b, c):
                return a + x[0] * b + x[1] * c

            for _ in range(nsamples):
                if nsamples > 1:
                    dat = np.array(
                        [np.append(d[:self.dim], [random.gauss(d[self.dim], d[self.dim + 1]), d[self.dim + 1]]) for
                         d in data])  # dat = data with y values sampled according to gauss(y,e)
                else:
                    dat = data
                for k in np.ndindex(tuple(self.nbin)):
                    x = self.xmin + k * self.dx
                    all_points = np.array(sorted(dat, key=lambda p: np.linalg.norm(p[:self.dim] - x)))
                    nearest_points = all_points[:nneighbor]

                    bin_lower =  self.xmin + k * self.dx
                    bin_upper = self.xmin + (np.array(k)+1) * self.dx

                    # For last bin along axis we will consider only points below the bin
                    shift = np.zeros(len(self.nbin))
                    for i, elem in enumerate(k):
                        # last bin along this axis
                        if elem == self.nbin[i] - 1:
                            shift[i] = 1
                    bin_lower -= shift * self.dx
                    bin_upper -= shift * self.dx

                    # Include points in fit that fall into current bin
                    bin_points = all_points[nneighbor:]
                    bin_points = bin_points[bin_points[:,0] > bin_lower[0]]
                    bin_points = bin_points[bin_points[:,1] > bin_lower[1]]
                    bin_points = bin_points[bin_points[:,0] < bin_upper[0]]
                    bin_points = bin_points[bin_points[:,1] < bin_upper[1]]
                    points = np.vstack((nearest_points,bin_points))

                    X = points[:, 0:2]
                    Y = points[:, 2]
                    E = 1. / np.clip(points[:, 3] ** 2, self.lowclip, self.highclip)  # * np.linalg.norm(points[:,:self.dim]- x)**(-2)
                    # sigma=[ p[3] * np.linalg.norm(p[:self.dim]-x)**2 for p in points] # with absolue_sigma=False, each point will contribute with relative weights w=1/sigma**2, non-linear dependence will increase quadracically with the distance and can be taken into account in the weight
                    sigma = np.clip(points[:, 3] ** 2, self.lowclip, self.highclip)
                    X = np.array([xx - x for xx in X]).T

                    popt, pcov2 = scipy.optimize.curve_fit(linfunc, X, Y, sigma=sigma, absolute_sigma=True)
                    popt, pcov1 = scipy.optimize.curve_fit(linfunc, X, Y, sigma=sigma, absolute_sigma=False)

                    # fig = plt.figure(dpi=100)
                    # ax = fig.add_subplot(111, projection='3d')
                    # fig.suptitle('Binned Points, Bin = ' + str(k))
                    # ax.plot([x[0]], [x[1]], [popt[0]], linestyle="None", marker="x")
                    # ax.plot(X[0]+x[0], X[1]+x[1], Y, linestyle="None", marker="o")
                    # ax.plot((np.array([xx for xx in dat]).T)[0]+x[0], (np.array([xx for xx in dat]).T)[1]+x[1], (np.array([xx for xx in dat]).T)[2], linestyle="None", marker="X")
                    # #for i in np.arange(0, len(X[0])):
                    # #    ax.plot([Xnoshift[0][i], Xnoshift[0][i]], [Xnoshift[1][i], Xnoshift[1][i]],
                    # #            [Y[i] + sigma[i], Y[i] - sigma[i]], marker="_")
                    # Xtest = np.arange(0, 1, 0.01)
                    # Ytest = np.arange(0, 1, 0.01)
                    # Xtest, Ytest = np.meshgrid(Xtest, Ytest)
                    # print('Xtest',Xtest)
                    # result = linfunc( (Xtest-x[0], Ytest-x[1]), popt[0], popt[1], popt[2])
                    # ax.plot_wireframe(Xtest, Ytest, result, rstride=10, cstride=10)
                    # ax.set_xlabel('beta')
                    # ax.set_ylabel('cos(theta)')
                    # ax.set_xlim3d(-1.,1.)
                    # ax.set_ylim3d(-1.,1.)
                    # ax.set_zlim3d(0.9*min(Y), 1.1*max(Y))
                    # plt.show()

                    # Warn about negative first entries in covariance matrix (only possible due to numerical instability)
                    if pcov1[0,0] < 0. or pcov2[0,0] < 0.:
                        print('WARNING: Negative covariance matrix in bin ' + str(k))
                        print(pcov1)
                        print(pcov2)
                        print('nearest_points',nearest_points)
                        print('bin_points',bin_points)

                    #self.data[k].add(linfunc((x[0],x[1]),popt[0],popt[1],popt[2]), max(sqrt(abs(pcov1[0, 0])), sqrt(abs(pcov2[0, 0]))))  # corresponds to multiplying error estimate of chi-sq fit with max(1,chisq)
                    self.data[k].add(popt[0], max(sqrt(abs(pcov1[0, 0])), sqrt(abs(pcov2[0, 0]))))  # corresponds to multiplying error estimate of chi-sq fit with max(1,chisq)
                    '''
                       model y = f(x) = c
                       y1   e1     y2   e2     popt   pcov1   pcov2    chisq
                       1    1e-5   2    1e-5   1.5    5e-11   0.25     5e9      # pcov2 = pcov1*chisq
                       0.9  1      1.1  1      1      0.5     0.0025   0.02
                       ==> use cov2 if chisq>>1, cov1 if chisq<1
                    '''

    def polyfit2d(self, x, y, z, orders):
        ncols = np.prod(orders + 1)
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(orders[0] + 1), range(orders[1] + 1))
        for k, (i, j) in enumerate(ij):
            G[:, k] = x ** i * y ** j
        m, _, _, _ = np.linalg.lstsq(G, z)
        return m

    def polyval2d(self, x, y, m, orders):
        ij = itertools.product(range(orders[0] + 1), range(orders[1] + 1))
        z = np.zeros_like(x)
        for a, (i, j) in zip(m, ij):
            z += a * x ** i * y ** j
        return z

    def x(self, k):  # coordinate after variable transformation applied to s,t
        if (self.binned):
            return (self.xmin + (np.array(k) + 0.5) * self.dx)
        else:
            return (self.xmin + (np.array(k) + 0.) * self.dx)

    def k(self, x):  # index of data array
        #    if (not all(x>self.xmin and x < self.xmax)):
        #      raise Exception('xout of bounds: ',x)
        return tuple((np.minimum((x - self.xmin) / self.dx, self.nbin - 1)).astype(int))

    def gridpoints(self, sample=False, flag='k', extendToBorder=False, returnError=False):
        if flag == 'k':
            return [[k, d.gety(sample)] for k, d in self.data.iteritems()]
        if flag == 'x':
            if (returnError):
                res = [[self.x(k), d.gety(sample), d.gete()] for k, d in self.data.iteritems()]
            else:
                res = [[self.x(k), d.gety(sample)] for k, d in self.data.iteritems()]
            if (extendToBorder):
                for d in range(self.dim):
                    nbin = self.nbin.copy()
                    nbin[d] = 1
                    for k in np.ndindex(tuple(nbin)):
                        x = self.x(k)
                        x[d] = self.xmin[d]
                        res += [[x.copy(), self.data[k].gety(sample)], ]
                        k = list(k)
                        k[d] = self.nbin[d] - 1
                        x[d] = self.xmax[d]
                        res += [[x.copy(), self.data[tuple(k)].gety(sample)], ]
                        #            print "b", self.xmin,self.xmax
                for corner in np.ndindex(tuple([2, ] * self.dim)):
                    k = corner * (self.nbin - 1)
                    x = np.array(corner, float)
                    res += [[x.copy(), self.data[tuple(k)].gety(sample)], ]
            return res
        if flag == 'plain':
            return [self.x(k).tolist() + [d.gety(sample), ] for k, d in self.data.iteritems()]

    def printgrid(self):
        for k, d in self.data.iteritems():
            print('k: ' + str(k) + ' d: ' + str(d) + ' d.n:' + str(d.n))

    def addPoint(self, data):
        if (type(data[-1]) is float or type(data[-1]) is np.float64):
            assert (len(data) == self.dim + 2)
            xx = data[0:self.dim]
            kk = self.k(xx)
            self.data[kk].add(*data[-2:])
        elif (type(data[-1]) is np.ndarray):
            for d in data:
                self.addPoint(d)

    def initInterpolation(self, nsamples=1):
        self.interpolators = []
        self.nsamples = nsamples
        for i in range(nsamples):
            temp = zip(*self.gridpoints(sample=(nsamples != 1), flag='x', extendToBorder=True))
            self.interpolators.append(interpolate.CloughTocher2DInterpolator(list(temp[0]), temp[1], fill_value=0.))

            # polynomial interpolation
            nneighbor = 2
            for k in self.data:
                kk = np.asarray(k)
                kmin = np.maximum(kk - np.full(self.dim, nneighbor, dtype=int), np.zeros(self.dim)).astype(int)
                kmax = np.minimum(kk + np.full(self.dim, nneighbor, dtype=int), self.nbin - 1).astype(int)
                degree = kmax - kmin
                self.deg[k] = degree

                xx = self.x(k)
                dat = [np.append(self.x(x + kmin), self.data[tuple(x + kmin)].gety(sample=(nsamples != 1))) for x in
                       np.ndindex(*(degree + 1))]
                x, y, z = np.asarray(zip(*dat))
                self.pol[k, i] = self.polyfit2d(x, y, z, orders=degree)

    def interpolate(self, x, y, selectsample=-1):
        # Return just the bin central value (for testing)
        # k=self.k(tuple([x, y]))
        # print('k',k)
        # print('self.data[k].gety()',self.data[k].gety())
        # return (self.data[k].gety(), 0.)
        if (self.method == 2):
            k = self.k(tuple([x, y]))
            temp = [self.polyval2d(x, y, self.pol[k, i], self.deg[k]) for i in range(self.nsamples)]
        else:
            temp = [interpolator(x, y) for interpolator in self.interpolators]
        if (selectsample == -1):
            return (np.mean(temp), np.std(temp))
        else:
            return (temp[selectsample], 0.)

    def __call__(self, X, Y, selectsample=-1):
        temp = np.array([X, Y]).T
        s = np.shape(X)
        temp = np.array([self.interpolate(x, y, selectsample)[0] for x, y in
                         np.array([x for x in np.vstack([X.ravel(), Y.ravel()])]).T]).reshape(s)
        return temp

class CreateGrid:
    def __init__(self,selected_grid):
        self.selected_grid = selected_grid
        self.selected_grid_dirname = os.path.dirname(self.selected_grid)
        self.mHs = 125.**2
        self.method = 1  # 1: CloughTocher;  2: Polynomial
        self.polydegree = 2
        self.tolerance = 1.e-8 # costh is nudged to 1 if within this tolerance

        # cdf: transformation x=f(beta) leading to uniform distributions of points in x
        self.cdfdata = np.loadtxt(os.path.join(self.selected_grid_dirname,'events.cdf'))
        self.cdf = interpolate.splrep(*self.cdfdata.T, k=1)

        x0, x1, y, e = np.loadtxt(self.selected_grid, unpack=True)

        # print('== Plotting Input Points ==')
        # fig = plt.figure(dpi=100)
        # ax = fig.add_subplot(111, projection='3d')
        # fig.suptitle('Input points')
        # ax.plot(x0, x1, y, linestyle="None", marker="o")
        # #for i in np.arange(0, len(x0)):
        # #    ax.plot([x0[i], x0[i]], [x1[i], x1[i]],
        # #            [y[i] + e[i], y[i] - e[i]], marker="_")
        # ax.set_xlabel('beta')
        # ax.set_ylabel('cos(theta)')
        # ax.set_xlim3d(0., 1.)
        # ax.set_ylim3d(0., 1.)
        # plt.show()

        x0Uniform = interpolate.splev(x0, self.cdf)

        # print('== Plotting Transformed Points ==')
        # fig = plt.figure(dpi=100)
        # ax = fig.add_subplot(111, projection='3d')
        # fig.suptitle('Transformed points')
        # ax.plot(x0Uniform, x1, y, linestyle="None", marker="o")
        # # for i in np.arange(0, len(x0Uniform)):
        # #     ax.plot([x0Uniform[i], x0Uniform[i]], [x1[i], x1[i]],
        # #             [y[i] + e[i], y[i] - e[i]], marker="_")
        # ax.set_xlabel('beta')
        # ax.set_ylabel('cos(theta)')
        # ax.set_xlim3d(0., 1.)
        # ax.set_ylim3d(0., 1.)
        # plt.show()

        dataInUniformFlat = np.array([x0Uniform, x1, y, e]).T
        self.interpolator = Grid(self.method, 2, (0, 0), (1, 1), (100, 30), data=dataInUniformFlat, binned=False)
        self.interpolator.initInterpolation(1)

    def u(self, s, t):
        return 2 * self.mHs - s - t

    def beta(self, s):
        return math.sqrt(1. - 4. * self.mHs / s)

    def costh(self, s, t):
        res = (t - self.u(s, t)) / (s * self.beta(s))
        if (abs(res) - 1.) > 0. :
            if (abs(res) - 1.) < self.tolerance:
                if res < 0.:
                    res = -1.
                else:
                    res = 1.
            else:
                raise ValueError("Grid called with cos(theta) > 1, cos(theta) = {:.40e}".format(res))
        return res

    def GetAmplitude(self, s, t):
        b = self.beta(s)
        cth = abs(self.costh(s, t))
        xUniform = interpolate.splev(b, self.cdf)
        return self.interpolator.interpolate(xUniform, cth)[0]

    def TestClosure(self):
        b, cth, y, e = np.loadtxt(self.selected_grid, unpack=True)
        percent_deviations = []
        std_deviations = []
        for b_value, cth_value, y_value, e_value in zip(b,cth,y,e):
            xUniform = interpolate.splev(b_value, self.cdf)
            percent_deviations.append( 100.*(y_value-self.interpolator.interpolate(xUniform, cth_value)[0])/y_value )
            std_deviations.append( (y_value-self.interpolator.interpolate(xUniform, cth_value)[0])/e_value )
            print (b_value, cth_value, y_value, self.interpolator.interpolate(xUniform, cth_value))
        percent_deviations = np.array(percent_deviations)
        std_deviations = np.array(std_deviations)

        #
        # Plot of grid points vs input points
        #
        x0Uniform = interpolate.splev(b, self.cdf)
        y_grid = []
        for x,c in zip(x0Uniform,cth):
            y_grid.append(self.interpolator.interpolate(x, c)[0])
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle('Closure test')
        ax.plot(x0Uniform, cth, y, linestyle="None", marker="o")
        ax.plot(x0Uniform, cth, y_grid, linestyle="None", marker="x")
        # Error bars
        # for i in np.arange(0, len(x0Uniform)):
        #     ax.plot([x0Uniform[i], x0Uniform[i]], [cth[i], cth[i]],
        #             [y[i] + e[i], y[i] - e[i]], marker="_")
        ax.set_xlabel('beta')
        ax.set_ylabel('cos(theta)')
        ax.set_xlim3d(0., 1.)
        ax.set_ylim3d(0., 1.)
        plt.show()

        #
        # Percent deviation of grid from central value of point
        #
        print(percent_deviations)
        bins = np.arange(-51,51,2)
        _, _, _ = plt.hist(np.clip(percent_deviations, bins[0], bins[-1]), bins=bins, facecolor='r', alpha=0.75, edgecolor='black', linewidth=1.2)
        plt.xlabel('difference [%]')
        plt.ylabel('Npoints')
        plt.show()

        #
        # Number of standard deviations between grid and point
        #
        print(std_deviations)
        bins = np.arange(-11,11,2)
        _, _, _ = plt.hist(np.clip(std_deviations, bins[0], bins[-1]), bins=bins, facecolor='r', alpha=0.75, edgecolor='black', linewidth=1.2)
        plt.xlabel('Standard Deviation')
        plt.ylabel('Npoints')
        plt.show()
