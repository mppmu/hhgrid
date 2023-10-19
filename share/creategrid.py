import numpy as np
import re
import math
import operator
import itertools
from scipy import interpolate
import scipy
import scipy.optimize
from math import sqrt
import random
import os, time
import lhapdf
import yaml

def combinegrids(grid_temp, cHHH, ct, ctt, cg, cgg, EFTcount=3, usesmeft=0, lhaid=90400, renfac=1.):

    print(f'py: combinegrids, called for: {grid_temp}')

    # Grid exists, proceed
    if os.path.exists(grid_temp):
        print(f'py: combinegrids, grid already exists: {grid_temp}')
        return

   #    -- Because the lock mechanism was not robust enough,
   #    -- the grids are now combined in one single process (by
   #    -- calling combinegrids() only once in the run warm-up
   #    -- phase).

    # Produce grid

    # Grid numbering format
    np.set_printoptions(formatter={'float': '{:.18E}'.format})

    print("******************************************************************************************")
    print(" Combining grids for chhh = " + str(cHHH) + ", ct = " + str(ct) + ", ctt = " + str(ctt) +", cg = " + str(cg) + ", cgg = " + str(cgg))
    print("******************************************************************************************")

    # Build grid for give value of cHHH
    incr_dirname, incr_grid_name = os.path.split(grid_temp)
    incr = os.path.join( incr_dirname, incr_grid_name[:9])
    cHHH_grids = [  incr + '_+5.000000E-01_+4.782609E-01_+1.000000E+00_+6.875000E-01_+8.888889E-01.grid',
  incr + '_-1.000000E+00_-2.500000E+00_+2.857143E-01_+4.666667E-01_+1.818182E-01.grid',
  incr + '_+4.545455E-02_-5.263158E-02_-6.875000E-01_+2.941176E-01_+6.923077E-01.grid',
  incr + '_+1.250000E-01_+1.111111E-01_-3.333333E-01_+2.173913E-01_-8.000000E-01.grid',
  incr + '_-1.142857E+00_+4.444444E-01_+9.090909E-02_+7.272727E-01_+6.428571E-01.grid',
  incr + '_+8.333333E-01_+4.545455E-01_-4.285714E-01_+9.523810E-02_+5.263158E-01.grid',
  incr + '_+1.000000E+00_+1.428571E-01_-3.333333E-01_+5.217391E-01_-1.500000E+00.grid',
  incr + '_-5.263158E-02_-7.142857E-02_+2.083333E-01_+6.923077E-01_+9.000000E-01.grid',
  incr + '_-3.750000E-01_+4.705882E-01_-3.750000E-01_-6.666667E-01_+4.166667E-01.grid',
  incr + '_-6.470588E-01_-8.333333E-01_+5.000000E-01_-1.333333E+00_+5.000000E-01.grid',
  incr + '_-6.666667E-01_+1.666667E-01_+1.100000E+00_+5.000000E-01_+1.000000E+00.grid',
  incr + '_+2.000000E-01_+6.000000E-01_+8.695652E-02_-3.478261E-01_-1.333333E+00.grid',
  incr + '_-4.210526E-01_+4.444444E-01_-2.222222E-01_+3.809524E-01_-5.714286E-01.grid',
  incr + '_-1.176471E-01_+2.200000E+00_-7.692308E-02_-1.875000E-01_+5.555556E-01.grid',
  incr + '_+2.000000E-01_-8.571429E-01_-1.000000E+00_+3.125000E-01_-1.166667E+00.grid',
  incr + '_+3.000000E-01_+1.111111E-01_+1.285714E+00_+1.285714E+00_-4.615385E-01.grid',
  incr + '_-4.347826E-01_-8.000000E-01_+1.111111E-01_-6.315789E-01_+4.347826E-02.grid',
  incr + '_-1.142857E+00_-3.333333E-01_-5.000000E-01_-5.000000E-01_+4.117647E-01.grid',
  incr + '_+2.250000E+00_-6.666667E-01_+2.727273E-01_+3.571429E-01_-1.000000E+00.grid',
  incr + '_+6.111111E-01_+2.777778E-01_+1.111111E-01_-8.000000E-01_+2.272727E-01.grid',
  incr + '_+2.173913E-01_+3.000000E+00_-5.263158E-01_+4.761905E-02_-3.809524E-01.grid',
  incr + '_+4.545455E-01_+4.000000E-01_-1.500000E+00_+5.454545E-01_+6.428571E-01.grid',
  incr + '_+2.500000E-01_+1.111111E+00_-4.166667E-01_-4.444444E-01_+5.000000E-02.grid']

    amps  = []
    ME2s  = []
    dME2s = []

    for grid in cHHH_grids:
        print(grid)
        amps.append(np.loadtxt(grid, unpack=True))
    print("Grids loaded.")

    C = np.array([[14641/279841., 1., 121/2116., 121/1024., 64/81., 
  121/529., 1331/24334., 11/46., 11/32., 8/9., 
  1331/16928., 968/4761., 121/1472., 44/207., 11/36., 
  14641/194672., 121/368., 1331/11776., 121/414., 
  14641/135424., 121/256., 1331/8192., 121/288.], [625/16., 
  4/49., 25/4., 49/225., 4/121., 25/14., 125/8., 5/7., 
  -2/15., 4/77., -35/12., 25/22., -7/6., 5/11., 
  -14/165., -175/24., -1/3., 49/90., -7/33., 49/36., 
  14/225., -343/3375., 98/2475.], [1/130321., 121/256., 
  1/174724., 25/139876., 81/169., -11/5776., -1/150898., 
  1/608., -5/544., -99/208., 5/135014., 9/4693., 
  -5/156332., -9/5434., 45/4862., -5/116603., 55/5168., 
  -25/120802., -45/4199., 25/104329., -275/4624., 
  125/108086., 225/3757.], [1/6561., 1/9., 1/5184., 
  25/33856., 16/25., -1/243., 1/5832., -1/216., -5/552., 
  4/15., 5/14904., -4/405., 5/13248., -1/90., -1/46., 
  5/16767., -5/621., 25/38088., -4/207., 25/42849., 
  -25/1587., 125/97336., -20/529.], [256/6561., 1/121., 
  1024/3969., 4096/5929., 81/196., 16/891., -512/5103., 
  -32/693., -64/847., 9/154., -1024/6237., 8/63., 
  2048/4851., -16/49., -288/539., 512/8019., 32/1089., 
  -2048/7623., 16/77., 1024/9801., 64/1331., -4096/9317., 
  288/847.], [625/14641., 9/49., 625/4356., 25/3969., 
  100/361., -75/847., 625/7986., -25/154., -5/147., 
  -30/133., 125/7623., 250/2299., 125/4158., 125/627., 
  50/1197., 250/27951., -10/539., 50/14553., 100/4389., 
  100/53361., -4/1029., 20/27783., 40/8379.], [1/2401., 
  1/9., 1/49., 144/529., 9/4., -1/147., 1/343., -1/21., 
  -4/23., 1/2., 12/1127., -3/98., 12/161., -3/14., 
  -18/23., 12/7889., -4/161., 144/3703., -18/161., 
  144/25921., -48/529., 1728/12167., -216/529.], [1/38416., 
  25/576., 1/70756., 81/61009., 81/100., 5/4704., 
  1/52136., 5/6384., -15/1976., 3/16., -9/48412., 9/1960.,
   -9/65702., 9/2660., -81/2470., -9/35672., -15/1456., 
  81/44954., -81/1820., 81/33124., 135/1352., -729/41743., 
  729/1690.], [4096/83521., 9/64., 9/289., 1/16., 25/144.,
   -24/289., -192/4913., 9/136., -3/32., -5/32., 16/289., 
  80/867., -3/68., -5/68., 5/48., -1024/14739., 2/17., 
  -4/51., -20/153., 256/2601., -1/6., 1/9., 
  5/27.], [625/1296., 1/4., 3025/10404., 1936/2601., 1/4.,
   25/72., 1375/3672., 55/204., 22/51., 1/4., 275/459., 
  25/72., 1210/2601., 55/204., 22/51., 125/162., 5/9., 
  440/459., 5/9., 100/81., 8/9., 704/459., 
  8/9.], [1/1296., 121/100., 1/81., 1/9., 1., 11/360., 
  -1/324., -11/90., -11/30., 11/10., -1/108., 1/36., 
  1/27., -1/9., -1/3., 1/432., 11/120., -1/36., 1/12., 
  1/144., 11/40., -1/12., 1/4.], [81/625., 4/529., 
  9/625., 64/13225., 16/9., 18/575., 27/625., 6/575., 
  -16/2645., -8/69., -72/2875., -12/25., -24/2875., 
  -4/25., 32/345., -216/2875., -48/2645., 192/13225., 
  32/115., 576/13225., 128/12167., -512/60835., 
  -256/1587.], [256/6561., 4/81., 1024/29241., 4096/159201.,
   16/49., -32/729., -512/13851., 64/1539., 128/3591., 
  8/63., -1024/32319., -64/567., 2048/68229., 128/1197., 
  256/2793., 512/15309., -64/1701., -2048/75411., 
  -128/1323., 1024/35721., -128/3969., -4096/175959., 
  -256/3087.], [14641/625., 1/169., 484/7225., 9/18496., 
  25/81., -121/325., -2662/2125., 22/1105., -3/1768., 
  -5/117., 363/3400., 121/45., -33/5780., -22/153., 
  5/408., -3993/2000., 33/1040., -99/10880., -11/48., 
  1089/6400., -9/3328., 27/34816., 5/256.], [1296/2401., 
  1., 36/1225., 1/256., 49/36., -36/49., -216/1715., 
  6/35., -1/16., 7/6., 9/196., -6/7., -3/280., 1/5., 
  -7/96., -135/686., 15/56., -15/896., 5/16., 225/3136., 
  -25/256., 25/4096., -175/1536.], [1/6561., 81/49., 
  1/900., 729/4900., 36/169., 1/63., 1/2430., 3/70., 
  243/490., -54/91., 1/210., -2/351., 9/700., -1/65., 
  -81/455., 1/567., 9/49., 27/490., -6/91., 1/49., 
  729/343., 2187/3430., -486/637.], [256/625., 1/81., 
  64/529., 14400/190969., 1/529., 16/225., 128/575., 
  8/207., 40/1311., 1/207., 384/2185., 16/575., 
  960/10051., 8/529., 120/10051., 768/2375., 16/285., 
  1152/8303., 48/2185., 2304/9025., 16/361., 17280/157757., 
  144/8303.], [1/81., 1/4., 64/441., 16/49., 49/289., 
  -1/18., 8/189., -4/21., -2/7., -7/34., 4/63., 7/153., 
  32/147., 8/51., 4/17., 1/54., -1/12., 2/21., 7/102., 
  1/36., -1/8., 1/7., 7/68.], [16/81., 9/121., 9/4., 
  2025/3136., 1., 4/33., -2/3., -9/22., 135/616., 
  -3/11., 5/14., -4/9., -135/112., 3/2., -45/56., 
  -20/189., -5/77., -75/392., 5/21., 25/441., 75/2156., 
  1125/10976., -25/196.], [625/104976., 1/81., 3025/104976.,
   484/2025., 25/484., 25/2916., 1375/104976., 55/2916., 
  -22/405., 5/198., -55/1458., 125/7128., -121/1458., 
  25/648., -1/9., -25/1458., -2/81., 44/405., -5/99., 
  4/81., 16/225., -352/1125., 8/55.], [81., 100/361., 
  225/529., 25/233289., 64/441., -90/19., 135/23., 
  -150/437., -50/9177., 80/399., 15/161., -24/7., 
  25/3703., -40/161., -40/10143., 9/7., -10/133., 5/3381.,
   -8/147., 1/49., -10/8379., 5/213003., 
  -8/9261.], [16/625., 9/4., 4/121., 900/14641., 81/196., 
  -6/25., 8/275., -3/11., -45/121., -27/28., 24/605., 
  18/175., 60/1331., 9/77., 135/847., 48/1375., -18/55., 
  72/1331., 54/385., 144/3025., -54/121., 1080/14641., 
  162/847.], [10000/6561., 25/144., 25/324., 1/81., 
  1/400., -125/243., 250/729., -25/216., 5/108., -1/48., 
  -100/729., 5/81., -5/162., 1/72., -1/180., -4000/6561., 
  50/243., 40/729., -2/81., 1600/6561., -20/243., 
  -16/729., 4/405.]])

    Cinv = np.linalg.inv(C)

    ctdim6=ct-1.
    cHHHdim6=cHHH-1.

    if EFTcount==3:
        coeffs = np.array([ct**4,ctt**2,ct**2*cHHH**2,cg**2*cHHH**2,cgg**2,ctt*ct**2,ct**3*cHHH,
                       ctt*ct*cHHH,ctt*cg*cHHH,ctt*cgg,ct**2*cg*cHHH,ct**2*cgg,
                       ct*cHHH**2*cg,ct*cHHH*cgg,cg*cHHH*cgg,
                       ct**3*cg,ct*ctt*cg,ct*cg**2*cHHH,ct*cg*cgg,
                       ct**2*cg**2,ctt*cg**2,cg**3*cHHH,cg**2*cgg])
    elif EFTcount==0:
        coeffs = np.array([
            1.+4.*ctdim6,
            0,
            1.+2.*(cHHHdim6+ctdim6),
            0,
            0,
            ctt,
            1.+cHHHdim6+3.*ctdim6,
            ctt,
            0,
            0,
            cg,
            cgg,
            cg,
            cgg,
            0,
            cg,
            0,
            0,
            0,
            0,
            0,
            0,
            0])
    elif EFTcount==1:
        coeffs = np.array([
            1.+4.*ctdim6+4.*ctdim6**2,
            ctt**2,
            1.+2.*(cHHHdim6+ctdim6)+(cHHHdim6+ctdim6)**2,
            cg**2,
            cgg**2,
            ctt*(1.+2.*ctdim6),
            1.+cHHHdim6+3.*ctdim6+2.*ctdim6*(cHHHdim6+ctdim6),
            ctt*(1.+ctdim6+cHHHdim6),
            cg*ctt,
            cgg*ctt,
            cg*(1.+2.*ctdim6), # a11 needs additional care for EFTcount 1
            cgg*(1.+2.*ctdim6),
            cg*(1.+cHHHdim6+ctdim6),
            cgg*(1.+cHHHdim6+ctdim6),
            cg*cgg,
            cg*(1.+2.*ctdim6),
            cg*ctt,
            0, # a18 needs additional care for EFTcount 1
            cg*cgg,
            0,
            0,
            0,
            0])
    elif EFTcount==2:
        coeffs = np.array([
            1.+4.*ctdim6+4.*ctdim6**2+2.*ctdim6**2,
            ctt**2,
            1.+2.*(cHHHdim6+ctdim6)+(cHHHdim6+ctdim6)**2+2.*cHHHdim6*ctdim6,
            cg**2,
            cgg**2,
            ctt*(1.+2.*ctdim6),
            1.+cHHHdim6+3.*ctdim6+2.*ctdim6*(cHHHdim6+ctdim6)+ctdim6*(cHHHdim6+ctdim6),
            ctt*(1.+ctdim6+cHHHdim6),
            cg*ctt,
            cgg*ctt,
            cg*(1.+2.*ctdim6+cHHHdim6),
            cgg*(1.+2.*ctdim6),
            cg*(1.+cHHHdim6+ctdim6+cHHHdim6),
            cgg*(1.+cHHHdim6+ctdim6),
            cg*cgg,
            cg*(1.+2.*ctdim6+ctdim6),
            cg*ctt,
            cg**2,
            cg*cgg,
            cg**2,
            0,
            0,
            0])
    else:
        print("ERROR: Wrong setting for EFTcount in Creategrid!!")
        return

    # Check that the grids have the same values for s, t
    for amp in amps[0:]:
      for amp2 in amps[0:]:

        if not (np.array_equal(amp[0], amp2[0]) and np.array_equal(amp[1], amp2[1])):
            print("The virtual grids do not contain the same phase-space points!")

    # load pdf to get values of alpha_s
    pdf=lhapdf.mkPDF(lhaid)

    for i, psp in enumerate(amps[0][0]):
        if usesmeft==1:
            mH = 125.
            beta = amps[0][0][i] # beta=sqrt(1. - 4. * mH**2 / s)  =>  s = 4. * mH**2 / (1 - beta**2)
            s = 4. * mH**2 / (1 - beta**2)
            muRs = s / 4. * renfac**2
            alphas = pdf.alphasQ2(muRs)

            coeffstmp = np.matmul(scipy.linalg.block_diag(1.,1.,1.,1./alphas**2,1./alphas**2,1.,1.,
                       1.,1./alphas,1./alphas,1./alphas,1./alphas,
                       1./alphas,1./alphas,1./alphas**2,
                       1./alphas,1./alphas,1./alphas**2,1./alphas**2,
                       1./alphas**2,1./alphas**2,1./alphas**3,1./alphas**3),coeffs)
        else:
            coeffstmp = coeffs

        A = np.matmul(Cinv, np.transpose(np.array([ [amps[j][2][i] for j in range(len(cHHH_grids))] ])))
        ME2 = np.matmul(coeffstmp, A)

        # Compute the uncertainties on the final PSP amplitude (uncorr. between PSPs, corr. between A coeffs)
        sigmaF = np.diag( np.array([ amps[j][3][i]**2 for j in range(len(cHHH_grids))] ))
        dA = np.matmul(Cinv, np.matmul(sigmaF, np.transpose(Cinv)))
        dME2 = np.matmul(coeffstmp, np.matmul(coeffstmp, dA))

        ME2s.append(ME2)
        dME2s.append(np.sqrt(dME2))

    np.savetxt(grid_temp, np.transpose([amps[0][0], amps[0][1], np.array(ME2s).flatten(), np.array(dME2s).flatten()]))
    print("Saved grid " + str(grid_temp))



class Bin:
    def __init__(self,n=0,y=0.,y2=0.,e2=0.):
        self.n = n
        self.y = y
        self.y2 = y2
        self.e2 = e2

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
    def __init__(self, method=None, dim=None, xmin=None, xmax=None, nbin=None, data=None, binned=False, grid_yaml=None):
        if grid_yaml:
            self.load(grid_yaml)
            return
        assert method is not None, "must specify method"
        assert dim is not None, "must specify dim"
        assert xmin is not None, "must specify xmin"
        assert xmax is not None, "must specify xmax"
        assert nbin is not None, "must specify nbin"
        assert data is not None, "must specify data"

        self.dim = dim
        self.xmin = np.array(xmin, dtype=float)
        self.xmax = np.array(xmax, dtype=float)
        self.nbin = np.array(nbin)
        self.binned = binned
        self.method = method
        self.lowclip = 1e-20 # error and 1/error clipped to [lowclip,highclip], this improves the numerical stability of the fit
        self.highclip = 1e20
        self.nneighbour = 9

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
                    nearest_points = all_points[:self.nneighbour]

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
                    bin_points = all_points[self.nneighbour:]
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

                    # Warn about negative first entries in covariance matrix (only possible due to numerical instability)
                    if pcov1[0,0] < 0. or pcov2[0,0] < 0.:
                        print('WARNING: Negative covariance matrix in bin ' + str(k))
                        print(pcov1)
                        print(pcov2)
                        print('nearest_points',nearest_points)
                        print('bin_points',bin_points)

                    self.data[k].add(popt[0], max(sqrt(abs(pcov1[0, 0])), sqrt(abs(pcov2[0, 0]))))  # corresponds to multiplying error estimate of chi-sq fit with max(1,chisq)
                    '''
                       model y = f(x) = c
                       y1   e1     y2   e2     popt   pcov1   pcov2    chisq
                       1    1e-5   2    1e-5   1.5    5e-11   0.25     5e9      # pcov2 = pcov1*chisq
                       0.9  1      1.1  1      1      0.5     0.0025   0.02
                       ==> use cov2 if chisq>>1, cov1 if chisq<1
                    '''

    def dump(self):
        # Dump the current grid to a dictionary
        def yaml_formatter(value):
            if isinstance(value,int):
                return str(value)
            elif isinstance(value, float):
                return "{:.20e}".format(value)
            else:
                raise Exception('yaml_formatter has no rule for value of type ' + type(value))
        # Create dict for yaml
        grid_yaml = {}
        grid_yaml['method'] = str(self.method)
        grid_yaml['dim'] = str(self.dim)
        grid_yaml['xmin'] = [ "{:.20e}".format(x) for x in self.xmin.tolist() ]
        grid_yaml['xmax'] = [ "{:.20e}".format(x) for x in self.xmax.tolist() ]
        grid_yaml['nbin'] = [str(x) for x in self.nbin.tolist()]
        grid_yaml['binned'] = int(self.binned)
        grid_yaml['lowclip'] = "{:.20e}".format(self.lowclip)
        grid_yaml['highclip'] = "{:.20e}".format(self.highclip)
        grid_yaml['nneighbour'] = str(self.nneighbour)
        grid_yaml['dx'] = [ "{:.20e}".format(x) for x in self.dx.tolist() ]
        grid_yaml['bins'] = {}
        # grid_yaml['deg'] = self.deg # Not saved
        # grid_yaml['pol'] = self.pol # Not saved
        for key, value in self.data.items():
            grid_yaml['bins'][str(key)] = {k: yaml_formatter(v) for k,v in vars(value).items()}
        return grid_yaml

    def load(self,grid_yaml):
        # Load the grid from a dictionary
        self.method = int(grid_yaml['method'])
        self.dim = int(grid_yaml['dim'])
        self.xmin = np.array( [float(x) for x in grid_yaml['xmin']] )
        self.xmax = np.array( [float(x) for x in grid_yaml['xmax']] )
        self.nbin = np.array( [int(x) for x in grid_yaml['nbin']] )
        self.binned = bool(grid_yaml['binned'])
        self.lowclip = np.array(float(grid_yaml['lowclip']))
        self.highclip = np.array(float(grid_yaml['highclip']))
        self.nneighbour = int(grid_yaml['nneighbour'])
        self.dx = np.array( [float(x) for x in grid_yaml['dx']] )
        self.data = {}
        self.deg = {}  # degree of interpolation poynomial (Note: not saved)
        self.pol = {}  # coefficients of interpolation polynomial (Note: not saved)
        key_pattern = re.compile( r'\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)' )
        for key,value in grid_yaml['bins'].items():
            key_match = key_pattern.search(key.strip())
            k = ( int(key_match.group(1)),int(key_match.group(2)) )
            self.data[k] = Bin( int(value['n']), float(value['y']), float(value['y2']), float(value['e2']) )

    def polyfit2d(self, x, y, z, orders):
        ncols = np.prod(orders + 1)
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(orders[0] + 1), range(orders[1] + 1))
        for k, (i, j) in enumerate(ij):
            G[:, k] = x ** i * y ** j
        m, _, _, _ = np.linalg.lstsq(G, z, rcond=None)
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
            return [[k, d.gety(sample)] for k, d in self.data.items()]
        if flag == 'x':
            if (returnError):
                res = [[self.x(k), d.gety(sample), d.gete()] for k, d in self.data.items()]
            else:
                res = [[self.x(k), d.gety(sample)] for k, d in self.data.items()]
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
                for corner in np.ndindex(tuple([2, ] * self.dim)):
                    k = corner * (self.nbin - 1)
                    x = np.array(corner, float)
                    res += [[x.copy(), self.data[tuple(k)].gety(sample)], ]
            return res
        if flag == 'plain':
            return [self.x(k).tolist() + [d.gety(sample), ] for k, d in self.data.items()]

    def printgrid(self):
        for k, d in self.data.iteritems():
            print('k: ' + str(k) + ' d.n: ' + str(d.n) + ' d.y: ' + str(d.y) + ' d.y2: ' + str(d.y2) + ' d.e2: ' + str(d.e2))

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
            temp = list(zip(*self.gridpoints(sample=(nsamples != 1), flag='x', extendToBorder=True)))
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
                x, y, z = np.asarray(list(zip(*dat)))
                self.pol[k, i] = self.polyfit2d(x, y, z, orders=degree)

    def interpolate(self, x, y, selectsample=-1):
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
        print(f'py: Initialising {selected_grid}')
        self.selected_grid = selected_grid
        self.selected_grid_dirname = os.path.dirname(self.selected_grid)
        self.mHs = 125.**2
        self.method = 1  # 1: CloughTocher;  2: Polynomial
        self.polydegree = 2
        self.tolerance = 1.e-8 # costh is nudged to 1 if within this tolerance
        self.cdfdata = None
        self.cdf = None
        self.interpolator = None

        if os.path.splitext(self.selected_grid)[1].strip() == ".yaml":
            # If input file is a .yaml file then grid can be loaded directly from the yaml

            # Load yaml file
            with open(self.selected_grid, 'r') as infile:
                grid_yaml = yaml.safe_load(infile)

            # Parse cdfdata
            self.cdfdata = np.array(grid_yaml['cdfdata'], dtype=float)
            self.cdf = interpolate.splrep(*self.cdfdata.T, k=1)

            # Initialise grid
            self.interpolator = Grid(grid_yaml=grid_yaml)
            self.interpolator.initInterpolation(1)

        else:
            # Generate grid via fitting/binning

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

    def dump(self):
        # Produce a .yaml snapshot of the current CreateGrid instance
        grid_yaml = self.interpolator.dump()
        # Append cdf data to grid_yaml
        grid_yaml['cdfdata'] = [ ["{:.20e}".format(ix) for ix in x] for x in self.cdfdata.tolist()]
        # Output yaml file representing grid
        filename_yaml = os.path.splitext(self.selected_grid)[0].strip() + '.yaml'
        with open(filename_yaml, 'w') as outfile:
            yaml.dump(grid_yaml, outfile, default_flow_style=False)

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

    def TestClosure(self,closure_grid):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import pylab as pl
        from mpl_toolkits.mplot3d import Axes3D
        #pl.rc('text',usetex=true) # Use LaTex
        #pl.rcParams['text.latex.preamble']=r'\usepackage{amstext}\newcommand{\dd}{\mathrm{d}},\mathchardef\mhyphen="2D'
        
        closure_grid_dirname, closure_grid_name = os.path.split(closure_grid)
        
        b, cth, y, e = np.loadtxt(closure_grid, unpack=True)

        percent_deviations = []
        std_deviations = []
        for b_value, cth_value, y_value, e_value in zip(b,cth,y,e):
            xUniform = interpolate.splev(b_value, self.cdf)
            percent_deviations.append( 100.*(y_value-self.interpolator.interpolate(xUniform, cth_value)[0])/y_value )
            std_deviations.append( (y_value-self.interpolator.interpolate(xUniform, cth_value)[0])/e_value )
        percent_deviations = np.array(percent_deviations)
        std_deviations = np.array(std_deviations)
        
        ##
        ## Plot of grid points vs input points
        ##
        #x0Uniform = interpolate.splev(b, self.cdf)
        #y_grid = []
        #for x,c in zip(x0Uniform,cth):
        #    y_grid.append(self.interpolator.interpolate(x, c)[0])
        #fig = plt.figure(dpi=100)
        #ax = fig.add_subplot(111, projection='3d')
        #fig.suptitle('Closure test')
        #ax.plot(x0Uniform, cth, y, linestyle="None", marker="o")
        #ax.plot(x0Uniform, cth, y_grid, linestyle="None", marker="x")
        ## Error bars
        ## for i in np.arange(0, len(x0Uniform)):
        ##     ax.plot([x0Uniform[i], x0Uniform[i]], [cth[i], cth[i]],
        ##             [y[i] + e[i], y[i] - e[i]], marker="_")
        #ax.set_xlabel('beta')
        #ax.set_ylabel('cos(theta)')
        #ax.set_xlim3d(0., 1.)
        #ax.set_ylim3d(0., 1.)
        #pl.savefig(os.path.join(closure_grid_dirnamem'closure'+str(closure_grid_name)+'.pdf'),bbox_inches='tight')

        #
        # Percent deviation of grid from central value of point
        #
        #print(percent_deviations)
        bins = np.arange(-20.25,20.25,0.5)
        _, _, _ = plt.hist(np.clip(percent_deviations, bins[0], bins[-1]), bins=bins, histtype='step', edgecolor='red', linewidth=1)
        plt.xlabel(r'$\mathrm{difference\ [\%]}$')
        plt.ylabel(r'$N_\mathrm{{points}}$')
        pl.savefig(os.path.join(closure_grid_dirname,'percent_difference_'+str(closure_grid_name)+'.pdf'),bbox_inches='tight')

        ##
        ## Number of standard deviations between grid and point
        ##
        ## print(std_deviations)
        ## bins = np.arange(-11,11,2)
        ## _, _, _ = plt.hist(np.clip(std_deviations, bins[0], bins[-1]), bins=bins, facecolor='r', alpha=0.75, edgecolor='black', linewidth=1.2)
        ## plt.xlabel('Standard Deviation')
        ## plt.ylabel('Npoints')
        ## plt.show()
