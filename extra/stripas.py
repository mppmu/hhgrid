from argparse import ArgumentParser

import numpy as np
import lhapdf
import csv

parser = ArgumentParser()

parser.add_argument("-i", "--infile", dest="infile",
                    action="store", type=str, required=True,
                    help='input grid file', metavar="INTFILE")

parser.add_argument("-o", "--outfile", dest="outfile",
                    action="store", type=str, required=True,
                    help='output grid file', metavar="OUTFILE")

parser.add_argument("-n","--noerror", dest="noerror",
                    action="store_true",
                    help='do not rescale error by alphas')

args = parser.parse_args()

mHs = 125.**2
pdf=lhapdf.mkPDF('PDF4LHC15_nlo_100',0)

def s(beta):
    return 4.*mHs/(1-beta**2)

input_grid = np.loadtxt(args.infile).tolist()
output_grid = []

for beta, cost, vfin, vfinerr in input_grid:
    mursq=s(beta)/4
    alphas = pdf.alphasQ2(mursq)
    if args.noerror:
        output_grid.append([beta, cost, vfin/(alphas**2), vfinerr])
    else:
        output_grid.append([beta, cost, vfin / (alphas ** 2), vfinerr/(alphas**2)])

with open(args.outfile, 'w') as output_file:
    wr = csv.writer(output_file, delimiter=" ")
    wr.writerows(output_grid)
