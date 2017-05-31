from argparse import ArgumentParser
import os
import creategrid

parser = ArgumentParser()

parser.add_argument("-s", "--seed", dest="seed",
                    action="store", type=str, required=True,
                    help='seed for FIFOs', default=False,
                    metavar="SEED")

parser.add_argument("-g", "--grid", dest="grid",
                    action="store", type=str, required=True,
                    help='the grid to be used', default=False,
                    metavar="GRID")

parser.add_argument("-v", "--verbose", dest="verbose",
                    action="store_true", required=False,
                    help='print i/o messages', default=False)

args = parser.parse_args()

# FIFO pipes created with 'mkfifo' command
pyFuncCallPipe   = "pyInputPipe-"  + str(args.seed).zfill(4)
pyFuncReturnPipe = "pyOutputPipe-" + str(args.seed).zfill(4)

# File to create to signal bash to continue processing
pyReadySignalFile = "pyReadySignal-" + str(args.seed).zfill(4)

# Delete FIFO pipes if they exist and recreate them
try:
    os.remove(pyFuncCallPipe)
except OSError:
    pass
try:
    os.remove(pyFuncReturnPipe)
except OSError:
    pass
os.mkfifo(pyFuncCallPipe)
os.mkfifo(pyFuncReturnPipe)

# Initialise Grid
if args.verbose:
    print '== Initialising Grid =='
grid = creategrid.CreateGrid(args.grid)

# Python function that we want to call via pyFuncCallPipe
def func(arg):
    s,t=map(lambda e: float(e),arg.split(','))
    return grid.GetAmplitude(s,t)

if args.verbose:
    print '== Ready to evaluate function =='
    print 'Input Pipe: ', pyFuncCallPipe
    print 'Output Pipe: ', pyFuncReturnPipe
    print 'Ready Signal: ', pyReadySignalFile

# Signal that we are ready
open(pyReadySignalFile, 'a').close()

# Read from pyFuncCallPipe
# Write result to pyFuncReturnPipe
# If sent:
#   - 'exit' then exit script
while True:
    with open(pyFuncCallPipe, 'r') as callPipe:
        line = callPipe.readline().strip()

    if args.verbose:
        print '(' + str(args.seed) + ') Received: ' + line
    if line == 'exit':
        break

    res = "{:.40e}".format(func(line))

    if args.verbose:
        print '(' + str(args.seed) + ') Sending: ' + res
    with open(pyFuncReturnPipe, 'w') as returnPipe:
        returnPipe.write(res)

os.remove(pyReadySignalFile)
os.remove(pyFuncCallPipe)
os.remove(pyFuncReturnPipe)

if args.verbose:
    print '== Exiting =='
