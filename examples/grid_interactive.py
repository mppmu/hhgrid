from argparse import ArgumentParser
import os
import creategrid

parser = ArgumentParser()

parser.add_argument("-g", "--grid", dest="grid",
                    action="store", type=str, required=True,
                    help='the grid to be used', default=False,
                    metavar="GRID")

args = parser.parse_args()

# Initialise Grid
print '== Initialising Grid =='
grid = creategrid.CreateGrid(args.grid)

# Python function that we want to call via pyFuncCallPipe
def func(arg):
    s,t=map(lambda e: float(e),arg.split(','))
    return grid.GetAmplitude(s,t)

print '== Ready to evaluate function =='

# Read from user input
# print result to screen
# If user types 'exit'then exit script
while True:
    line = raw_input("s, t:").strip()

    if line == 'exit':
        break

    res = "{:.40e}".format(func(line))

    print res

print '== Exiting =='
