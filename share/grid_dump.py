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

if os.path.splitext(args.grid)[1].strip() != '.yaml':
    print '== Dumping Grid =='
    grid.dump()
else:
    print 'Can not dump grid, file extension must not be .yaml'

print '== Exiting =='
