from argparse import ArgumentParser
import os
import random

parser = ArgumentParser()

parser.add_argument("-g", "--grid", dest="grid",
                    action="store", type=str, required=True,
                    help='the grid to be used', default=False,
                    metavar="GRID")

args = parser.parse_args()

random.seed(123)

# Produce Grids for Closure test
print('== Creating Closure Test Grids ==')

grid_filename, grid_extension = os.path.splitext(os.path.basename(args.grid))

if grid_extension == '.grid':

    lines = []
    with open(args.grid,'r') as f:
        lines = f.readlines()
        lines[-1] = lines[-1] + '\n' # Add missing newline to last entry

    random.shuffle(lines)
    with open(grid_filename+'_training_80'+grid_extension,'w') as f:
        f.write(''.join(lines[:int(0.8*len(lines))]))
    with open(grid_filename + '_test_20' + grid_extension, 'w') as f:
        f.write(''.join(lines[int(0.8*len(lines)):]))

    random.shuffle(lines)
    with open(grid_filename+'_training_50'+grid_extension,'w') as f:
        f.write(''.join(lines[:int(0.5*len(lines))]))
    with open(grid_filename + '_test_50' + grid_extension, 'w') as f:
        f.write(''.join(lines[int(0.5*len(lines)):]))
