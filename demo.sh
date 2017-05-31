#!/bin/bash

seed=0
grid="Virt_EFT.grid"

readyfile=pyReadySignal-$(printf %04d $seed)

echo "Using seed: $seed, readyfile: $readyfile"

# Remove pre-existing readyfile (could exist if a previous run failed)
rm -f $readyfile

# Launch grid and wait for readyfile to be created
python grid.py --seed="$seed" --grid="$grid" &
while [ ! -f $readyfile ]
do
sleep 10
done

# Launch job
./cpp_testgrid.x
