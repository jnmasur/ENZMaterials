#! /bin/bash

for nthreads in 8
do
  export OMP_NUM_THREADS=$nthreads

  python test.py --U .5 --N 10 --c 1.
done
