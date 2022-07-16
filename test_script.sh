#! /bin/bash

for N in 6 8 10 12
do
for ut in 0. .5 1. 2.
do
for eps in 1e-2 1e-3 1e-4 1e-5
do python3 evolution.py --N $N --U $ut --eps $eps
done
done
done  

for N in 12
do
for ut in 0. .5 1. 2.
do
for maxdim in 600 800 1000 1200 1400 1600 1800 2000
do python3 evolution.py --N $N --U $ut --dim $maxdim
done
done
done
