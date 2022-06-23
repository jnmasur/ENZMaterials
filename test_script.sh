#! /bin/zsh

for ut in 0. .5 1. 2.
do
for maxdim in 600 800 1200 1400 1600 1800 2000
do python3 evolution.py $ut $maxdim 0
done
done
