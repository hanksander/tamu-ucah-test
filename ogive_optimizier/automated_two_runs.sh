#!/bin/bash

python3 full_optimizer_ogive.py

mkdir -p results/ogive_optimization_1
mv best_solution.log results/ogive_optimization_1/
mv control_points.log results/ogive_optimization_1/
mv temp_ogive_cases results/ogive_optimization_1/

python3 full_optimizer_ogive_2.py
mkdir -p results/ogive_optimization_2
mv best_solution.log results/ogive_optimization_2/
mv control_points.log results/ogive_optimization_2/
mv temp_ogive_cases results/ogive_optimization_2/