#!/bin/bash

echo "Running all experiments..."

mkdir -p Figures
find . -type f -name "*.sh" -exec chmod +x {} +

echo "----------------------------------"
echo "Running 6.1 – Baselines"
cd 6.1 || exit
./run_experiment.sh
cp Figures/* ../Figures/6_1.pdf 2>/dev/null
cd ..
echo "----------------------------------"
echo "Running 6.2 – Set Points"
cd 6.2 || exit
./run.sh
cp Figures/* ../Figures/6_2.pdf 2>/dev/null
cd ..

echo "----------------------------------"
echo "Running 6.3"
cd 6.3 || exit
./experiments.sh
cp Figures/6_3a.pdf ../Figures/6_3a.pdf
cp Figures/6_3b.pdf ../Figures/6_3b.pdf
cd ..

echo "----------------------------------"
echo "Running 6.4"
cd 6.4 || exit
./run.sh
cp Figures/* ../Figures/6_4.pdf 2>/dev/null
cd ..
echo "----------------------------------"
echo "All experiments completed."
