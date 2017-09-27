#!/bin/bash
mkdir -p sounds
mkdir -p paper/plots
echo '0 & 0 & 0.20 & 2.45 $\times 10^{-6}$ & 500 & 600 \\
1 & 0 & 0.39 & 4.91 $\times 10^{-6}$ & 1000 & 1200 \\
2 & 0 & 0.59 & 7.36 $\times 10^{-6}$ & 1500 & 1800 \\' > paper/plots/mq_lp_compare_chirp_params.txt
sox -n -r 16000 -c 1 -t f64 sounds/chirps_simple.f64 synth 1 \
sine 500-600   \
sine 1000-1200 \
sine 1500-1800 \
#sine 2000-2400 \
#sine 2500-3000 \
#sine 3000-3600 \
