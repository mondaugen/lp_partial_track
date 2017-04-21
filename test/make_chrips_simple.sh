#!/bin/bash
sox -n -r 16000 -c 1 -t f64 sounds/chirps_simple.f64 synth 1 \
sine 500-600   \
sine 1000-1200 \
sine 1500-1800 \
#sine 2000-2400 \
#sine 2500-3000 \
#sine 3000-3600 \
