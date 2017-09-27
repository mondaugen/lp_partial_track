#!/bin/bash

# clone these repositories in parent directory
if [ ! -d ../signal_modeling ]
then
    (cd ..; git clone https://github.com/mondaugen/signal_modeling)
fi
if [ ! -d ../cont_win_comp ]
then
    (cd ..; git clone https://github.com/mondaugen/cont_win_comp)
fi

# setup python path
source setup_pythonpath.sh

# build plot by first synthesizing test file
test/make_chirps_simple.sh
# then track partials and plot results
test/bplp_dp_mq_simple.py

# now you can build the paper
cd paper
./build_paper.sh
