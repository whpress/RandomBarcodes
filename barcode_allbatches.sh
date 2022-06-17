#!/bin/bash
python barcode_batch.py 0 1/4 randomcode_1e6_34.pkl randomcode_1e6_34_sim_1e6reads batch_sim_output &
pid0=$!
python barcode_batch.py 1 2/4 randomcode_1e6_34.pkl randomcode_1e6_34_sim_1e6reads batch_sim_output &
pid1=$!
python barcode_batch.py 0 3/4 randomcode_1e6_34.pkl randomcode_1e6_34_sim_1e6reads batch_sim_output &
pid2=$!
python barcode_batch.py 1 4/4 randomcode_1e6_34.pkl randomcode_1e6_34_sim_1e6reads batch_sim_output &
pid3=$!
wait $pid0
wait $pid1
wait $pid2
wait $pid3
