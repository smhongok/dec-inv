#!/bin/bash

# the number of instances, default=100
num=10

# encoder
python main_watermark_detection.py --run_name encoder --mode encoder --with_tracking --end $num

# grad-free (ours)
python main_watermark_detection.py --run_name no_grad --mode no_grad --with_tracking --end $num

# grad-based 
python main_watermark_detection.py --run_name grad --mode grad --with_tracking --end $num