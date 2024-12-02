#!/bin/bash

# the number of instances, default=100
num=1

# # grad-free, 16 bit (ours)
# for iter in 20 50 100 200
# do
#   python main_fsm.py --run_name nograd_half_$iter --mode no_grad --decoder_inv_numstep $iter --with_tracking --precision half --adam --end $num
# done

# grad-free, 32 bit (ours)
for iter in 20 50 100 200
do
  python main_fsm.py --run_name nograd_full_$iter --mode no_grad --decoder_inv_numstep $iter --with_tracking --precision full --adam --end $num
done

# grad-based, 32 bit
for iter in 20 30 50 100
do
  python main_fsm.py --run_name grad_full_fixed_$iter --mode grad --decoder_inv_numstep $iter --with_tracking --precision full --adam --end $num
  python main_fsm.py --run_name grad_full_scheduled_$iter --mode grad --decoder_inv_numstep $iter --with_tracking --precision full --adam --lr_scheduling --end $num
done
