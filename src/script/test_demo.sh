#!/usr/bin/env bash
set -e
set -x

python test_demo.py --task_name MEF --dataset_name YouTube-demo --exp_path UniVF-MEF --fps 30
python test_demo.py --task_name MFF --dataset_name DAVIS-demo --exp_path UniVF-MFF --fps 24
python test_demo.py --task_name IVF --dataset_name VTMOT-demo --exp_path UniVF-IVF --fps 24
python test_demo.py --task_name MVF --dataset_name Harvard-demo --exp_path UniVF-MVF --fps 10
