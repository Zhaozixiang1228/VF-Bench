#!/usr/bin/env bash
set -e

data_dir=data2/MVF
mkdir -p $data_dir
cd $data_dir

if test -f "Harvard.zip" ; then
    echo "Zip file exists: Harvard.zip"
    exit 1
fi

wget -nv --show-progress https://share.phys.ethz.ch/~pf/zixiangdata/vfbench/Harvard.zip
