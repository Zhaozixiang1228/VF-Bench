#!/usr/bin/env bash
set -e

data_dir=data2/MFF
mkdir -p $data_dir
cd $data_dir

if test -f "DAVIS-480P.zip" ; then
    echo "Zip file exists: DAVIS-480P.zip"
    exit 1
fi

wget -nv --show-progress https://share.phys.ethz.ch/~pf/zixiangdata/vfbench/DAVIS-480P.zip
