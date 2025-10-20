#!/usr/bin/env bash
set -e

data_dir=data2/IVF
mkdir -p $data_dir
cd $data_dir

if test -f "VTMOT.zip" ; then
    echo "Zip file exists: VTMOT.zip"
    exit 1
fi

wget -nv --show-progress https://share.phys.ethz.ch/~pf/zixiangdata/vfbench/VTMOT.zip
