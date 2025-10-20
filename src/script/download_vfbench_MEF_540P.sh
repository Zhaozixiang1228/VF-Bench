#!/usr/bin/env bash
set -e

data_dir=data2/MEF
mkdir -p $data_dir
cd $data_dir

if test -f "YouTube-HDR-540P.zip" ; then
    echo "Zip file exists: YouTube-HDR-540P.zip"
    exit 1
fi

wget -nv --show-progress https://share.phys.ethz.ch/~pf/zixiangdata/vfbench/YouTube-HDR-540P.zip
