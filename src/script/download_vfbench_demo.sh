#!/usr/bin/env bash
set -e

if test -f "vfbench-demo.zip" ; then
    echo "Zip file exists: vfbench-demo.zip"
    exit 1
fi

wget -nv --show-progress https://share.phys.ethz.ch/~pf/zixiangdata/vfbench/vfbench-demo.zip

unzip vfbench-demo.zip
rm vfbench-demo.zip