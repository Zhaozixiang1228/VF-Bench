#!/usr/bin/env bash
set -e

if test -f "UniVF-checkpoints.zip" ; then
    echo "Zip file exists: UniVF-checkpoints.zip"
    exit 1
fi

wget -nv --show-progress https://share.phys.ethz.ch/~pf/zixiangdata/vfbench/UniVF-checkpoints.zip

unzip UniVF-checkpoints.zip
rm UniVF-checkpoints.zip