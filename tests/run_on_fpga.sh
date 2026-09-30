#!/bin/bash

if [ -z "$1" ] ; then
    echo "Usage: $0 test_bench_name"
    echo ""
    echo "[Available test benches]"
    find -name 'tb_*.py' | sed -E 's|.*/tb_(.*)\.py$|\1|'
    echo ""
    exit 0
fi

make -f makefile.fpga

target=$1

rm -f output.log

python fpga_tb_$target.py

cat output.log

