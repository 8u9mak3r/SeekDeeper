#!/bin/bash
BUDDY_BUILD_DIR := /home/dahaoming/buddy-mlir/build/
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt
SCALEHLS_BUILD_DIR := /home/dahaoming/scalehls-mlir/build/
SCALEHLS_OPT := ${BUDDY_BUILD_DIR}/bin/scalehls-opt

linalg-i8:
	@${BUDDY_OPT} ./test.mlir \
		-pass-pipeline \
			"builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" -o ./bertlayer-i8.mlir

linalg-i4:
	@${BUDDY_OPT} ./subgraph0.mlir \
		-pass-pipeline \
			"builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" -o ./bertlayer-i4.mlir

clear:
	rm *.cpp hls.mlir subgraph0.mlir