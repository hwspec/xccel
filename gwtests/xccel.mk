#
# Repository-specific settings for gwtests/Makefile (xccel)
#

HWMODULENAME=AxiXccel
BRIDGEBN=xccel_bridge
TOPNAME=wrapper
SRCDIR = $(abspath ../generated/$(HWMODULENAME))

TARGETS = braggnn_batch braggnn_conv braggnn_fc  braggnn_inference_opt  braggnn_inference  braggnn_nlb1x1  braggnn_nlb_core  matmul

.DEFAULT_GOAL := braggnn_inference

SCALA_SRCS := $(shell find ../src -name '*.scala') ../build.sbt
GENCMD = (cd ../ ; sbt "runMain xccel.AxiXccel")
