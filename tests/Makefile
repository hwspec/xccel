#
# This is Makefile for cocotb
#

SIM ?= verilator
TOPLEVEL_LANG ?= verilog
TOPNAME=user_accel_bd_wrapper
HWMODULENAME=AxiXccel
SRCDIR = $(abspath ../generated/$(HWMODULENAME))
VERILOG_SOURCES += $(SRCDIR)/*.v $(SRCDIR)/*.sv

# WAVES=1 (default): cocotb's Verilator makefile builds with --trace
# --trace-structs and runs with --trace (dump.vcd); WAVES=0 builds and runs
# without tracing
WAVES ?= 1
# compile the Verilated model in parallel
BUILD_ARGS ?= -j$(shell nproc)

TARGETS=matmul braggnn_conv braggnn_fc braggnn_nlb1x1 braggnn_nlb_core braggnn_inference braggnn_inference_opt braggnn_batch

all:
	make matmul

$(TARGETS) : $(SRCDIR)/$(HWMODULENAME).sv
	@echo $(VERILOG_SOURCES)
	@PARAMFN=$(SRCDIR)/params.json $(MAKE) sim MODULE=tb_$@ TOPLEVEL=$(TOPNAME)
	@echo -e "\n[Output]"
	@cat output.log

# the list-scheduled BraggNN program runs on its own configuration:
#   make braggnn_inference_opt HWMODULENAME=AxiXccelOpt SIM_BUILD=sim_build_opt
# and on three arrays (SystolicArrayN):
#   make braggnn_inference_opt HWMODULENAME=AxiXccelOptN SIM_BUILD=sim_build_optn
# and on a 3 x 3 window of arrays with wide weight beats:
#   make braggnn_inference_opt HWMODULENAME=AxiXccelOpt9 SIM_BUILD=sim_build_opt9
$(SRCDIR)/$(HWMODULENAME).sv gen:
	(cd ../ ; ./mill runMain xccel.$(HWMODULENAME))

include $(shell cocotb-config --makefiles)/Makefile.sim

clean::
	rm -f output.txt results.xml *.vcd fpga_simple.py
	rm -rf __pycache__
