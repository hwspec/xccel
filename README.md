# XCCEL

XCCEL is a framework for exploring the SRAM design and compute unit in X-ray detector ASICs simulateniously.
This framework abstracts near-sensor AI accelerators to the input activation buffer, weight buffer, accumulator, compute unit, and instraction decoder, assuming deterministic dataflow such as Groq.
The user (whichever the human or AI agents) can specify the number of SRAMs and design of each buffer including depth, word length, the number of ports, port bandwidth, mask granularity, and read/write latency.
Moreover, the user can replace a compute unit with their IP such as systolic array, reduction tree, and vector engine.
It does not have any caches, reservation station, and arbiters inside the accelerator since it adopts deterministic dataflow architecture.

## Usage

```bash
make -C garageworks setup
source garageworks/.venv/bin/activate
./mill runMain xccel.AxiXccelOpt9x64T
cd tests
make braggnn_inference_opt HWMODULENAME=AxiXccelOpt9x64T SIM_BUILD=sim_build_opt9x64t
```

## Design Principle

At scheduler view, data movement is like 'move data at SRAM address i to compute-unit port j (other SRAM address j)'.
However, the **static** connectivities at RTL-level vary.

- 1:1
  - one bank, one port
- 1:N
  - one wide bank feeds several ports
- N:1
  - several narrow banks feed one wide port
- N:M
  - direct wire, but one bank/port have several wire

On top of this, SRAM port can take several patterns including 1RW, 1R1W, and 2RW.

Additionally, **dynamic** cast modes, including unicast, multicast, and broadcast, should be considered.
This cast mode is implemented by the shifter and mux.

- shiftAmt: implemented by a barrel shifter between components (e.g., interface of SRAM and systolic array mesh)
- groupShift: how many grains (e.g., SRAM bank ports) correspond to ports (e.g., mesh ports)


## Scheduler (tentative)





## Related Works

- A. Parashar et al., "Timeloop: A Systematic Approach to DNN Accelerator Evaluation," 2019 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), Madison, WI, USA, 2019, pp. 304-315, doi: 10.1109/ISPASS.2019.00042. keywords

