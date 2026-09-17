P_PEAK    = 512.0                 # ops/cycle (16x16 systolic array)
BW_CONFIG = 16.0 / (3.0 * 3.0)    # theoretical config bandwidth, bytes/cycle
BW_MEM    = 16.0                  # memory bandwidth, bytes/cycle
FREQ      = 1.5e9                 # clock frequency, Hz (assumes 1.5 GHz)

# ---------- BraggNN base data: (name, shape, ops, mem_bytes, config_bytes) ----------
braggnn_kernels = [
    ('conv1',    '11x11x1, 3x3x1, 64',   93312,      6137,      112),
    ('theta',    '9x9x64, 1x1x64, 32',   331776,     9952,      112),
    ('phi',      '9x9x64, 1x1x64, 32',   331776,     9952,      112),
    ('g',        '9x9x64, 1x1x64, 32',   331776,     9952,      112),
    ('matmul-θφ','81x32, 32x81',         331776,     11745,     192),  # TODO: check source code
    ('matmul-g', '81x81, 81x32',         419904,     11745,     96),
    ('conv-nlb', '9x9x32, 1x1x32, 64',   331776,     26464,     112),
    ('resadd',   '9x9x64, 9x9x64',       10368,      15552,     160),
    ('conv2',    '9x9x64, 3x3x64, 32',   1806336,    25312,     112),
    ('conv3',    '7x7x32, 3x3x32, 8',    115200,     4104,      112),
    ('fc1',      '200, 200x16, 16',      6400,       3480,      96),
    ('fc2',      '16, 16x8, 8',          256,        184,       96),
    ('fc3',      '8, 8x4, 4',            64,         60,        96),
    ('fc4',      '4, 4x2, 2',            16,         22,        96),
    ('output',   '2, 2x2, 2',            8,          16,        96),
]

# ---------- PtychoNN base data: derived from conv layer shapes ----------
ptychonn_layers = [
    # name,          in_ch, out_ch, H,  W
    ('enc.conv1',    1,     32,     64, 64),
    ('enc.conv2',    32,    32,     64, 64),
    ('enc.conv3',    32,    64,     32, 32),
    ('enc.conv4',    64,    64,     32, 32),
    ('enc.conv5',    64,    128,    16, 16),
    ('enc.conv6',    128,   128,    16, 16),
    #('dec1.conv7',   128,   128,    8,  8),
    #('dec1.conv8',   128,   128,    8,  8),
    #('dec1.conv9',   128,   64,     16, 16),
    #('dec1.conv10',  64,    64,     16, 16),
    #('dec1.conv11',  64,    64,     32, 32),
    #('dec1.conv12',  64,    64,     32, 32),
    #('dec1.conv13',  64,    1,      64, 64),
]
k = 3
CFG_BYTES = 77.0


def build_braggnn_table1():
    table1 = []
    for name, shape, ops, mem_b, cfg_b in braggnn_kernels:
        i_op = ops / mem_b
        i_oc = ops / cfg_b
        table1.append((name, shape, ops, mem_b, cfg_b, i_op, i_oc))
    return table1


def build_ptychonn_table1():
    table1 = []
    for name, cin, cout, h, w in ptychonn_layers:
        ops = 2 * k * k * cin * cout * h * w
        mem_b = (cin*h*w) + (k*k*cin*cout) + (4*cout) + (cout*h*w)
        shape = f'{h}x{w}x{cin}, {k}x{k}x{cin}, {cout}'
        i_op = ops / mem_b
        i_oc = ops / CFG_BYTES
        table1.append((name, shape, ops, mem_b, CFG_BYTES, i_op, i_oc))
    return table1


# ---------- wo/memory access (config-only, Eq. 3 w/ bare P_PEAK) ----------
def print_wo_memory(table1, title):
    print(f'### {title}: wo/memory access')
    print('| Kernel (input, weight, feature) | Ops | $I_{OC}$ | '
          '$BW_{Config}\\times I_{OC}$ | Bound | $P_{Attainable,Seq.}$ | '
          'Cycles | Time @ 1.5GHz (μs) |')
    print('|---|---|---|---|---|---|---|---|')
    tot_cycles = 0.0
    for name, shape, ops, mem_b, cfg_b, i_op, i_oc in table1:
        bwcfg_ioc = BW_CONFIG * i_oc
        bound = 'Config' if bwcfg_ioc < P_PEAK else 'Compute'
        p_att = 1.0 / (1.0/P_PEAK + 1.0/bwcfg_ioc)
        cycles = ops / p_att
        time_us = cycles / FREQ * 1e6
        tot_cycles += cycles
        print(f'| {name} ({shape}) | {ops:,} | {i_oc:.6f} | {bwcfg_ioc:.6f} | '
              f'{bound} | {p_att:.6f} | {cycles:.4f} | {time_us:.6f} |')
    tot_time_us = tot_cycles / FREQ * 1e6
    print(f'| **Total** | | | | | | **{tot_cycles:.4f}** | '
          f'**{tot_time_us:.6f} (μs)** |')
    print()


# ---------- w/memory access (Eq. 1 combined with Eq. 3) ----------
def print_w_memory(table1, title):
    print(f'### {title}: w/memory access')
    print('| Kernel (input, weight, feature) | Ops | $I_{Operational}$ | '
          '$BW_{Mem}\\times I_{Op}$ | $I_{OC}$ | $BW_{Config}\\times I_{OC}$ | '
          'Bound | $P_{Attainable,Seq.}$ | Cycles | Time @ 1.5GHz (μs) |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    tot_cycles = 0.0
    for name, shape, ops, mem_b, cfg_b, i_op, i_oc in table1:
        bwmem_iop = BW_MEM * i_op
        bwcfg_ioc = BW_CONFIG * i_oc
        exec_rate = min(P_PEAK, bwmem_iop)
        compute_or_memory_bound = 'Memory' if bwmem_iop < P_PEAK else 'Compute'
        bound = 'Configuration' if bwcfg_ioc < exec_rate else compute_or_memory_bound
        p_att = 1.0 / (1.0/exec_rate + 1.0/bwcfg_ioc)
        cycles = ops / p_att
        time_us = cycles / FREQ * 1e6
        tot_cycles += cycles
        print(f'| {name} ({shape}) | {ops:,} | {i_op:.6f} | {bwmem_iop:.6f} | '
              f'{i_oc:.6f} | {bwcfg_ioc:.6f} | {bound} | {p_att:.6f} | '
              f'{cycles:.4f} | {time_us:.6f} |')
    tot_time_us = tot_cycles / FREQ * 1e6
    print(f'| **Total** | | | | | | | | **{tot_cycles:.4f}** | '
          f'**{tot_time_us:.6f} (μs)** |')
    print()


braggnn_t1 = build_braggnn_table1()
ptychonn_t1 = build_ptychonn_table1()

print_wo_memory(braggnn_t1, 'BraggNN')
print_w_memory(braggnn_t1, 'BraggNN')
print_wo_memory(ptychonn_t1, 'PtychoNN')
print_w_memory(ptychonn_t1, 'PtychoNN')
