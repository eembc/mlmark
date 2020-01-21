# 
# Copyright (C) 2019 EEMBC(R). All Rights Reserved
# 
# All EEMBC Benchmark Software are products of EEMBC and are provided under the
# terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
# are proprietary intellectual properties of EEMBC and its Members and is
# protected under all applicable laws, including all applicable copyright laws.  
# 
# If you received this EEMBC Benchmark Software without having a currently
# effective EEMBC Benchmark License Agreement, you must discontinue use.
# 

VERSION = "1.0.3"

# Note: developer errors are CRITICAL, user config errors are ERRORS

# Note: Since there are no structurally-enforced datatypes in Python, I'm
# using constants to refer to data members to prevent soft errors for things
# written by developers and users, but not EEMBC (on the todo list)

# Scoring modes
TOPN = 'topn'
MAP = 'map'

# Run modes
LATENCY = 'latency'
THROUGHPUT = 'throughput'
ACCURACY = 'accuracy'

# Parameters
BATCH = 'batch'
MODE = 'mode'
RUNITERATIONS = 'iterations'
HARDWARE = 'hardware'
PRECISION = 'precision'
# Concurrency is when the predict function is called more than once at a time
CONCURRENCY = 'concurrency'

# Precision types
FP32 = 'fp32'
FP16 = 'fp16'
INT8 = 'int8'

# Targets
CPU = 'cpu'
GPU = 'gpu'
FPGA = 'fpga'
NPU = 'npu'
VPU = 'myriad'
HDDL = 'hddl'
TPU = 'tpu'

# Units
FPS = 'fps'

# For computing latency and throughput
PERFORMANCE_PERCENTILE = 0.95
