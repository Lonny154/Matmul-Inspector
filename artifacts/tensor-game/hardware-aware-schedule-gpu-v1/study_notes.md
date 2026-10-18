# HA-Schedule-3 GPU study

Implementation commit: `c469232a3cb56dac1e4685be74ce5ee76f20d4b3`.
Primary decision: `SCHEDULE_SIGNAL_CONFIRMED` with an implementation-limit qualification.

All four GPU candidates passed five seeds at all four sizes. The retained run
has 15 rounds of 100 CUDA-event complete operations per candidate and size.
The rank-8 stored/recomputed paired median ratio is 1.101 at 128, 1.097 at
256, 1.100 at 512, and 1.082 at 1024. The 1024 round-bootstrap interval
includes parity. High 256/512 round dispersion and variable clocks limit
generalization. Nsight Compute captured only the changed kernel subset at
512, not a whole-operation traffic total.

CUDA-event times are GPU timeline times, not host invocation times. Modeled HBM
bytes and ON_CHIP capacity are abstract. Actual launches, compiler metadata,
and traffic counters are separate evidence layers. See
`docs/hardware_aware_schedule_gpu_study.md` for methods and limitations.
