# Computational-cost analysis

The cost comparison is based on the frozen hardware, software, checkpoint,
fold, seed, and batch-size settings. Training time is averaged per outer fold;
inference and preprocessing time are averaged over individual prefixes with
batch size one. Peak memory is the maximum CUDA memory allocated during one
representative frozen training step (forward, frozen training loss,
backward, and Adam update), measured for all three folds and three seeds.

D²Pred has 191,405 trainable parameters, an average training time of 55.93 s
per fold, an inference time of 4.805 ms per prefix, a preprocessing time of
71.693 ms per prefix, and a training peak memory of 0.0378 GiB. The compared
baselines range from 6,465 to 115,650 parameters, 6.00--32.53 s per fold,
0.275--1.353 ms per prefix for inference, and 0.0088--0.0527 GiB for training
peak memory. D²Pred's additional preprocessing cost is substantially larger
than that of the baselines because the reported preprocessing stage includes
causal decomposition and AEP token construction; this cost is therefore not
the forward-pass cost of the regression head alone.

These values describe a computational trade-off under the frozen measurement
protocol. They do not establish a hardware-independent efficiency ranking,
nor do they imply real-time suitability for any method. In particular,
comparisons should not be generalized beyond the stated hardware, software,
prefix-level timing unit, and training-step memory definition.
