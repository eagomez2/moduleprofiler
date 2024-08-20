# Reference
This complementary section explains the derivation of the formulas used by `moduleprofiler` to estimate the complexity of different neural network modules. It is intended to be used both as a quick reference and for educational purposes.

!!! note
    It is important to consider that all formulas derived in this section are purely based on the mathematical relationship between the input and output of each module. In practice, there could be additional optimizations performed by linear algebra libraries or hardware-specific capabilities that are used to accelerate or minimize the calculations required by a specific module type to compute the corresponding output. For this reason, estimating complexity using `moduleprofiler` may result in bigger numbers compared to other packages used for similar purposes.

This package currently supports the following modules:

- [Linear (`torch.nn.Linear`)](modules/linear.md)
- [Conv1d (`torch.nn.Conv1d`)](modules/conv1d.md)