# Validation performed

Validated on Windows x64 on 22 October 2026 using LLVM-MinGW, Clang 23.1.4, C++17. The initial locally installed GCC could not start due to a Windows DLL relocation error, so a portable LLVM-MinGW toolchain was downloaded into the scratch workspace with network permission. No global compiler installation was changed. The toolchain is not part of this deliverable.

## Correctness

Direct release compilation with `-O2 -Wall -Wextra -Wpedantic -static` completed without warnings. A separate CMake release build using the MinGW Makefiles generator also succeeded, and CTest passed 1/1 registered tests. The test executable passed:

| Numerical gradient checks | Coordinates tested |
|---|---:|
| Matrix multiplication, broadcast bias, cross-entropy | 22 |
| Layer normalization, including gamma/beta | 12 |
| ReLU | 6 |
| Causal softmax | 9 |
| Repeated embedding indices | 15 |
| Transpose, column slicing, concatenation | 6 |
| Shared graph / residual path | 3 |
| Complete two-layer encoder–decoder Transformer | 524 |
| **Total** | **597** |

All numerical derivatives matched the analytical backward implementation within the tolerances in `tests.hpp`. Additional passing checks covered future-token isolation, dependence on source memory, no-gradient inference, repeated backward accumulation, stable cross-entropy, BLEU clipping/aggregation/brevity/short-sentence cases, target shifting, unknown IDs, exact checkpoint round-trip, and an Adam update with clipping.

The built-in learning test reduced CE from **2.290666 to 0.002465** in 80 updates and greedily generated the correct target followed by EOS.

## Full command-line training and inference

The supplied 12-pair toy dataset trained with default architecture/optimizer settings for 150 epochs:

| Training-set evaluation | After epoch 1 | After epoch 150 |
|---|---:|---:|
| CE | 2.204934 | 0.001205 |
| Perplexity | 9.069650 | 1.001205 |
| Teacher-forced token accuracy | 35% | 100% |
| Generated exact match | 0% | 100% |
| Corpus BLEU4 | 0 | 100 |

The final checkpoint is included as `toy.model` in the local packaged delivery; generated checkpoints are excluded from the Git repository. The README's training command reproduces this experiment. Separate-process loading, evaluation, and generation succeeded:

```text
i like cats  -> j aime les chats
we see dogs  -> nous voyons les chiens
you see dogs -> tu vois les chiens
```

These scores measure **memorization of the supplied training examples**, not general translation ability. The French is simplified for whitespace tokenization.

## Validation selection and unknown vocabulary

A separate pipeline check used the first 10 toy rows as training and the last two as validation, with `max_len=12`, 30 epochs, and evaluation every 10 epochs. The held-out target word `voyons` was absent from the training vocabulary, giving a 25% target-UNK rate.

Validation CE was 3.580874, 2.660605, 2.581279, and 3.024570 at epochs 1, 10, 20, and 30. Reloading the saved checkpoint returned CE **2.581279**, confirming that the best evaluated checkpoint was retained, rather than the last one. Generated exact match and BLEU4 were zero on this intentionally difficult split. Original unknown reference words were preserved for generated-text evaluation.

## Interpretation and limits

The tests provide evidence for numerical gradients, masking, training, serialization, inference, and metric logic on small examples. They do not establish production robustness, large-corpus speed, or translation quality beyond the toy data. Gradient checks sample the full model's parameter tensors; they are not exhaustive over every parameter and possible input. No external training corpus was used.
