# Understand and train a Transformer in C++

A working, educational **encoder–decoder Transformer for translation**, built from scratch in C++17 using only the standard library. This includes the tensor operations, reverse-mode automatic differentiation, optimizer, tokenizer, training loop, inference, and evaluation metrics. No PyTorch, LibTorch, Eigen, BLAS, or metric packages.

Start with **[the illustrated-by-numbers guide](docs/HOW_IT_WORKS.md)**. Then run `trace`, read the corresponding small functions, and train the toy model. The supplied English–French data is deliberately tiny and simplified: learning it demonstrates memorization and a working training pipeline, not useful general translation.

## 1. Run or build, then inspect

The repository contains the source, tests, documentation, and toy data. Build the executable using the steps below, then train a toy checkpoint in section 2. Executables, build directories, and generated checkpoints are excluded from Git. The separate local packaged delivery also includes a Windows x64 executable and a trained toy checkpoint. See [VALIDATION.md](docs/VALIDATION.md) for the checks and observed results.

Use a C++17 compiler and CMake. From this project directory:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Executable locations depend on the generator:

| Build | Executable |
|---|---|
| Windows Visual Studio | `build/Release/transformer.exe` |
| Windows Ninja / MinGW | `build/transformer.exe` |
| Linux / macOS | `build/transformer` |

You can also compile directly:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude src/main.cpp -o transformer
```

The commands below use `./transformer`. In PowerShell, use `.\transformer.exe`, or substitute your build executable path.

```sh
./transformer trace
./transformer test
```

`trace` prints actual query/key/value matrices, attention probabilities before and after a causal mask, weighted values, and the shapes throughout the full model. `test` runs finite-difference gradient checks and behavioral checks, including actual learning and greedy generation.

## 2. Train a complete small example

```sh
./transformer train --train data/toy_train.tsv --save toy.model --epochs 150 --eval-every 25
./transformer translate --model toy.model --text "i like cats"
./transformer translate --model toy.model --text "we see dogs"
./transformer evaluate --model toy.model --data data/toy_train.tsv
```

Training-set evaluation is a plumbing check. It does **not** estimate performance on new sentences. No checkpoint is pretrained on external data.

Training defaults: `d=32`, `heads=4`, `ff=64`, `layers=1` in each stack, `max_len=64`, `batch=8`, `lr=0.003`, `seed=42`. Every option is printed by running the executable without arguments. The seed makes a run repeatable with the same compiler/toolchain; C++ random distribution implementations can differ across platforms.

## 3. Use your translation data later

Provide UTF-8 text files with **one source sentence, one tab, one target sentence per line**. No header. Example:

```text
i like cats<TAB>j aime les chats
you see dogs<TAB>tu vois les chiens
```

`<TAB>` means an actual tab, not those five characters. The supplied TSV has actual tabs. Empty lines are ignored; malformed rows and empty sentences are rejected. UTF-8 BOM and Windows CRLF are supported. Do not include the reserved strings `<pad>`, `<bos>`, `<eos>`, `<unk>` as ordinary words.

Prepare separate `train.tsv`, `valid.tsv`, and `test.tsv` files. Keep related documents and duplicate sentence pairs in the same split. Build preprocessing rules on training data only, and apply the same rules to all splits. The program learns vocabulary **only from training data**; unseen validation/test words become `<unk>`.

```sh
./transformer train --train train.tsv --valid valid.tsv --save translation.model --epochs 100 --d 64 --heads 4 --ff 128 --layers 2 --batch 8 --lr 0.001 --max-len 128
./transformer evaluate --model translation.model --data test.tsv
./transformer translate --model translation.model --text "your source sentence"
```

Choose capacity and training duration after seeing the data; the example values are a starting experiment, not tuned settings. With validation, the checkpoint with lowest validation cross-entropy among evaluation epochs is saved. Without validation, final weights are saved. `--save` overwrites that file and its parent directory must already exist. Evaluation runs at epoch 1, each `--eval-every` interval, and the final epoch. The final model in memory may differ from the best saved checkpoint.

The checkpoint stores configuration, both vocabularies, and every parameter with round-trip double precision. It supports fresh-process inference and evaluation. It does **not** store optimizer state or support resuming training.

## Metrics implemented here

| Output | Meaning |
|---|---|
| `online_train_CE` | Token-weighted training loss measured as weights change during the epoch |
| `CE` | Mean target-token negative log probability under teacher forcing, including EOS |
| `PPL` | `exp(CE)`; lower is better, comparable only with the same tokenization/vocabulary |
| `teacher_token_acc` | Correct next-token argmax predictions with gold prefixes, including EOS |
| `generated_exact` | Percentage of freely generated sentences matching all reference tokens and emitting EOS |
| `BLEU4` | Corpus BLEU with orders 1–4, clipped counts, brevity penalty, no smoothing, 0–100 |
| `BLEU_effective` | Same, but omits n-gram orders with no candidate n-grams; useful on very short examples |
| `EOS_rate` | Percentage of generations that terminate before the length cap |
| `source_UNK`, `target_UNK` | Unknown-token rates on original words, excluding special tokens |

BLEU uses a single reference, case-sensitive whitespace tokens, and original reference words rather than their potentially collapsed UNK IDs. Strict `BLEU4` is zero when any of its four precisions is zero or undefined, including perfect corpora containing only sentences shorter than four tokens. Effective-order BLEU is explicitly a variant. Neither number is automatically comparable with published BLEU using different tokenization or smoothing. Both exclude special tokens. Do not optimize BLEU directly with backpropagation: training minimizes differentiable cross-entropy; BLEU evaluates generated text.

## Read the source in this order

1. `include/tensor.hpp`: matrices, the computation graph, elementary forward and backward rules.
2. `include/model.hpp`: embeddings, positions, multi-head attention, encoder/decoder, Adam.
3. `include/data.hpp`: vocabulary, shifted labels, TSV input, checkpoint format.
4. `include/metrics.hpp`: teacher-forced evaluation and corpus BLEU.
5. `src/main.cpp`: training, numerical walkthrough, and command-line entry points.
6. `include/tests.hpp`: numerical derivatives and executable correctness checks.

## Deliberate simplifications

This is a CPU learning implementation, not a production translation system. It uses doubles and straightforward matrix loops. It has pre-layer normalization and final stack normalization, sinusoidal positions, ReLU feed-forward layers, and independent source/target embeddings. This is a Transformer variant, not a line-for-line reproduction of the original paper's post-norm training recipe.

There is no dropout, label smoothing, learning-rate scheduler, subword tokenizer, weight tying, GPU execution, mixed precision, beam search, or decoder KV cache. Each sentence has its own graph; gradients are accumulated for a token-weighted batch update. There is no sequence padding or padding mask because no padded sequence is processed. PAD is reserved for a future batching extension and cannot be generated. Decoder self-attention does use a causal mask.

Whitespace tokenization preserves UTF-8 bytes but does not normalize Unicode, split punctuation, or segment languages without spaces. For larger real translation data, a learned subword tokenizer, efficient matrix kernels, padded batches with masks, and a careful generalization experiment would be the next phase. Vocabulary projection and naive attention make this code slow on large datasets. The program rejects overlong examples instead of silently truncating them. `max_len` includes source EOS and target BOS/EOS positions.

## Sources

The architecture follows the encoder–decoder attention formulation in [Attention Is All You Need](https://arxiv.org/abs/1706.03762), with the pre-norm differences described above. Corpus BLEU follows clipped n-gram precision and brevity penalty from [Papineni et al. (2002)](https://aclanthology.org/P02-1040/). All code in this project is written here from scratch.
