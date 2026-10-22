# From words to a trained translation model

Read this in two passes. First follow the forward computation through sections 1–9. Then follow how learning changes the same computation in sections 10–13. Run `transformer trace` to see real numbers at each step of one attention head.

## 1. What problem are we solving?

Given a source sequence `x`, predict a target sequence `y` one token at a time:

```text
P(y | x) = P(y1 | x)
         * P(y2 | x, y1)
         * P(y3 | x, y1, y2) * ... * P(EOS | x, y1, ..., yn)
```

The **encoder** reads the whole source and produces one context-aware vector per source position. The **decoder** uses those vectors and the target prefix to predict the next target token. EOS means “the sentence is finished.” No word-alignment table or translation rule is hardcoded.

For `i like cats` → `j aime les chats`:

```text
source to encoder: i      like   cats   <eos>
decoder input:    <bos>   j      aime   les    chats
desired output:   j      aime   les    chats  <eos>
```

The offset is crucial: at decoder position 0, the model sees BOS and predicts `j`. The label `j` is not in its visible target input. During training all positions are computed together, with a causal mask preventing future information from leaking.

## 2. Tokens are IDs, not meanings

`Vocab` assigns integer IDs to whitespace-delimited tokens. Source and target have separate vocabularies. Fixed special IDs are PAD=0, BOS=1, EOS=2, UNK=3. Other IDs are assigned in first-occurrence order in training data.

IDs are arbitrary labels: ID 12 is not more similar to ID 13 than to ID 87. Meanings are represented by learned **embeddings**. If the embedding table is `[V,d]`, indexing the rows with `T` IDs gives `[T,d]`. Repeated IDs select the same parameter row and their gradients must be added into that row.

## 3. Position must enter the computation

Self-attention has no intrinsic left-to-right ordering. Add a vector describing each position:

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
X = sqrt(d) * Embedding(token IDs) + PE
```

The paired coordinates use the same frequency. The frequencies span different scales. Position vectors are fixed constants; embeddings are trainable. In this code the initial `X` shape is `[T,d]` and no batch dimension is hidden.

## 4. One attention head, exactly

Attention lets one position collect information from other positions. Three learned projections play different roles:

- **Query**: what this position is looking for.
- **Key**: what another position offers for matching.
- **Value**: the information taken from that position if it receives weight.

For one head of width `dk`:

```text
Q = query_input Wq + bq       [Tq, dk]
K = memory_input Wk + bk      [Tk, dk]
V = memory_input Wv + bv      [Tk, dk]
S = Q K^T / sqrt(dk)         [Tq, Tk]
A = row_softmax(S + mask)    [Tq, Tk]
O = A V                     [Tq, dk]
```

`S[i,j]` is the scaled dot product between query `i` and key `j`. Softmax turns that row into positive weights summing to one. `O[i]` is a weighted mixture of value vectors. The scale helps avoid dot products becoming too large as head width increases.

For `Q[0]=[1,0]`, keys `[1,0]`, `[0,1]`, `[1,1]`, and `dk=2`:

```text
scores = [0.7071, 0, 0.7071]
weights ≈ [0.4011, 0.1978, 0.4011]
values = [[10,0], [0,10], [5,5]]
output ≈ [6.0167, 3.9833]
```

These are illustrative projected vectors. In the actual model the projection weights are learned. Attention weights need not correspond to a human-readable linguistic alignment.

## 5. The three uses of attention

| Location | Queries | Keys and values | Allowed visibility |
|---|---|---|---|
| Encoder self-attention | Encoder sequence | Same encoder sequence | All source positions |
| Decoder self-attention | Decoder sequence | Same decoder sequence | Current and earlier target positions |
| Decoder cross-attention | Decoder sequence | Final encoder memory | All source positions |

For decoder self-attention, row `i` must have zero weight at every column `j > i`:

```text
position 0: visible  hidden   hidden
position 1: visible  visible  hidden
position 2: visible  visible  visible
```

Mathematically the hidden scores are set to negative infinity before softmax. The implementation excludes them from the maximum and denominator, leaves their probabilities exactly zero, and propagates zero gradients through them. At position 0 in the example above, masked output is exactly `[10,0]`.

Cross-attention is the bridge between languages: the decoder can attend to source positions in whatever order is useful. It is **not** causally masked along the source axis.

## 6. Multiple heads

`d` must be divisible by `heads`; `dk=d/heads`. The implementation projects Q, K, and V to width `d`, slices their columns into heads, computes each head separately, concatenates their outputs, then applies a learned output projection:

```text
MultiHead = Concat(head0, ..., headH-1) Wo + bo
```

Different learned subspaces can capture different relations. Heads are not assigned specific meanings. With `d=32` and `heads=4`, each head has width 8; concatenation restores width 32. See `Attention::operator()`.

## 7. Residual paths and layer normalization

A residual addition preserves the incoming representation while adding an update:

```text
output = input + sublayer(input)
```

The backward pass splits the incoming gradient across both paths and adds their contributions when they meet. This gives an additional direct route for gradients through deep stacks.

Layer normalization independently normalizes the `d` features at **each token position**:

```text
mean = sum(x[j]) / d
variance = sum((x[j] - mean)^2) / d
normalized[j] = (x[j] - mean) / sqrt(variance + 1e-5)
output[j] = gamma[j] * normalized[j] + beta[j]
```

`gamma` and `beta` are learned vectors shared across positions. This is not batch normalization and does not mix different sentences or token positions. We use **pre-norm**: normalize the input of each sublayer, retain the original residual stream, and apply a final normalization at each stack's output.

## 8. Feed-forward layers and complete blocks

Attention mixes **positions**. A feed-forward network transforms **features**, separately at each position, using the same weights everywhere:

```text
FFN(x) = ReLU(x W1 + b1) W2 + b2
[T,d] -> [T,ff] -> [T,d]
```

`ReLU(z)=max(0,z)`. Its derivative is 1 for positive inputs and 0 for negative inputs; this code chooses 0 exactly at zero.

One encoder block:

```text
n = LayerNorm1(x)
x = x + MultiHead(n, n, n)
x = x + FFN(LayerNorm2(x))
```

One decoder block, with fixed encoder memory `m`:

```text
n = LayerNorm1(x)
x = x + MaskedMultiHead(n, n, n)
x = x + CrossMultiHead(LayerNorm2(x), m, m)
x = x + FFN(LayerNorm3(x))
```

Repeat blocks `layers` times. All blocks have independent parameters. With source length 4 and target input length 5, every self-attention head in the decoder forms a `[5,5]` matrix; every cross-attention head forms a `[5,4]` matrix. The output of either attention operation has target length 5.

## 9. Turn hidden vectors into next-token probabilities

The final decoder normalization produces `[Ttarget,d]`. A learned linear projection maps each row to the target vocabulary:

```text
logits = hidden Wvocab + bvocab      [Ttarget,Vtarget]
P(next token = j) = softmax(logits)[j]
```

Logits are arbitrary real scores, not probabilities. Greedy generation chooses the largest permitted logit; applying softmax would give the same argmax. PAD and BOS are excluded from generation, while EOS and UNK remain possible.

Inference starts with `[BOS]`, predicts one token, appends it, and repeats until EOS or the length limit. The encoder is computed once; this educational implementation recomputes the decoder prefix at every step. Greedy decoding is not guaranteed to find the most probable whole sentence.

## 10. The loss tells us what to improve

For gold next-token ID `y` and logits `z`:

```text
loss = -log softmax(z)[y]
     = max(z) + log(sum(exp(z[j]-max(z)))) - z[y]
```

Subtracting the maximum prevents overflow. For a uniform distribution over 10 tokens, loss is `log(10) ≈ 2.302585` and perplexity is 10. Raising the correct token's probability to 0.8 reduces its loss to `-log(0.8) ≈ 0.223144`.

Loss averages over all target labels, including EOS. For a batch of variable-length sentences, each sentence mean is multiplied by `its_token_count / batch_token_count` before backward. Thus every token in that batch has equal weight. Averaging sentence means equally would overweight short sentences.

Teacher forcing supplies the true prefix during training. At inference the decoder supplies its own earlier predictions; errors can compound. That is why teacher-forced token accuracy is not a substitute for evaluating freely generated translations.

## 11. How derivatives flow through the model

Every tracked operation creates a node storing its inputs and a local backward rule. Starting from `dLoss/dLoss=1`, `backward()` visits nodes in reverse topological order, applying the chain rule. Parameters are leaf tensors whose gradient arrays accumulate contributions from all uses.

Essential local rules (`G` means the incoming gradient):

```text
Y = A B           -> dA += G B^T; dB += A^T G
Y = A + B         -> dA += G; dB += G (sum rows for a shared bias)
Y = c A           -> dA += c G
p = softmax(z)    -> dz[j] += p[j] * (G[j] - sum_k G[k]*p[k])
CE(logits, y)     -> dlogits[j] += (p[j] - indicator(j==y)) / token_count
```

For layer norm, let `h` be the normalized row, `g=G*gamma`, and `r=1/sqrt(variance+eps)`. Then:

```text
dx[j] += r * (g[j] - mean(g) - h[j]*mean(g*h))
dgamma[j] += G[j]*h[j]
dbeta[j] += G[j]
```

Slice/concatenate/transpose route gradients back to the original positions. Embedding backward adds each output row's gradient into the selected table row. Those additions matter when a token occurs more than once. Attention's derivatives emerge by composing these rules; there is no hidden library implementation.

Intermediate node gradients reset for each backward call. Parameter gradients accumulate across sentences until the optimizer update. `NoGrad` prevents graph construction during inference/evaluation. Backward closures reference their parents but never own their output node, avoiding a shared-pointer reference cycle.

## 12. Adam changes the weights

First compute the global gradient norm and, if necessary, scale **all** gradients by the same factor so the norm is at most 1. Then Adam keeps two moving statistics per parameter:

```text
m = 0.9*m + 0.1*g
v = 0.999*v + 0.001*g*g
mhat = m / (1 - 0.9^step)
vhat = v / (1 - 0.999^step)
weight -= learning_rate * mhat / (sqrt(vhat) + 1e-8)
```

Moments begin at zero, hence the bias correction. One optimizer step occurs per gradient-accumulated batch. Graphs are discarded after backward; gradients are cleared at the start of the next batch. There is no dropout, weight decay, or learning-rate schedule in this baseline.

## 13. What exactly does BLEU calculate?

Generate translations without gold target prefixes. For each sentence and n-gram order `n=1..4`, count candidate n-grams and cap each match count at its count in the reference. Then add those clipped counts across sentences:

```text
p_n = total clipped matching n-grams / total candidate n-grams
BP = exp(min(0, 1 - reference_length / candidate_length))
BLEU4 = 100 * BP * exp((log(p1)+log(p2)+log(p3)+log(p4))/4)
```

Clipping stops repeating a reference word from earning unlimited credit. The brevity penalty discourages very short guesses. This implementation returns zero for an empty candidate corpus or a missing/zero precision. Counts are pooled before taking logs; this is not the average of sentence BLEU scores.

`BLEU_effective` omits orders for which the entire candidate corpus contains no n-grams. It does not smooth zero matches at an available order. For example, an exact two-token translation scores 0 with strict BLEU4 and 100 with effective-order BLEU. Both use the program's specified tokenization, not an external benchmark tokenizer.

Unknown target words are collapsed to UNK when calculating CE; high target-UNK rates can therefore make CE misleading. Generated text metrics compare against original reference words and cannot receive credit just because multiple unseen words share an UNK ID.

## 14. How to convince yourself the implementation is correct

`transformer test` checks analytical gradients against central differences:

```text
dLoss/dw ≈ (Loss(w + 1e-5) - Loss(w - 1e-5)) / (2e-5)
```

It checks every element of small primitive examples, and up to eight evenly spaced elements in **every parameter tensor** of a two-layer Transformer. Numerical checks use absolute and relative tolerances and are sampled, not a proof covering all possible inputs. Additional tests verify causal prefix invariance, dependence on source memory, repeated embedding IDs, gradient accumulation, stable loss, BLEU edge cases, checkpoint round-trip, Adam, and actual reduction of loss followed by correct greedy generation.

Try these experiments in order:

1. In `trace`, change one value vector while leaving Q and K unchanged. Attention weights remain the same; weighted output changes.
2. Change one key. Its matching scores and the normalized weights change.
3. Compare full and causal attention. Confirm the first causal row has exactly one nonzero entry.
4. Train the toy corpus and follow CE, generated exact match, and EOS rate. Their improvements need not happen at the same time.
5. Hold out some sentence combinations manually. Keep the vocabulary known but evaluate unseen combinations. Compare memorization with generalization.
6. Change `--heads`, keeping `--d` divisible by it. Then compare model capacity, convergence, and runtime.

The baseline is complete enough to learn and debug the full training process. Once your real data is available, language pair, segmentation, dataset size, length distribution, and hardware should determine the next engineering steps.
