# Oracle

A halting oracle, a grammar that uses it, and a byte-level transformer that queries the same oracle while learning text.

The oracle is `HaltingOracle`:

```go
Halts(machineEncoding string, input string) bool
```

`UniversalBoundedSimulator` is the implementation. It runs a binary Turing machine for at most `MaxSteps` steps. If the machine enters state 0, `Halts` returns true. If the step limit is reached, it returns false. A machine that would run forever is reported as non-halting once the bound is hit. Malformed encodings are treated as halting immediately.

Each state is a 16-bit block: 8 bits for the transition on `0`, 8 bits for the transition on `1`. Inside a transition, bit 0 is the write symbol, bit 1 is the move (`0` left, `1` right), and bits 2–7 are the next state (`0` means halt).

## Grammar

`HyperGrammar` is the language of strings `M#W` where `M` is a machine and `W` is a tape, and the oracle says `M` does not halt on `W`.

```bash
go run .
```

That prints a parse of a halting machine and a non-halting machine, then a random string accepted by the grammar. `ss/` is an earlier, smaller version of the same idea (`go run ./ss`).

## Transformer

`OracleTransformer` is a decoder-only transformer. Every block pools its residual stream, maps that vector to a total binary machine and a tape, and calls `HaltingOracle`. The halt-or-loop bit is embedded and added back into the residual stream.

The simulator is not differentiated. A straight-through estimator on the machine bits, plus a small loss that predicts the oracle bit, trains the query. The projections that write the oracle answer start at zero, so the first step matches an ordinary transformer.

The default model is a byte-level language model: vocabulary 256, 3 layers, width 64, 4 heads, context 64. Each query is a 2-state machine on an 8-bit tape, simulated for at most 64 steps.

Matrix products use the standard library `simd` package when Go is built with `GOEXPERIMENT=simd`. On this machine that is 128-bit NEON: four float32 lanes and a fused multiply-add. With the experiment off, those kernels are scalar float32. A product narrower than one vector matches the scalar kernel bit for bit. Wider products agree to about `1e-4` relative.

## Training

`100.txt.utf-8` is the training text (Project Gutenberg eBook #100). The first 90% of the bytes are training windows. The last 10% are validation. The two slices do not overlap.

```bash
go run . train
go run . train -h
```

With no `-steps` value, training walks that 90% once. Windows start every 64 bytes, and one last window covers the tail. Validation scores every window in the remaining 10%. Batch size is 16, the learning rate is `3e-4`, and warmup is 100 steps. `-steps 1000` instead draws that many random training windows and validates on a sample of the holdout.

`-more` appends the first 10 texts in `txt-files.tar.zip` to the training split. The validation suffix of `100.txt.utf-8` stays as it is.

An earlier random run, `go run . train -steps 6000 -eval-every 200 -warmup 200` with seed 42, reached validation next-byte cross-entropy 1.82 nats (perplexity 6.19), down from 5.53 (perplexity 253) at initialization.

Checkpoints go to `checkpoints/` (`best.gob`, `result.json`, `train.log`). That directory is gitignored. Pass `-out ""` to skip writing it, and `-sample 0` to skip the generated continuation.

## Inference

`infer` loads a checkpoint and continues a prompt. Each new byte is one random draw from the model's next-byte distribution. Logits are divided by the temperature before the softmax: `1` leaves the distribution unchanged, lower values make it sharper, and `0` is greedy.

```bash
go run . infer -prompt "To be, or not to be" -temp 0.8 -n 200
go run . infer -h
```

The default checkpoint is `checkpoints/best.gob`. A negative `-seed` (the default) uses the current time, so two runs differ. Pass `-seed 1` to repeat a sample.

`-mcts` draws `-sims` continuations. Each byte is sampled from the temperature softmax, and a continuation is scored by the sum of the log probabilities of its bytes. The continuation with the highest sum is printed.

```bash
go run . infer -mcts -sims 8 -prompt "To be, or not to be" -temp 0.8 -n 80
```

## Tests

```bash
go test
```

The tests check the simulator, the machine encoding, gradients of the autograd ops, and that a tiny model can reduce loss on a repeated string.

## License

BSD. See [LICENSE](LICENSE).
