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

## Training

`100.txt.utf-8` is the training text (Project Gutenberg eBook #100). The first 90% of the bytes are training windows. The last 10% are validation. The two slices do not overlap.

```bash
go run . train
go run . train -h
```

Defaults are 1000 steps, batch 16, context 64, learning rate `3e-4`, and 100 warmup steps. A longer run:

```bash
go run . train -steps 6000 -eval-every 200 -warmup 200
```

On that command, seed 42, validation next-byte cross-entropy was 1.82 nats (perplexity 6.19), down from 5.53 (perplexity 253) at initialization.

Checkpoints go to `checkpoints/` (`best.gob`, `result.json`, `train.log`). That directory is gitignored. Pass `-out ""` to skip writing it, and `-sample 0` to skip the generated continuation.

## Tests

```bash
go test
```

The tests check the simulator, the machine encoding, gradients of the autograd ops, and that a tiny model can reduce loss on a repeated string.

## License

BSD. See [LICENSE](LICENSE).
