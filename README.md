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

`-oracle=false` trains the same width, data, and step count with the oracle sublayer removed. Initialization still draws the oracle weights, then drops them, so the remaining weights match an oracle model from the same seed. `-compare` trains both and writes `checkpoints/oracle`, `checkpoints/no-oracle`, and `checkpoints/compare.json`. That doubles the run.

`-task halt` trains a halt head on `M#W` strings labeled by `Halts`. Half the strings halt inside the step bound and half do not. Next states are drawn from `0..states`, the same choices the oracle layer can emit. The first 90% of the strings are training examples and the last 10% are validation. The report is validation binary cross-entropy and accuracy, written to the same checkpoint directory. `-halt-n` is the number of strings (default 2048). `-more` and `-compare` apply to the language-model task.

```bash
go run . train -oracle=false
go run . train -compare -steps 1000
go run . train -task halt
```

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

## Recurrent bit model

`evolve` Huffman-codes `100.txt.utf-8` and trains a recurrent next-bit predictor by genetic search. The code is built from the first 90% of the bytes. Those bits are the training stream. The last 10% are encoded with the same code and scored after the search.

Each step feeds two bits into a one-hidden-layer net. One bit is the current Huffman bit. The other is the halt bit from the previous step, starting at 0. The net writes a total binary machine and a tape: a write bit, a move bit, and a next state in `0..states` for each transition. `Halts` on that query is the prediction of the next code bit, and that bit is fed back. The rollout is closed-loop, so the true bit is not substituted for the feedback. There are four possible inputs, so each genome calls the oracle four times and then walks the bitstream from that table.

Fitness is the fraction of next bits predicted correctly. The search keeps the best genome each generation, crosses pairs chosen by tournament, and adds gaussian noise to genes. `-bits N` scores only the first N training bits. The default scores the whole training stream. The result also prints the majority-bit baseline, the accuracy of always predicting the more common bit.

```bash
go run . evolve
go run . evolve -h
```

## Oracle network

`net` builds random networks of step-bounded total machines. Each machine reads a tape whose cells are the halt bits of other machines in the same network. A self-loop is allowed, and the same source can be drawn more than once. A halt is the character `1`. Every node has the same in-degree.

`Halts` stops at `-oracle-steps` and reports a machine that is still running as non-halting. The network is that approximation. It is not a decider for the halting problem.

The report has three parts.

Isolated machines. Every tape of length `-degree` is run. The report gives the fraction of machines whose transition on `0` is a halt, the halt rate on the all-zero tape, the halt rate over a uniform tape, and the fraction of machines whose answer depends on the tape. It also gives two per-cell scores. Influence is the mean absolute change in the halt rate when that cell flips. The effect of a 1 is the signed change, positive when a 1 makes the machine more likely to halt. Cell 0 is the symbol under the head at the start.

A feedforward stack. Every node starts non-halting. Each layer replaces the halt vector by one simultaneous round of queries. Echo is the fraction of nodes whose bit still differs after one input bit is flipped, averaged over which bit was flipped.

A recurrent iteration of the same map, once from the zero vector and once from a random vector. A repeated vector within `-iters` steps is a cycle. Length 1 is a fixed point. The mean cycle length and the halt rate are averages over the runs that closed.

```bash
go run . net
go run . net -h
```

The defaults are 32 nodes, in-degree 4, a 64-step simulator bound, 8 layers, 40 networks, 64 recurrent steps, and 2048 isolated machines, with seed 1. Those settings run once for 1-state machines and once for 2-state machines.

## Oracle network as a language model

`netlm` gives every symbol its own halting oracle and generates the symbol whose oracle does not halt. The file is split 90/10. The alphabet is the set of bytes in the fitness prefix. Each symbol is a different 2-symbol Turing machine. The machine reads the previous byte, most significant bit on cell 0. The machine that is still running at `-oracle-steps` emits its byte. If several are still running, the smallest byte is emitted. If every machine halts, the step emits nothing. The search enumerates the programs and keeps the assignment with the highest next-byte accuracy. A 1-state machine has 64 programs. A 2-state machine has 20736.

The report compares that accuracy with two baselines fit on the fitness prefix. Unigram always predicts the most common next byte. Bigram predicts the most common next byte given the previous byte. A closed-loop sample feeds each generated byte back in as the next tape.

```bash
go run . netlm
go run . netlm -h
```

The default tries 1-state and 2-state machines, keeps the higher accuracy, and uses an 8-step bound, the first 20,000 training bytes, and 2,000 validation bytes. `-states 0` is that default.

## Gaussian tape

`gauss` runs total 2-symbol machines whose tape cells are Gaussians. Each transition's write weight is drawn once from a standard normal. A blank cell is sampled from that same normal. A sample above 0 is the symbol 1. Every other sample, including 0, is the symbol 0. The write then folds its weight into the cell with Welford's update. The cell's mean is the average of the weights written there, and its deviation is the root mean square of those weights. A prepared input cell starts as one observation of `+1` or `-1`, so its first sample is that bit. The head, the state, and the step bound work as they do in `UniversalBoundedSimulator`. A run that is still going at `-oracle-steps` counts as non-halting.

The report covers 1-state and 2-state machines. It gives the halt rate on a blank tape, how often that halt bit changes when the same machine is repeated, and the halt rate on a short tape of fixed `-1` cells and of fixed `+1` cells. It also compares each machine with the binary machine that writes the sign of the same weights and reads a tape of zeros. Cell lines say how many writes land on one cell and whether that cell's deviation stays 0.

```bash
go run . gauss
go run . gauss -h
```

The defaults are a 64-step bound, 2048 machines, 16 repeats, input length 8, and seed 1.

`-text` trains a 256-symbol machine whose tape should spell that string. The machine has one write weight per state, shared by every read symbol in that state. The head starts on cell 0, writes that state's weight, then moves one cell left or right. Cells 0 through the last target byte are the string. A cell written more than once keeps the running mean, and each of those write weights receives an equal share of the partial of the loss. A write outside that window does not change the loss. The loss is the cross-entropy of a Gaussian softmax over byte codes 0..255, using the cell mean. The hard nearest-byte read is constant between integers, so the softmax carries the derivative. A write learning rate of 0 uses the temperature squared.

An empty `-moves` learns the direction at each state. The machine is unrolled into a binary tree, the way an RNN is unrolled in time, except each step branches left and right. The loss is the expected cross-entropy under the path probabilities. Each step follows the partial of that loss through the write weights and the move logits. The move after the last write does not change the tape, so its partial is 0. Each unroll covers at most 16 symbols. Longer text is trained in successive blocks of 16. Each block starts on its first symbol, and the block tapes are joined in order. `-size` is the tape length as a fraction of the symbols in that block. The default is 0.8, and the cell count is the nearest integer, at least 1. When that count is shorter than the block, the head wraps inside the tape, and each symbol is the cell mean just after that step's write. Writes then follow the partial along the mode path, and the move logits follow the partial of the expected loss over the tree. A size of 1 keeps one cell per symbol and reads the finished cells. A non-empty `-moves` is a fixed string of `L` and `R` for the whole text and does not unroll the tree. `-move-lr` of 0 uses 1. After training succeeds, the learned machine converts to a binary Turing machine. A positive weight writes 1, and every other weight writes 0. On a 0 the machine writes that bit, moves as learned, and goes to the next state. The last state halts. On a 1 it writes the other bit, moves the other way, and goes to the previous state. The first state halts. A one-state machine stays on a 1. The discrete machine overwrites each cell, so its tape can differ from the sequence of write bits. The encoded machine is then run on a blank tape, and the states it visits are printed. The learned symbol bytes are recorded in a buffer and read back as text. They are not written on a tape.

`-iris` classifies the flowers in that zip or data file. The file is the UCI iris table: four measurements and a species on each row. The head scans the four feature cells from the left. Each state writes its weight times that measurement onto an output cell, and the last write is a bias. The output cell sums those writes. The loss is the cross-entropy of a Gaussian softmax over the class symbols. A fifth of each species is held out. The split is seeded, and the features are centered and scaled on the training flowers.

`-lm` trains a byte language model on that UTF-8 file. The default text for the rest of this program is `100.txt.utf-8`. The first 90% of the bytes are training and the last 10% are validation. The head reads the previous `-order` bytes. `-order` runs from 0 to 8 and defaults to 2. Contexts are stored sparsely, so an order above 2 keeps only the byte strings that occurred. Each context's write weight is trained, by the same Gaussian partial as a spelled string, on the most common next byte. The hard read is the nearest byte to that weight. An unseen context uses the longest suffix that was written. A sample is drawn from the bytes written in the context. The log gives the training and validation cross-entropy, the nearest-byte accuracy, and the unigram accuracy.

`-bb 2` finds the initial state of the 2-state busy beaver by backpropagation. The backward pass starts at the halting state: state 0 observed at step 6, with four 1s on the tape. The tape at the initial step is all zeros. The initial state is a logit over state 1 and state 2, and gradient descent follows the partial of the expected halt cost. State 2 also halts, but earlier and with two 1s, so the partial selects state 1. That state is then run forward on the blank tape. The halting tape is a Gaussian cell of width 1 on each written square, the same width as a blank. The learned probability is the mass of that Gaussian above 0, and a sample above 0 reads as 1. Each cell's mean starts at 0, probability 1/2, and moves until the probability on the busy beaver's 1s is at least 0.9. An unwritten cell stays the blank normal.

`-bb 5` asks whether the 5-state champion reaches its halting state from a blank tape. The machine is `1RB1LC_1RC1RB_1RD0LE_1LA1LD_1RZ0LA`. A forward run from state 1 records the state at every step and the tape where the run enters state 0. The step bound is one past that observation. A run still going there is not a halt. Each state on the chain is a Gaussian softmax over the six state symbols, and gradient descent follows the partial of the mean cross-entropy. The step is one visit's partial, so a state that appears once still moves. Each visited cell, including a cell that holds 0, is a Gaussian of width 1. Gradient descent learns the mass above 0, and a mean above 0 is guessed as 1. The backward walk starts in state 0 at the halt head. The previous state and the state being undone pick the rule, the written bit is checked, and the symbol that rule read is put back. The walk has reached the blank-tape start when every cell is 0, the head is on cell 0, and the state is 1. On this machine the forward run enters state 0 at step 47176870. The visited tape is 12289 cells, 4098 of them 1 and 8191 of them 0, and the head is on cell -12242. One gradient step puts each state's mean on that state. Three steps put the mass above 0 at 0.9187 on the 1s and 0.0813 on the 0s. The guessed bits match, and the backward walk takes 47176870 steps back to a blank tape.

`-bb 5 -no-halt` runs that machine with the halt transition removed. State 5 reading 0 writes 1, moves right, and stays in state 5 (`1RB1LC_1RC1RB_1RD0LE_1LA1LD_1RE0LA`). No rule enters state 0, so the machine does not halt. The forward run stops at the same step bound, still going. Gradient descent fits the state chain and a unit-width Gaussian on each visited cell, and the bits are guessed the same way. Running backward from the state at the bound rewinds that run to a blank tape. Running backward from state 0 does not, because the machine never entered the halting state. On this machine the forward run is still going at step 47176871, in state 5. The visited tape is 12289 cells, 4099 of them 1 and 8190 of them 0, and the head is on cell -12241. The guessed bits match. Backward from state 5 takes 47176871 steps to a blank tape. Backward from state 0 takes 0 steps and the tape is not blank.

```bash
go run . gauss -text 'Hello World!' -size .8
go run . gauss -text 'Hello World!' -moves RRRRRRRRRRRLLLLLLLLLLLR
go run . gauss -iris iris.zip
go run . gauss -lm 100.txt.utf-8
go run . gauss -bb 2
go run . gauss -bb 5
go run . gauss -bb 5 -no-halt
```

## Tests

```bash
go test
```

The tests check the simulator, the machine encoding, gradients of the autograd ops, that a tiny model can reduce loss on a repeated string, that the oracle-off control and the halt-label task run, that the Huffman code round-trips and the genetic search raises next-bit accuracy, that a network passes halt bits along its tapes and sits at a fixed point when every machine ignores its tape, and that a symbol oracle emits the byte whose machine does not halt and a search over distinct programs learns a high-bit switch, and that a gaussian tape reads a positive sample as 1 and folds each write into a running mean and deviation, and that the partial of a symbol cross-entropy teaches a 256-symbol gaussian machine to write Hello World!, including when the head moves both left and right and when those directions are learned by unrolling both branches, including text longer than one block of 16, and that a tape of 0.8 of the symbols still spells the string, and that the learned machine converts to a binary machine whose writes are 0 or 1, and that running that machine prints the states it visits, and that the learned symbol bytes are recorded in a buffer, and that the same machine classifies held-out iris flowers, and that a Gaussian language model trained on Shakespeare beats a unigram on held-out bytes, and that backpropagation from the BB(2) halt finds the initial state on a blank tape, and that a Gaussian halt tape and the BB(5) state chain undo to a blank tape, and that the BB(5) machine with no halt transition rewinds its run to a blank tape and does not rewind from state 0.

## License

BSD. See [LICENSE](LICENSE).
