// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
)

// recurSpec is the feedforward net that writes a halting-oracle query.
// Input 0 is the previous halt bit. Input 1 is the current Huffman bit.
// The net has one sigmoid hidden layer. Each transition is a write logit,
// a move logit, and a next-state categorical over 0..states (0 means halt).
// The remaining outputs are tape bits. Halts on that machine is the prediction
// of the next Huffman bit, and it is fed back as input 0 on the following step.
type recurSpec struct {
	Hidden   int
	States   int
	Tape     int
	MaxSteps int
}

func (s recurSpec) nOut() int {
	return s.States*2*(2+s.States+1) + s.Tape
}

func (s recurSpec) nGenes() int {
	h, o := s.Hidden, s.nOut()
	return h*2 + h + o*h + o
}

func (s recurSpec) validate() error {
	if s.Hidden < 1 {
		return fmt.Errorf("hidden units must be positive")
	}
	if s.States < 1 || s.States > 63 {
		return fmt.Errorf("states must be in 1..63")
	}
	if s.Tape < 1 || s.Tape > 64 {
		return fmt.Errorf("tape length must be in 1..64")
	}
	if s.MaxSteps < 2 {
		return fmt.Errorf("oracle steps must be at least 2")
	}
	return nil
}

// EvolveConfig trains the recurrent halt predictor by genetic search.
type EvolveConfig struct {
	Data     []byte
	DataPath string
	Pop      int
	Gens     int
	Hidden   int
	States   int
	Tape     int
	MaxSteps int
	Bits     int // training-bit prefix; 0 uses the whole training split
	MutRate  float64
	MutSigma float64
	Seed     int64
	OutDir   string
	Log      func(string)
}

// EvolveResult is the best genome and its bit accuracy.
type EvolveResult struct {
	TrainBytes  int       `json:"train_bytes"`
	ValBytes    int       `json:"val_bytes"`
	Symbols     int       `json:"huffman_symbols"`
	TrainBits   int       `json:"train_bits"`
	ValBits     int       `json:"val_bits"`
	FitBits     int       `json:"fit_bits"`
	BitsPerByte float64   `json:"bits_per_byte"`
	Pop         int       `json:"pop"`
	Gens        int       `json:"gens"`
	Hidden      int       `json:"hidden"`
	States      int       `json:"states"`
	Tape        int       `json:"tape"`
	OracleSteps int       `json:"oracle_steps"`
	Genes       int       `json:"genes"`
	TrainAcc    float64   `json:"train_acc"`
	ValAcc      float64   `json:"val_acc"`
	Majority    float64   `json:"majority_baseline"`
	Rules       [4]int    `json:"rules"`
	Genome      []float64 `json:"genome"`
}

func (cfg *EvolveConfig) logf(format string, args ...any) {
	if cfg.Log != nil {
		cfg.Log(fmt.Sprintf(format, args...))
	}
}

func logistic(x float64) float64 {
	if x > 20 {
		return 1
	}
	if x < -20 {
		return 0
	}
	return 1 / (1 + math.Exp(-x))
}

func forwardLogits(g []float64, spec recurSpec, feedback, data int) []float64 {
	h := spec.Hidden
	o := spec.nOut()
	ensure(len(g) == spec.nGenes(), "genome %d != %d", len(g), spec.nGenes())
	w1 := g[:h*2]
	b1 := g[h*2 : h*2+h]
	w2 := g[h*2+h : h*2+h+o*h]
	b2 := g[h*2+h+o*h:]
	in0 := float64(feedback)
	in1 := float64(data)
	hid := make([]float64, h)
	for i := 0; i < h; i++ {
		hid[i] = logistic(b1[i] + w1[i*2]*in0 + w1[i*2+1]*in1)
	}
	out := make([]float64, o)
	for i := 0; i < o; i++ {
		z := b2[i]
		row := w2[i*h : (i+1)*h]
		for j := 0; j < h; j++ {
			z += row[j] * hid[j]
		}
		out[i] = logistic(z)
	}
	return out
}

// queryFromLogits turns net outputs into a machine encoding and a tape.
func queryFromLogits(y []float64, states, tape int) (string, string) {
	per := 2 + states + 1
	mp := make(map[int][2]Transition, states)
	p := 0
	for s := 1; s <= states; s++ {
		var pair [2]Transition
		for sym := 0; sym < 2; sym++ {
			write := byte('0')
			if y[p] >= 0.5 {
				write = '1'
			}
			move := -1
			if y[p+1] >= 0.5 {
				move = 1
			}
			next := 0
			best := y[p+2]
			for k := 1; k <= states; k++ {
				if y[p+2+k] > best {
					best = y[p+2+k]
					next = k
				}
			}
			pair[sym] = Transition{WriteSymbol: write, MoveDir: move, NextState: next}
			p += per
		}
		mp[s] = pair
	}
	w := make([]byte, tape)
	for i := 0; i < tape; i++ {
		if y[p+i] >= 0.5 {
			w[i] = '1'
		} else {
			w[i] = '0'
		}
	}
	return EncodeMachine(mp), string(w)
}

// haltTable evaluates the oracle on the four possible inputs.
// Index with table[feedback][data].
func haltTable(g []float64, spec recurSpec, sim *UniversalBoundedSimulator) [2][2]int {
	var table [2][2]int
	for feedback := 0; feedback < 2; feedback++ {
		for data := 0; data < 2; data++ {
			m, w := queryFromLogits(forwardLogits(g, spec, feedback, data), spec.States, spec.Tape)
			if sim.Halts(m, w) {
				table[feedback][data] = 1
			}
		}
	}
	return table
}

// predictAcc is the fraction of next bits matched by a closed-loop rollout.
// The feedback bit starts at 0. Each prediction is fed back; the true bit is not.
func predictAcc(bits []byte, table [2][2]int) float64 {
	acc, _, _ := predictDetail(bits, table)
	return acc
}

func predictDetail(bits []byte, table [2][2]int) (acc float64, correct, total int) {
	if len(bits) < 2 {
		return 0, 0, 0
	}
	h := 0
	total = len(bits) - 1
	for t := 0; t < total; t++ {
		pred := table[h][int(bits[t])]
		if pred == int(bits[t+1]) {
			correct++
		}
		h = pred
	}
	return float64(correct) / float64(total), correct, total
}

func majorityBaseline(bits []byte) float64 {
	if len(bits) < 2 {
		return 0
	}
	ones := 0
	total := len(bits) - 1
	for t := 1; t < len(bits); t++ {
		ones += int(bits[t])
	}
	best := ones
	if total-ones > best {
		best = total - ones
	}
	return float64(best) / float64(total)
}

func initPop(rng *rand.Rand, n, genes int) [][]float64 {
	pop := make([][]float64, n)
	for i := range pop {
		g := make([]float64, genes)
		for j := range g {
			g[j] = rng.NormFloat64()
		}
		pop[i] = g
	}
	return pop
}

func tournament(rng *rand.Rand, fits []float64, k int) int {
	best := rng.Intn(len(fits))
	for i := 1; i < k; i++ {
		j := rng.Intn(len(fits))
		if fits[j] > fits[best] {
			best = j
		}
	}
	return best
}

func crossover(rng *rand.Rand, a, b []float64) []float64 {
	c := make([]float64, len(a))
	for i := range a {
		if rng.Intn(2) == 0 {
			c[i] = a[i]
		} else {
			c[i] = b[i]
		}
	}
	return c
}

func mutate(rng *rand.Rand, g []float64, rate, sigma float64) {
	for i := range g {
		if rng.Float64() < rate {
			g[i] += rng.NormFloat64() * sigma
		}
	}
}

func breed(rng *rand.Rand, pop [][]float64, fits []float64, rate, sigma float64) [][]float64 {
	next := make([][]float64, len(pop))
	bi := 0
	for i := 1; i < len(fits); i++ {
		if fits[i] > fits[bi] {
			bi = i
		}
	}
	next[0] = append([]float64(nil), pop[bi]...)
	for i := 1; i < len(pop); i++ {
		a := tournament(rng, fits, 3)
		b := tournament(rng, fits, 3)
		child := crossover(rng, pop[a], pop[b])
		mutate(rng, child, rate, sigma)
		next[i] = child
	}
	return next
}

func evalPop(pop [][]float64, spec recurSpec, sim *UniversalBoundedSimulator, bits []byte) []float64 {
	fits := make([]float64, len(pop))
	parallelN(len(pop), func(i, j int) {
		for k := i; k < j; k++ {
			fits[k] = predictAcc(bits, haltTable(pop[k], spec, sim))
		}
	})
	return fits
}

func bestIndex(fits []float64) int {
	bi := 0
	for i := 1; i < len(fits); i++ {
		if fits[i] > fits[bi] {
			bi = i
		}
	}
	return bi
}

func meanFit(fits []float64) float64 {
	var s float64
	for _, v := range fits {
		s += v
	}
	return s / float64(len(fits))
}

// Evolve Huffman-codes the text and searches feedforward weights by genetic optimization.
// Fitness is closed-loop next-bit accuracy on the training bitstream.
func Evolve(cfg EvolveConfig) (*EvolveResult, error) {
	if cfg.Pop <= 0 {
		cfg.Pop = 32
	}
	if cfg.Gens <= 0 {
		cfg.Gens = 25
	}
	if cfg.Hidden <= 0 {
		cfg.Hidden = 8
	}
	if cfg.States <= 0 {
		cfg.States = 1
	}
	if cfg.Tape <= 0 {
		cfg.Tape = 1
	}
	if cfg.MaxSteps <= 0 {
		cfg.MaxSteps = 64
	}
	if cfg.MutRate < 0 || cfg.MutRate > 1 {
		return nil, fmt.Errorf("mutation rate must be in 0..1")
	}
	if cfg.MutSigma < 0 {
		return nil, fmt.Errorf("mutation sigma must be non-negative")
	}
	if cfg.Pop < 2 {
		return nil, fmt.Errorf("population must be at least 2")
	}
	spec := recurSpec{Hidden: cfg.Hidden, States: cfg.States, Tape: cfg.Tape, MaxSteps: cfg.MaxSteps}
	if err := spec.validate(); err != nil {
		return nil, err
	}
	data := cfg.Data
	if data == nil {
		if cfg.DataPath == "" {
			return nil, fmt.Errorf("no training data")
		}
		b, err := os.ReadFile(cfg.DataPath)
		if err != nil {
			return nil, err
		}
		data = b
	}
	train, val, err := split90(data)
	if err != nil {
		return nil, err
	}
	freq := freqOf(train)
	added := 0
	for _, b := range val {
		if freq[b] == 0 {
			freq[b] = 1
			added++
		}
	}
	code, err := buildHuffman(freq)
	if err != nil {
		return nil, err
	}
	trainPacked, trainN, err := encodeHuffman(code, train)
	if err != nil {
		return nil, err
	}
	valPacked, valN, err := encodeHuffman(code, val)
	if err != nil {
		return nil, err
	}
	trainBits := unpackBits(trainPacked, trainN)
	valBits := unpackBits(valPacked, valN)
	fitBits := trainBits
	if cfg.Bits > 0 && cfg.Bits < len(fitBits) {
		fitBits = fitBits[:cfg.Bits]
	}
	if len(fitBits) < 2 {
		return nil, fmt.Errorf("need at least 2 training bits, got %d", len(fitBits))
	}
	sim := &UniversalBoundedSimulator{MaxSteps: spec.MaxSteps}
	rng := rand.New(rand.NewSource(cfg.Seed))
	pop := initPop(rng, cfg.Pop, spec.nGenes())
	base := majorityBaseline(fitBits)

	cfg.logf("recurrent halt predictor")
	cfg.logf("  huffman %d symbols, train %d bytes -> %d bits (%.3f bits/byte), val %d bytes -> %d bits",
		huffmanSymbols(code), len(train), trainN, float64(trainN)/float64(len(train)), len(val), valN)
	if added > 0 {
		cfg.logf("  codebook added %d validation bytes missing from the training prefix", added)
	}
	cfg.logf("  fitness bits %d of %d, majority baseline %.4f", len(fitBits), len(trainBits), base)
	cfg.logf("  feedforward hidden %d, %d-state machines, tape %d, oracle %d steps, genes %d",
		spec.Hidden, spec.States, spec.Tape, spec.MaxSteps, spec.nGenes())
	cfg.logf("  genetic search pop %d, gens %d, mutation %.2f / sigma %.2f, seed %d",
		cfg.Pop, cfg.Gens, cfg.MutRate, cfg.MutSigma, cfg.Seed)

	var best []float64
	var bestFit float64
	for gen := 1; gen <= cfg.Gens; gen++ {
		fits := evalPop(pop, spec, sim, fitBits)
		bi := bestIndex(fits)
		if best == nil || fits[bi] > bestFit {
			bestFit = fits[bi]
			best = append([]float64(nil), pop[bi]...)
		}
		cfg.logf("gen %d/%d  best %.4f  mean %.4f", gen, cfg.Gens, fits[bi], meanFit(fits))
		if gen == cfg.Gens {
			break
		}
		pop = breed(rng, pop, fits, cfg.MutRate, cfg.MutSigma)
	}
	table := haltTable(best, spec, sim)
	trainAcc := predictAcc(trainBits, table)
	valAcc := 0.0
	if len(valBits) >= 2 {
		valAcc = predictAcc(valBits, table)
	}
	var rules [4]int
	for h := 0; h < 2; h++ {
		for x := 0; x < 2; x++ {
			rules[h*2+x] = table[h][x]
			cfg.logf("  feedback %d data %d -> halt %d", h, x, table[h][x])
		}
	}
	res := &EvolveResult{
		TrainBytes:  len(train),
		ValBytes:    len(val),
		Symbols:     huffmanSymbols(code),
		TrainBits:   len(trainBits),
		ValBits:     len(valBits),
		FitBits:     len(fitBits),
		BitsPerByte: float64(trainN) / float64(len(train)),
		Pop:         cfg.Pop,
		Gens:        cfg.Gens,
		Hidden:      spec.Hidden,
		States:      spec.States,
		Tape:        spec.Tape,
		OracleSteps: spec.MaxSteps,
		Genes:       spec.nGenes(),
		TrainAcc:    trainAcc,
		ValAcc:      valAcc,
		Majority:    majorityBaseline(trainBits),
		Rules:       rules,
		Genome:      best,
	}
	cfg.logf("train acc %.4f  val acc %.4f  majority %.4f", res.TrainAcc, res.ValAcc, res.Majority)
	if cfg.OutDir != "" {
		path := filepath.Join(cfg.OutDir, "recur.json")
		if err := writeResult(path, res); err != nil {
			return nil, err
		}
		cfg.logf("wrote %s", path)
	}
	return res, nil
}

func evolveFromArgs(args []string) error {
	fs := flag.NewFlagSet("evolve", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: oracle evolve [flags]\n\nHuffman-code the text and train a recurrent next-bit predictor by genetic search.\nEach step feeds the current code bit and the previous halt bit into a feedforward net.\nThe net writes a machine and a tape. The halt oracle's bit is the prediction of the next code bit, and it is fed back.\n\n")
		fs.PrintDefaults()
	}
	data := fs.String("data", "100.txt.utf-8", "UTF-8 text file")
	pop := fs.Int("pop", 32, "population size")
	gens := fs.Int("gens", 25, "generations")
	hidden := fs.Int("hidden", 8, "feedforward hidden units")
	states := fs.Int("states", 1, "states in each queried machine")
	tape := fs.Int("tape", 1, "tape length of each query")
	steps := fs.Int("oracle-steps", 64, "maximum simulator steps")
	bits := fs.Int("bits", 0, "training-bit prefix used for fitness; 0 uses the whole training split")
	mut := fs.Float64("mut", 0.1, "per-gene mutation probability")
	sigma := fs.Float64("sigma", 0.4, "mutation standard deviation")
	seed := fs.Int64("seed", 42, "seed")
	out := fs.String("out", "checkpoints", "directory for recur.json; empty skips the write")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	logFn := func(msg string) {
		fmt.Fprintln(os.Stdout, msg)
		_ = os.Stdout.Sync()
	}
	_, err := Evolve(EvolveConfig{
		DataPath: *data,
		Pop:      *pop,
		Gens:     *gens,
		Hidden:   *hidden,
		States:   *states,
		Tape:     *tape,
		MaxSteps: *steps,
		Bits:     *bits,
		MutRate:  *mut,
		MutSigma: *sigma,
		Seed:     *seed,
		OutDir:   *out,
		Log:      logFn,
	})
	return err
}
