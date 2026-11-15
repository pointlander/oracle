// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
)

// symbolSpec is a generative model with one bounded machine per alphabet symbol.
// Every machine reads the previous byte, most significant bit on cell 0.
// The symbol whose machine halts in the fewest steps is the next byte.
// Equal halt steps keep the smaller byte. A machine still running at MaxSteps
// does not generate a symbol.
type symbolSpec struct {
	Alphabet []byte
	States   int
	MaxSteps int
}

func (s symbolSpec) per() int {
	return s.States * 2 * 3
}

func (s symbolSpec) nGenes() int {
	return len(s.Alphabet) * s.per()
}

func (s symbolSpec) validate() error {
	if len(s.Alphabet) < 1 {
		return fmt.Errorf("alphabet must be non-empty")
	}
	if s.States < 1 || s.States > 63 {
		return fmt.Errorf("states must be in 1..63")
	}
	if s.MaxSteps < 2 {
		return fmt.Errorf("oracle steps must be at least 2")
	}
	return nil
}

func collectAlphabet(data []byte) []byte {
	var seen [256]bool
	out := make([]byte, 0)
	for _, b := range data {
		if !seen[b] {
			seen[b] = true
			out = append(out, b)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// byteTape is the previous byte, most significant bit first.
func byteTape(b byte) string {
	buf := make([]byte, 8)
	for i := 0; i < 8; i++ {
		if b&(1<<(7-i)) != 0 {
			buf[i] = '1'
		} else {
			buf[i] = '0'
		}
	}
	return string(buf)
}

func machineFromGenes(g []int, states int) string {
	mp := make(map[int][2]Transition, states)
	p := 0
	for s := 1; s <= states; s++ {
		var pair [2]Transition
		for sym := 0; sym < 2; sym++ {
			write := byte('0')
			if g[p] != 0 {
				write = '1'
			}
			move := -1
			if g[p+1] != 0 {
				move = 1
			}
			pair[sym] = Transition{WriteSymbol: write, MoveDir: move, NextState: g[p+2]}
			p += 3
		}
		mp[s] = pair
	}
	return EncodeMachine(mp)
}

func decodeSymbols(g []int, spec symbolSpec) []string {
	ensure(len(g) == spec.nGenes(), "genome %d != %d", len(g), spec.nGenes())
	out := make([]string, len(spec.Alphabet))
	per := spec.per()
	for i := range spec.Alphabet {
		out[i] = machineFromGenes(g[i*per:(i+1)*per], spec.States)
	}
	return out
}

// earliestSymbol runs one oracle per symbol. The winner is the machine that
// halts in the fewest steps. ok is false when every machine is still running
// at the step bound.
func earliestSymbol(sim *UniversalBoundedSimulator, machines []string, alphabet []byte, prev byte) (sym byte, ok bool) {
	tape := byteTape(prev)
	best := 0
	found := false
	for i, m := range machines {
		at := sim.HaltAt(m, tape)
		if at < 0 {
			continue
		}
		if !found || at < best {
			found = true
			best = at
			sym = alphabet[i]
		}
	}
	return sym, found
}

func symbolDetail(sim *UniversalBoundedSimulator, machines []string, alphabet, data []byte) (acc float64, correct, total, missed int) {
	if len(data) < 2 {
		return 0, 0, 0, 0
	}
	total = len(data) - 1
	for i := 0; i < total; i++ {
		sym, ok := earliestSymbol(sim, machines, alphabet, data[i])
		if !ok {
			missed++
			continue
		}
		if sym == data[i+1] {
			correct++
		}
	}
	return float64(correct) / float64(total), correct, total, missed
}

func continueSymbols(sim *UniversalBoundedSimulator, machines []string, alphabet, prompt []byte, count int, fallback byte) []byte {
	out := make([]byte, count)
	var prev byte
	if len(prompt) > 0 {
		prev = prompt[len(prompt)-1]
	}
	for i := 0; i < count; i++ {
		sym, ok := earliestSymbol(sim, machines, alphabet, prev)
		if !ok {
			sym = fallback
		}
		out[i] = sym
		prev = sym
	}
	return out
}

func majorityByte(data []byte) (byte, float64) {
	if len(data) < 2 {
		return 0, 0
	}
	var count [256]int
	for i := 1; i < len(data); i++ {
		count[data[i]]++
	}
	mode := 0
	for b := 1; b < 256; b++ {
		if count[b] > count[mode] {
			mode = b
		}
	}
	return byte(mode), float64(count[mode]) / float64(len(data)-1)
}

func applyUnigram(data []byte, mode byte) float64 {
	if len(data) < 2 {
		return 0
	}
	correct := 0
	for i := 1; i < len(data); i++ {
		if data[i] == mode {
			correct++
		}
	}
	return float64(correct) / float64(len(data)-1)
}

// bigramTable maps each previous byte to its most common successor in data.
// Ties keep the smaller byte. Missing contexts use fallback.
func bigramTable(data []byte, fallback byte) [256]byte {
	var cond [256][256]int
	for i := 0; i+1 < len(data); i++ {
		cond[data[i]][data[i+1]]++
	}
	var pred [256]byte
	for prev := 0; prev < 256; prev++ {
		best := int(fallback)
		bestN := cond[prev][fallback]
		for b := 0; b < 256; b++ {
			if cond[prev][b] > bestN || (cond[prev][b] == bestN && b < best) {
				best = b
				bestN = cond[prev][b]
			}
		}
		if bestN == 0 {
			pred[prev] = fallback
			continue
		}
		pred[prev] = byte(best)
	}
	return pred
}

func applyBigram(data []byte, pred [256]byte) float64 {
	if len(data) < 2 {
		return 0
	}
	correct := 0
	total := len(data) - 1
	for i := 0; i < total; i++ {
		if pred[data[i]] == data[i+1] {
			correct++
		}
	}
	return float64(correct) / float64(total)
}

func randomSymbolGene(rng *rand.Rand, spec symbolSpec) []int {
	g := make([]int, spec.nGenes())
	ntrans := spec.States * 2
	per := spec.per()
	for i := range spec.Alphabet {
		base := i * per
		for tr := 0; tr < ntrans; tr++ {
			g[base+tr*3] = rng.Intn(2)
			g[base+tr*3+1] = rng.Intn(2)
			g[base+tr*3+2] = rng.Intn(spec.States + 1)
		}
	}
	return g
}

func crossoverInt(rng *rand.Rand, a, b []int) []int {
	c := make([]int, len(a))
	for i := range a {
		if rng.Intn(2) == 0 {
			c[i] = a[i]
		} else {
			c[i] = b[i]
		}
	}
	return c
}

func mutateSymbolGene(rng *rand.Rand, g []int, spec symbolSpec, rate float64) {
	ntrans := spec.States * 2
	per := spec.per()
	for i := range spec.Alphabet {
		base := i * per
		for tr := 0; tr < ntrans; tr++ {
			if rng.Float64() < rate {
				g[base+tr*3] = rng.Intn(2)
			}
			if rng.Float64() < rate {
				g[base+tr*3+1] = rng.Intn(2)
			}
			if rng.Float64() < rate {
				g[base+tr*3+2] = rng.Intn(spec.States + 1)
			}
		}
	}
}

func breedSymbols(rng *rand.Rand, pop [][]int, fits []float64, spec symbolSpec, rate float64) [][]int {
	next := make([][]int, len(pop))
	bi := bestIndex(fits)
	next[0] = append([]int(nil), pop[bi]...)
	for i := 1; i < len(pop); i++ {
		a := tournament(rng, fits, 3)
		b := tournament(rng, fits, 3)
		child := crossoverInt(rng, pop[a], pop[b])
		mutateSymbolGene(rng, child, spec, rate)
		next[i] = child
	}
	return next
}

// NetLMConfig trains one halting oracle per symbol.
type NetLMConfig struct {
	Data     []byte
	DataPath string
	Pop      int
	Gens     int
	States   int
	MaxSteps int
	Bytes    int
	ValBytes int
	MutRate  float64
	GenBytes int
	Seed     int64
	OutDir   string
	Log      func(string)
}

// NetLMResult is the best symbol assignment and its next-byte accuracy.
type NetLMResult struct {
	TrainBytes  int     `json:"train_bytes"`
	ValBytes    int     `json:"val_bytes"`
	FitBytes    int     `json:"fit_bytes"`
	ScoreBytes  int     `json:"score_bytes"`
	Alphabet    int     `json:"alphabet"`
	Pop         int     `json:"pop"`
	Gens        int     `json:"gens"`
	States      int     `json:"states"`
	OracleSteps int     `json:"oracle_steps"`
	Genes       int     `json:"genes"`
	FitAcc      float64 `json:"fit_acc"`
	ValAcc      float64 `json:"val_acc"`
	FitUnigram  float64 `json:"fit_unigram"`
	ValUnigram  float64 `json:"val_unigram"`
	FitBigram   float64 `json:"fit_bigram"`
	ValBigram   float64 `json:"val_bigram"`
	FitMiss     float64 `json:"fit_miss"`
	ValMiss     float64 `json:"val_miss"`
	Mode        byte    `json:"mode"`
	Prompt      string  `json:"prompt"`
	Sample      string  `json:"sample"`
	Genome      []int   `json:"genome"`
}

func (cfg *NetLMConfig) logf(format string, args ...any) {
	if cfg.Log != nil {
		cfg.Log(fmt.Sprintf(format, args...))
	}
}

func (cfg *NetLMConfig) applyDefaults() {
	if cfg.Pop <= 0 {
		cfg.Pop = 24
	}
	if cfg.Gens <= 0 {
		cfg.Gens = 20
	}
	if cfg.States <= 0 {
		cfg.States = 1
	}
	if cfg.MaxSteps <= 0 {
		cfg.MaxSteps = 8
	}
	if cfg.Bytes <= 0 {
		cfg.Bytes = 2000
	}
	if cfg.ValBytes <= 0 {
		cfg.ValBytes = 800
	}
	if cfg.GenBytes < 0 {
		cfg.GenBytes = 0
	}
}

// NetLM searches one machine per symbol. Fitness is the fraction of next bytes
// equal to the symbol whose oracle halts first.
func NetLM(cfg NetLMConfig) (*NetLMResult, error) {
	cfg.applyDefaults()
	if cfg.Pop < 2 {
		return nil, fmt.Errorf("population must be at least 2")
	}
	if cfg.Gens < 1 {
		return nil, fmt.Errorf("generations must be positive")
	}
	if cfg.MutRate < 0 || cfg.MutRate > 1 {
		return nil, fmt.Errorf("mutation rate must be in 0..1")
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
	fit := train
	if cfg.Bytes < len(fit) {
		fit = fit[:cfg.Bytes]
	}
	if len(fit) < 2 {
		return nil, fmt.Errorf("need at least 2 training bytes, got %d", len(fit))
	}
	score := val
	if cfg.ValBytes < len(score) {
		score = score[:cfg.ValBytes]
	}
	spec := symbolSpec{Alphabet: collectAlphabet(fit), States: cfg.States, MaxSteps: cfg.MaxSteps}
	if err := spec.validate(); err != nil {
		return nil, err
	}
	sim := &UniversalBoundedSimulator{MaxSteps: spec.MaxSteps}
	rng := rand.New(rand.NewSource(cfg.Seed))
	pop := make([][]int, cfg.Pop)
	for i := range pop {
		pop[i] = randomSymbolGene(rng, spec)
	}
	mode, fitUni := majorityByte(fit)
	biTab := bigramTable(fit, mode)
	cfg.logf("symbol halting oracles")
	cfg.logf("  train %d bytes, val %d bytes, fitness %d bytes, val score %d bytes", len(train), len(val), len(fit), len(score))
	cfg.logf("  alphabet %d, %d-state machines, %d-step bound, genes %d", len(spec.Alphabet), spec.States, spec.MaxSteps, spec.nGenes())
	cfg.logf("  the previous byte is the tape; the earliest halt generates the next byte")
	cfg.logf("  unigram %.4f (%q)  bigram %.4f on the fitness bytes", fitUni, mode, applyBigram(fit, biTab))
	cfg.logf("  genetic search pop %d, gens %d, mutation %.2f, seed %d", cfg.Pop, cfg.Gens, cfg.MutRate, cfg.Seed)

	var best []int
	var bestFit float64
	for gen := 1; gen <= cfg.Gens; gen++ {
		fits := make([]float64, len(pop))
		parallelN(len(pop), func(a, b int) {
			for k := a; k < b; k++ {
				acc, _, _, _ := symbolDetail(sim, decodeSymbols(pop[k], spec), spec.Alphabet, fit)
				fits[k] = acc
			}
		})
		bi := bestIndex(fits)
		if best == nil || fits[bi] > bestFit {
			bestFit = fits[bi]
			best = append([]int(nil), pop[bi]...)
		}
		cfg.logf("gen %d/%d  best %.4f  mean %.4f", gen, cfg.Gens, fits[bi], meanFit(fits))
		if gen == cfg.Gens {
			break
		}
		pop = breedSymbols(rng, pop, fits, spec, cfg.MutRate)
	}
	machines := decodeSymbols(best, spec)
	_, _, fitTotal, fitMissN := symbolDetail(sim, machines, spec.Alphabet, fit)
	res := &NetLMResult{
		TrainBytes:  len(train),
		ValBytes:    len(val),
		FitBytes:    len(fit),
		ScoreBytes:  len(score),
		Alphabet:    len(spec.Alphabet),
		Pop:         cfg.Pop,
		Gens:        cfg.Gens,
		States:      spec.States,
		OracleSteps: spec.MaxSteps,
		Genes:       spec.nGenes(),
		FitAcc:      bestFit,
		FitUnigram:  fitUni,
		FitBigram:   applyBigram(fit, biTab),
		Mode:        mode,
		Genome:      best,
	}
	if fitTotal > 0 {
		res.FitMiss = float64(fitMissN) / float64(fitTotal)
	}
	if len(score) >= 2 {
		acc, _, total, miss := symbolDetail(sim, machines, spec.Alphabet, score)
		res.ValAcc = acc
		res.ValUnigram = applyUnigram(score, mode)
		res.ValBigram = applyBigram(score, biTab)
		if total > 0 {
			res.ValMiss = float64(miss) / float64(total)
		}
	}
	promptN := 32
	if promptN > len(fit) {
		promptN = len(fit)
	}
	res.Prompt = string(fit[:promptN])
	if cfg.GenBytes > 0 && promptN > 0 {
		res.Sample = string(continueSymbols(sim, machines, spec.Alphabet, fit[:promptN], cfg.GenBytes, mode))
	}
	cfg.logf("fit acc %.4f  unigram %.4f  bigram %.4f  no-halt %.4f", res.FitAcc, res.FitUnigram, res.FitBigram, res.FitMiss)
	cfg.logf("val acc %.4f  unigram %.4f  bigram %.4f  no-halt %.4f", res.ValAcc, res.ValUnigram, res.ValBigram, res.ValMiss)
	if res.Sample != "" {
		cfg.logf("prompt %s", strconv.Quote(res.Prompt))
		cfg.logf("sample %s", strconv.Quote(res.Sample))
	}
	if cfg.OutDir != "" {
		path := filepath.Join(cfg.OutDir, "netlm.json")
		if err := writeResult(path, res); err != nil {
			return nil, err
		}
		cfg.logf("wrote %s", path)
	}
	return res, nil
}

func netLMFromArgs(args []string) error {
	fs := flag.NewFlagSet("netlm", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: oracle netlm [flags]\n\nTrain one halting oracle per symbol.\nEach machine reads the previous byte. The symbol whose machine halts in the fewest steps is generated.\nA genetic search picks the machines. The simulator is step-bounded.\n\n")
		fs.PrintDefaults()
	}
	data := fs.String("data", "100.txt.utf-8", "UTF-8 text file")
	pop := fs.Int("pop", 24, "population size")
	gens := fs.Int("gens", 20, "generations")
	states := fs.Int("states", 1, "states in each symbol machine")
	steps := fs.Int("oracle-steps", 8, "maximum simulator steps")
	nbytes := fs.Int("bytes", 2000, "training-byte prefix used for fitness")
	valBytes := fs.Int("val-bytes", 800, "validation-byte prefix used for the report")
	mut := fs.Float64("mut", 0.02, "per-gene probability of resampling that gene")
	genBytes := fs.Int("gen-bytes", 80, "closed-loop bytes to print after the prompt")
	seed := fs.Int64("seed", 1, "seed")
	out := fs.String("out", "checkpoints", "directory for netlm.json; empty skips the write")
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
	_, err := NetLM(NetLMConfig{
		DataPath: *data,
		Pop:      *pop,
		Gens:     *gens,
		States:   *states,
		MaxSteps: *steps,
		Bytes:    *nbytes,
		ValBytes: *valBytes,
		MutRate:  *mut,
		GenBytes: *genBytes,
		Seed:     *seed,
		OutDir:   *out,
		Log:      logFn,
	})
	return err
}
