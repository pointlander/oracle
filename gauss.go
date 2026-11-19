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
	"sort"
)

// gaussTransition is one total-machine rule whose write weight is a constant.
// The weight is drawn once from N(0,1) when the machine is built.
type gaussTransition struct {
	Write float64
	Move  int
	Next  int
}

// gaussMachine is a 2-symbol machine. state[s][0] is the rule for symbol 0.
// State 0 is halt and is not stored. Next is in 0..states.
type gaussMachine struct {
	state [][2]gaussTransition
}

// gaussCell is the running mean and root-mean-square deviation of the weights
// written into one cell. n is the number of those weights. A cell with n == 0
// is a blank and is sampled from N(0,1); that prior is not one of the weights.
type gaussCell struct {
	n    int
	mean float64
	m2   float64
}

func (c *gaussCell) dev() float64 {
	if c == nil || c.n <= 1 || c.m2 <= 0 {
		if c == nil || c.n == 0 {
			return 1
		}
		return 0
	}
	return math.Sqrt(c.m2 / float64(c.n))
}

// observe folds one write weight into the cell. Equal weights keep the
// deviation at 0. The first weight sets the mean and leaves a point mass.
func (c *gaussCell) observe(x float64) {
	c.n++
	delta := x - c.mean
	c.mean += delta / float64(c.n)
	c.m2 += delta * (x - c.mean)
}

// sampleGauss draws the cell. A positive draw is symbol 1. Zero and every
// negative draw are symbol 0.
func sampleGauss(c *gaussCell, rng *rand.Rand) float64 {
	if c == nil || c.n == 0 {
		return rng.NormFloat64()
	}
	d := c.dev()
	if d == 0 {
		return c.mean
	}
	return c.mean + d*rng.NormFloat64()
}

func gaussBit(x float64) int {
	if x > 0 {
		return 1
	}
	return 0
}

func randomGaussMachine(rng *rand.Rand, states int) *gaussMachine {
	m := &gaussMachine{state: make([][2]gaussTransition, states+1)}
	for s := 1; s <= states; s++ {
		for sym := 0; sym < 2; sym++ {
			move := -1
			if rng.Intn(2) == 1 {
				move = 1
			}
			m.state[s][sym] = gaussTransition{
				Write: rng.NormFloat64(),
				Move:  move,
				Next:  rng.Intn(states + 1),
			}
		}
	}
	return m
}

// signTwin is the binary machine with the same moves and next states.
// Its write bit is the sign of the gaussian weight. A non-positive weight
// writes '0'.
func (m *gaussMachine) signTwin() string {
	mp := make(map[int][2]Transition, len(m.state)-1)
	for s := 1; s < len(m.state); s++ {
		var pair [2]Transition
		for sym := 0; sym < 2; sym++ {
			write := byte('0')
			if m.state[s][sym].Write > 0 {
				write = '1'
			}
			pair[sym] = Transition{
				WriteSymbol: write,
				MoveDir:     m.state[s][sym].Move,
				NextState:   m.state[s][sym].Next,
			}
		}
		mp[s] = pair
	}
	return EncodeMachine(mp)
}

type gaussRun struct {
	Halted     bool
	HaltStep   int
	Reads      int
	Reads1     int
	FirstRead  int
	Rereads    int
	Flips      int
	Visited    int
	Written    int
	Point      int
	Multi      int
	MultiPoint int
	AbsMeanSum float64
	MultiStd   float64
	cells      map[int]*gaussCell
}

// runGauss executes m for at most maxSteps. init may place point-mass input
// cells before the first read. State 0 is observed at the start of a step, so
// a transition into 0 is reported on the following step. A machine still
// running at maxSteps returns Halted false.
func runGauss(m *gaussMachine, maxSteps int, init func(map[int]*gaussCell), rng *rand.Rand) gaussRun {
	cells := make(map[int]*gaussCell)
	if init != nil {
		init(cells)
	}
	run := gaussRun{HaltStep: -1, cells: cells}
	last := make(map[int]int)
	seen := make(map[int]bool)
	head := 0
	state := 1
	for step := 0; step < maxSteps; step++ {
		if state == 0 || state >= len(m.state) {
			run.Halted = true
			run.HaltStep = step
			break
		}
		sample := sampleGauss(cells[head], rng)
		bit := gaussBit(sample)
		if seen[head] {
			run.Rereads++
			if last[head] != bit {
				run.Flips++
			}
		} else {
			seen[head] = true
		}
		last[head] = bit
		if run.Reads == 0 {
			run.FirstRead = bit
		}
		run.Reads++
		if bit == 1 {
			run.Reads1++
		}
		rule := m.state[state][bit]
		cell := cells[head]
		if cell == nil {
			cell = &gaussCell{}
			cells[head] = cell
		}
		cell.observe(rule.Write)
		head += rule.Move
		state = rule.Next
	}
	run.Visited = len(seen)
	idx := make([]int, 0, len(cells))
	for k, c := range cells {
		if c.n > 0 {
			idx = append(idx, k)
		}
	}
	sort.Ints(idx)
	for _, k := range idx {
		c := cells[k]
		run.Written++
		run.AbsMeanSum += math.Abs(c.mean)
		d := c.dev()
		if d == 0 {
			run.Point++
		}
		if c.n >= 2 {
			run.Multi++
			run.MultiStd += d
			if d == 0 {
				run.MultiPoint++
			}
		}
	}
	return run
}

func pointMass(v float64) *gaussCell {
	return &gaussCell{n: 1, mean: v}
}

func fillPoint(cells map[int]*gaussCell, n int, v float64) {
	for i := 0; i < n; i++ {
		cells[i] = pointMass(v)
	}
}

// gaussReport is one survey of random gaussian-tape machines.
type gaussReport struct {
	States   int
	MaxSteps int
	Survey   int
	Repeats  int
	Input    int
	Seed     int64

	BlankHalt    float64
	AlwaysHalt   float64
	NeverHalt    float64
	Unstable     float64
	MeanHaltStep float64
	TwinZeroHalt float64
	TwinAgree    float64
	Read1        float64
	RereadPerRun float64
	FlipRate     float64
	Cells        float64
	PointFrac    float64
	MultiFrac    float64
	MultiSame    float64
	MultiStd     float64
	AbsMean      float64
	NegHalt      float64
	PosHalt      float64
	BlankFirst1  float64
	NegFirst1    float64
	PosFirst1    float64
}

type gaussConfig struct {
	States   int
	MaxSteps int
	Survey   int
	Repeats  int
	Input    int
	Seed     int64
}

func (cfg gaussConfig) validate() error {
	if cfg.States < 1 || cfg.States > 2 {
		return fmt.Errorf("states must be 1 or 2")
	}
	if cfg.MaxSteps < 2 {
		return fmt.Errorf("oracle steps must be at least 2")
	}
	if cfg.Survey < 1 {
		return fmt.Errorf("survey must be positive")
	}
	if cfg.Repeats < 1 {
		return fmt.Errorf("repeats must be positive")
	}
	if cfg.Input < 1 {
		return fmt.Errorf("input length must be positive")
	}
	return nil
}

func analyzeGauss(cfg gaussConfig) (gaussReport, error) {
	if err := cfg.validate(); err != nil {
		return gaussReport{}, err
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	sim := &UniversalBoundedSimulator{MaxSteps: cfg.MaxSteps}
	zeros := make([]byte, cfg.Input)
	for i := range zeros {
		zeros[i] = '0'
	}
	zeroTape := string(zeros)
	rep := gaussReport{
		States: cfg.States, MaxSteps: cfg.MaxSteps, Survey: cfg.Survey,
		Repeats: cfg.Repeats, Input: cfg.Input, Seed: cfg.Seed,
	}
	var (
		haltEvents, haltSteps int
		always, never         int
		reads, reads1         int
		rereads, flips        int
		visited               int
		written, point        int
		multi, multiPoint     int
		multiStd, absMean     float64
		twinHalt, twinMatch   int
		negHalt, posHalt      int
		blank1, neg1, pos1    int
	)
	for range cfg.Survey {
		m := randomGaussMachine(rng, cfg.States)
		halts := 0
		for r := 0; r < cfg.Repeats; r++ {
			run := runGauss(m, cfg.MaxSteps, nil, rng)
			if run.Halted {
				halts++
				haltEvents++
				haltSteps += run.HaltStep
			}
			if r == 0 {
				reads += run.Reads
				reads1 += run.Reads1
				rereads += run.Rereads
				flips += run.Flips
				visited += run.Visited
				written += run.Written
				point += run.Point
				multi += run.Multi
				multiPoint += run.MultiPoint
				multiStd += run.MultiStd
				absMean += run.AbsMeanSum
				blank1 += run.FirstRead
				twinH := sim.Halts(m.signTwin(), zeroTape)
				if twinH {
					twinHalt++
				}
				if twinH == run.Halted {
					twinMatch++
				}
			}
		}
		if halts == cfg.Repeats {
			always++
		} else if halts == 0 {
			never++
		}
		neg := runGauss(m, cfg.MaxSteps, func(cells map[int]*gaussCell) {
			fillPoint(cells, cfg.Input, -1)
		}, rng)
		pos := runGauss(m, cfg.MaxSteps, func(cells map[int]*gaussCell) {
			fillPoint(cells, cfg.Input, 1)
		}, rng)
		if neg.Halted {
			negHalt++
		}
		if pos.Halted {
			posHalt++
		}
		neg1 += neg.FirstRead
		pos1 += pos.FirstRead
	}
	n := float64(cfg.Survey)
	rep.BlankHalt = float64(haltEvents) / (n * float64(cfg.Repeats))
	rep.AlwaysHalt = float64(always) / n
	rep.NeverHalt = float64(never) / n
	rep.Unstable = 1 - rep.AlwaysHalt - rep.NeverHalt
	if haltEvents > 0 {
		rep.MeanHaltStep = float64(haltSteps) / float64(haltEvents)
	}
	rep.TwinZeroHalt = float64(twinHalt) / n
	rep.TwinAgree = float64(twinMatch) / n
	if reads > 0 {
		rep.Read1 = float64(reads1) / float64(reads)
	}
	rep.RereadPerRun = float64(rereads) / n
	if rereads > 0 {
		rep.FlipRate = float64(flips) / float64(rereads)
	}
	rep.Cells = float64(visited) / n
	if written > 0 {
		rep.PointFrac = float64(point) / float64(written)
		rep.MultiFrac = float64(multi) / float64(written)
		rep.AbsMean = absMean / float64(written)
	}
	if multi > 0 {
		rep.MultiSame = float64(multiPoint) / float64(multi)
		rep.MultiStd = multiStd / float64(multi)
	}
	rep.NegHalt = float64(negHalt) / n
	rep.PosHalt = float64(posHalt) / n
	rep.BlankFirst1 = float64(blank1) / n
	rep.NegFirst1 = float64(neg1) / n
	rep.PosFirst1 = float64(pos1) / n
	return rep, nil
}

func printGaussReport(log func(string), rep gaussReport) {
	log(fmt.Sprintf("gaussian tape: %d-state machines, %d-step bound, %d draws, %d repeats, input %d, seed %d",
		rep.States, rep.MaxSteps, rep.Survey, rep.Repeats, rep.Input, rep.Seed))
	log("  each write weight is one N(0,1) sample; a blank cell is sampled from N(0,1)")
	log("  a sample above 0 is symbol 1; every other sample is symbol 0")
	log("  a write folds that weight into the cell mean and root-mean-square deviation")
	log(fmt.Sprintf("  halt rate on a blank tape                  %.4f", rep.BlankHalt))
	log(fmt.Sprintf("  halt on every repeat                       %.4f", rep.AlwaysHalt))
	log(fmt.Sprintf("  halt on no repeat                          %.4f", rep.NeverHalt))
	log(fmt.Sprintf("  halt bit changes across repeats            %.4f", rep.Unstable))
	log(fmt.Sprintf("  mean halt step, counting halting runs      %.2f", rep.MeanHaltStep))
	log(fmt.Sprintf("  sign-twin halt rate on a zero tape         %.4f", rep.TwinZeroHalt))
	log(fmt.Sprintf("  agreement with that twin's first run       %.4f", rep.TwinAgree))
	log(fmt.Sprintf("  fraction of reads that are 1               %.4f", rep.Read1))
	log(fmt.Sprintf("  first blank read is 1                      %.4f", rep.BlankFirst1))
	log(fmt.Sprintf("  re-reads per run                           %.2f", rep.RereadPerRun))
	log(fmt.Sprintf("  re-reads whose bit differs                 %.4f", rep.FlipRate))
	log(fmt.Sprintf("  cells visited per run                      %.2f", rep.Cells))
	log(fmt.Sprintf("  written cells with deviation 0             %.4f", rep.PointFrac))
	log(fmt.Sprintf("  written cells written more than once       %.4f", rep.MultiFrac))
	log(fmt.Sprintf("  of those, deviation still 0                %.4f", rep.MultiSame))
	log(fmt.Sprintf("  mean deviation after repeated writes       %.4f", rep.MultiStd))
	log(fmt.Sprintf("  mean |mean| of written cells               %.4f", rep.AbsMean))
	log(fmt.Sprintf("  halt rate on %d cells fixed at -1           %.4f", rep.Input, rep.NegHalt))
	log(fmt.Sprintf("  halt rate on %d cells fixed at +1           %.4f", rep.Input, rep.PosHalt))
	log(fmt.Sprintf("  first read is 1 on a -1 cell               %.4f", rep.NegFirst1))
	log(fmt.Sprintf("  first read is 1 on a +1 cell               %.4f", rep.PosFirst1))
}

func gaussFromArgs(args []string) error {
	fs := flag.NewFlagSet("gauss", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: oracle gauss [flags]\n\nRun 2-symbol machines whose tape cells are Gaussians.\nA sample above 0 is symbol 1. Each write updates that cell's running mean and deviation.\nThe simulator is step-bounded.\n\nWith -text, train a 256-symbol machine so its tape spells that string.\nEach state has two write weights and two move logits, one for a 0 on the tape and one for any other byte.\nA blank cell or a zero byte reads 0. The loss follows the rule for the symbol that is there.\nAn empty -moves trains the mode path, and the other branch only while that move is still undecided.\nThe whole string is one machine.\n-size is the tape length as a fraction of those symbols. The default is 0.8.\nWhen the tape is shorter than the symbols, the head wraps. The cell mean selects the rule, and the symbol is the weight that rule wrote. Move logits follow the cell-mean loss.\nA non-empty -moves is a fixed string of L and R for both reads.\nThe loss is the cross-entropy of a Gaussian over the 256 byte codes.\nEach step follows the partial of that loss through the write weights and, when moves are learned, the move logits.\nThe move after the last write has partial 0.\nAfter training, the discrete machine writes the learned character.\nOn a 0 it writes that character, moves as learned, and goes to the next state. The last state halts.\nOn a 1 it writes its own character, moves as learned, and goes to the previous state. The first state halts. A one-state machine stays on a 1.\nThe machine is then run, and the states it visits are printed as those characters.\nThe learned symbol bytes are recorded in a buffer and read back.\n\nWith -iris, the machine classifies the flowers in that zip or data file.\nThe head scans the four measurements. Each state writes its weight times that measurement onto an output cell.\nThe last write is a bias. The output is the text machine's Gaussian softmax over the 256 byte codes.\nEach species is a byte around 128, and the class is the nearest of those bytes.\nThe write step is half the temperature squared.\n\nWith -lm, the machine learns a byte language model on that UTF-8 file.\nThe head reads the previous -order bytes. Each context's weight is trained on its most common next byte.\nThe first 90%% of the bytes are training. The last 10%% are validation.\n\nWith -bb 2, backpropagation starts at the busy beaver's halting state.\nGradient descent finds the initial state on an all-zero tape that reaches that halt.\nThe state it selects is run forward on the blank tape.\nThe halting tape is a Gaussian cell per written square.\nGradient descent learns the probability that a sample from that cell is above 0.\n\nWith -bb 5, backpropagation starts at the 5-state busy beaver's halting state.\nThe forward run records the state chain and the halt tape.\nGradient descent fits a Gaussian softmax for each state and a unit-width Gaussian for each visited cell.\nThe probability is the mass above 0. A mean above 0 is guessed as 1.\nThose bits are run backward along the learned states.\nA blank tape, the head on cell 0, and state 1 means that halt is reached from a blank tape.\nA run still going at the step bound is not a halt.\n\nWith -bb 5 -no-halt, that machine's halt transition stays in state 5.\nNo rule enters state 0, so the machine does not halt.\nThe forward run stops at the step bound. Gradient descent fits the state chain and the tape there.\nRunning backward from the state at the bound rewinds the run to a blank tape.\nRunning backward from state 0 does not.\n\n")
		fs.PrintDefaults()
	}
	states := fs.Int("states", 0, "states in each machine; 0 runs 1 and 2")
	steps := fs.Int("oracle-steps", 64, "maximum simulator steps")
	survey := fs.Int("survey", 2048, "machines to draw")
	repeats := fs.Int("repeats", 16, "blank-tape repeats of each machine")
	input := fs.Int("input", 8, "cells in the fixed +1 and -1 tapes")
	seed := fs.Int64("seed", 1, "seed")
	text := fs.String("text", "", "train write weights so the tape spells this text")
	moves := fs.String("moves", "", "fixed L and R moves when -text is set; empty learns a move for each read")
	iters := fs.Int("iters", 32, "gradient steps when -text or -iris is set")
	lr := fs.Float64("lr", 0, "write learning rate when -text is set; 0 uses the temperature squared")
	moveLR := fs.Float64("move-lr", 0, "move-logit learning rate when -text learns moves; 0 uses 1")
	size := fs.Float64("size", 0.8, "tape length as a fraction of the symbols being learned")
	temp := fs.Float64("temp", 8, "symbol-softmax temperature when -text or -iris is set")
	irisPath := fs.String("iris", "", "classify the iris zip or data file at this path")
	lmPath := fs.String("lm", "", "train a byte language model on this UTF-8 text")
	order := fs.Int("order", 2, "bytes of context when -lm is set, from 0 to 8")
	bb := fs.Int("bb", 0, "learn BB(2) or BB(5) from the halting state; 2 or 5")
	noHalt := fs.Bool("no-halt", false, "with -bb 5, drop the transition into halt")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	log := func(msg string) {
		fmt.Fprintln(os.Stdout, msg)
	}
	mode := 0
	if *text != "" {
		mode++
	}
	if *irisPath != "" {
		mode++
	}
	if *lmPath != "" {
		mode++
	}
	if *bb != 0 {
		mode++
	}
	if mode > 1 {
		return fmt.Errorf("-text, -iris, -lm, and -bb cannot be combined")
	}
	if *noHalt && *bb != 5 {
		return fmt.Errorf("-no-halt requires -bb 5")
	}
	if *irisPath != "" {
		return learnIris(irisConfig{Path: *irisPath, Iters: *iters, Temp: *temp, Seed: *seed, Log: log})
	}
	if *lmPath != "" {
		return learnGaussLM(lmConfig{Path: *lmPath, Order: *order, Iters: *iters, Temp: *temp, Seed: *seed, Log: log})
	}
	if *bb != 0 {
		return learnBB(bbConfig{States: *bb, Iters: *iters, NoHalt: *noHalt, Log: log})
	}
	if *text != "" {
		return learnGaussText(gaussLearnConfig{
			Text: *text, Moves: *moves, Iters: *iters, LR: *lr, MoveLR: *moveLR, Size: *size, Temp: *temp, Log: log,
		})
	}
	if *states < 0 || *states > 2 {
		return fmt.Errorf("states must be 1 or 2, or 0 to run both")
	}
	list := []int{*states}
	if *states == 0 {
		list = []int{1, 2}
	}
	for _, st := range list {
		rep, err := analyzeGauss(gaussConfig{
			States:   st,
			MaxSteps: *steps,
			Survey:   *survey,
			Repeats:  *repeats,
			Input:    *input,
			Seed:     *seed,
		})
		if err != nil {
			return err
		}
		printGaussReport(log, rep)
		log("")
	}
	return nil
}
