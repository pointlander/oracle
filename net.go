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
)

// oracleNet is a directed network of bounded halting queries.
// The tape of each machine is the halt bits of its sources, in source order.
// A true halt bit is the character '1'. The simulator is step-bounded.
type oracleNet struct {
	machines []string
	sources  [][]int
}

func newOracleNet(rng *rand.Rand, nodes, degree, states int) *oracleNet {
	n := &oracleNet{
		machines: make([]string, nodes),
		sources:  make([][]int, nodes),
	}
	for i := 0; i < nodes; i++ {
		n.machines[i] = randomTotalMachine(rng, states)
		src := make([]int, degree)
		for k := range src {
			src[k] = rng.Intn(nodes)
		}
		n.sources[i] = src
	}
	return n
}

func tapeOf(sources []int, halt []bool) string {
	b := make([]byte, len(sources))
	for i, s := range sources {
		if halt[s] {
			b[i] = '1'
		} else {
			b[i] = '0'
		}
	}
	return string(b)
}

func (n *oracleNet) step(sim *UniversalBoundedSimulator, halt []bool) []bool {
	out := make([]bool, len(n.machines))
	tapes := make([]string, len(n.machines))
	for i := range n.machines {
		tapes[i] = tapeOf(n.sources[i], halt)
	}
	parallelN(len(n.machines), func(a, b int) {
		for i := a; i < b; i++ {
			out[i] = sim.Halts(n.machines[i], tapes[i])
		}
	})
	return out
}

func tapeFromMask(mask, k int) string {
	b := make([]byte, k)
	for c := 0; c < k; c++ {
		if (mask>>c)&1 == 1 {
			b[c] = '1'
		} else {
			b[c] = '0'
		}
	}
	return string(b)
}

// machineSurvey is the tape-dependence of total machines, ignoring the network.
// Immediate is the fraction whose transition on '0' goes to state 0. Those
// machines halt within two steps on a blank tape. CellInfluence[c] is the mean
// absolute change in halt rate when tape cell c flips. CellSigned[c] is the
// signed change, positive when a 1 raises the halt rate. Cell 0 is the symbol
// under the head at the start.
type machineSurvey struct {
	Immediate, ZeroHalt, RandHalt, Sensitive float64
	CellInfluence                            []float64
	CellSigned                               []float64
}

func surveyMachines(rng *rand.Rand, n, states, degree, maxSteps int) machineSurvey {
	sim := &UniversalBoundedSimulator{MaxSteps: maxSteps}
	masks := 1 << degree
	cell := make([]float64, degree)
	signed := make([]float64, degree)
	var zero, randSum, sens, immediate int
	for range n {
		m := randomTotalMachine(rng, states)
		if m[2:8] == "000000" {
			immediate++
		}
		haltOn := make([]int, degree)
		haltOff := make([]int, degree)
		halts := 0
		var first bool
		varies := false
		for mask := 0; mask < masks; mask++ {
			h := sim.Halts(m, tapeFromMask(mask, degree))
			if h {
				halts++
			}
			if mask == 0 {
				first = h
				if h {
					zero++
				}
			} else if h != first {
				varies = true
			}
			for c := 0; c < degree; c++ {
				if (mask>>c)&1 == 1 {
					if h {
						haltOn[c]++
					}
				} else if h {
					haltOff[c]++
				}
			}
		}
		if varies {
			sens++
		}
		randSum += halts
		half := float64(masks / 2)
		for c := 0; c < degree; c++ {
			p1 := float64(haltOn[c]) / half
			p0 := float64(haltOff[c]) / half
			delta := p1 - p0
			cell[c] += math.Abs(delta)
			signed[c] += delta
		}
	}
	nf := float64(n)
	for c := range cell {
		cell[c] /= nf
		signed[c] /= nf
	}
	return machineSurvey{
		Immediate:     float64(immediate) / nf,
		ZeroHalt:      float64(zero) / nf,
		RandHalt:      float64(randSum) / (nf * float64(masks)),
		Sensitive:     float64(sens) / nf,
		CellInfluence: cell,
		CellSigned:    signed,
	}
}

func meanHalt(h []bool) float64 {
	if len(h) == 0 {
		return 0
	}
	n := 0
	for _, v := range h {
		if v {
			n++
		}
	}
	return float64(n) / float64(len(h))
}

func hammingBool(a, b []bool) int {
	d := 0
	for i := range a {
		if a[i] != b[i] {
			d++
		}
	}
	return d
}

func flipBit(h []bool, i int) []bool {
	c := append([]bool(nil), h...)
	c[i] = !c[i]
	return c
}

func keyOf(h []bool) string {
	b := make([]byte, len(h))
	for i, v := range h {
		if v {
			b[i] = '1'
		} else {
			b[i] = '0'
		}
	}
	return string(b)
}

type cycleInfo struct {
	Closed bool
	Fixed  bool
	Length int
	Halt   float64
}

func findCycle(n *oracleNet, sim *UniversalBoundedSimulator, start []bool, iters int) cycleInfo {
	seen := make(map[string]int, iters)
	h := append([]bool(nil), start...)
	seen[keyOf(h)] = 0
	for t := 1; t <= iters; t++ {
		h = n.step(sim, h)
		key := keyOf(h)
		if prev, ok := seen[key]; ok {
			length := t - prev
			return cycleInfo{Closed: true, Fixed: length == 1, Length: length, Halt: meanHalt(h)}
		}
		seen[key] = t
	}
	return cycleInfo{Halt: meanHalt(h)}
}

// netConfig describes one ensemble of oracle networks.
type netConfig struct {
	Nodes    int
	Degree   int
	States   int
	MaxSteps int
	Depth    int
	Trials   int
	Iters    int
	Survey   int
	Seed     int64
}

func (c netConfig) validate() error {
	if c.Nodes < 1 || c.Degree < 1 || c.Degree > 8 {
		return fmt.Errorf("nodes must be positive and degree must be in 1..8")
	}
	if c.States < 1 || c.States > 63 || c.MaxSteps < 2 {
		return fmt.Errorf("bad machine shape")
	}
	if c.Depth < 1 || c.Trials < 1 || c.Iters < 1 || c.Survey < 1 {
		return fmt.Errorf("depth, trials, iters, and survey must be positive")
	}
	return nil
}

// netReport is the measured behavior of an ensemble.
// ZeroCycle and RandCycle are mean cycle lengths over the runs that closed.
// The matching halt fields are the mean halt fraction on those cycles.
type netReport struct {
	States, Nodes, Degree, MaxSteps, Trials, Depth int
	SurveyN                                        int
	Survey                                         machineSurvey
	LayerHalt                                      []float64
	LayerEcho                                      []float64
	ZeroClosed, ZeroFixed, ZeroCycle, ZeroHalt     float64
	RandClosed, RandFixed, RandCycle, RandHalt     float64
}

func analyzeNets(cfg netConfig) (netReport, error) {
	if err := cfg.validate(); err != nil {
		return netReport{}, err
	}
	// The survey and the ensemble each start from the same seed, so changing
	// the survey size does not change which networks are drawn.
	rng := rand.New(rand.NewSource(cfg.Seed))
	sim := &UniversalBoundedSimulator{MaxSteps: cfg.MaxSteps}
	rep := netReport{
		States:    cfg.States,
		Nodes:     cfg.Nodes,
		Degree:    cfg.Degree,
		MaxSteps:  cfg.MaxSteps,
		Trials:    cfg.Trials,
		Depth:     cfg.Depth,
		SurveyN:   cfg.Survey,
		Survey:    surveyMachines(rand.New(rand.NewSource(cfg.Seed)), cfg.Survey, cfg.States, cfg.Degree, cfg.MaxSteps),
		LayerHalt: make([]float64, cfg.Depth),
		LayerEcho: make([]float64, cfg.Depth),
	}
	var zeroClosed, zeroFixed, zeroCycle, zeroHalt float64
	var randClosed, randFixed, randCycle, randHalt float64
	for range cfg.Trials {
		net := newOracleNet(rng, cfg.Nodes, cfg.Degree, cfg.States)
		zero := make([]bool, cfg.Nodes)
		base := make([][]bool, cfg.Depth+1)
		base[0] = zero
		for d := 1; d <= cfg.Depth; d++ {
			base[d] = net.step(sim, base[d-1])
			rep.LayerHalt[d-1] += meanHalt(base[d])
		}
		echo := make([]float64, cfg.Depth)
		for i := 0; i < cfg.Nodes; i++ {
			h := flipBit(zero, i)
			for d := 1; d <= cfg.Depth; d++ {
				h = net.step(sim, h)
				echo[d-1] += float64(hammingBool(h, base[d])) / float64(cfg.Nodes)
			}
		}
		for d := range echo {
			rep.LayerEcho[d] += echo[d] / float64(cfg.Nodes)
		}
		z := findCycle(net, sim, zero, cfg.Iters)
		if z.Closed {
			zeroClosed++
			zeroCycle += float64(z.Length)
			zeroHalt += z.Halt
			if z.Fixed {
				zeroFixed++
			}
		}
		r := findCycle(net, sim, randBools(rng, cfg.Nodes), cfg.Iters)
		if r.Closed {
			randClosed++
			randCycle += float64(r.Length)
			randHalt += r.Halt
			if r.Fixed {
				randFixed++
			}
		}
	}
	tf := float64(cfg.Trials)
	for i := range rep.LayerHalt {
		rep.LayerHalt[i] /= tf
		rep.LayerEcho[i] /= tf
	}
	rep.ZeroClosed = zeroClosed / tf
	rep.ZeroFixed = zeroFixed / tf
	if zeroClosed > 0 {
		rep.ZeroCycle = zeroCycle / zeroClosed
		rep.ZeroHalt = zeroHalt / zeroClosed
	}
	rep.RandClosed = randClosed / tf
	rep.RandFixed = randFixed / tf
	if randClosed > 0 {
		rep.RandCycle = randCycle / randClosed
		rep.RandHalt = randHalt / randClosed
	}
	return rep, nil
}

func randBools(rng *rand.Rand, n int) []bool {
	v := make([]bool, n)
	for i := range v {
		v[i] = rng.Intn(2) == 1
	}
	return v
}

func printNetReport(log func(string), rep netReport) {
	log(fmt.Sprintf("oracle network: %d nodes, in-degree %d, %d-state machines, %d-step bound, %d draws",
		rep.Nodes, rep.Degree, rep.States, rep.MaxSteps, rep.Trials))
	log(fmt.Sprintf("  isolated machine, tape length %d, %d samples:", rep.Degree, rep.SurveyN))
	log(fmt.Sprintf("    read-0 transition is halt       %.3f", rep.Survey.Immediate))
	log(fmt.Sprintf("    halt rate on the zero tape      %.3f", rep.Survey.ZeroHalt))
	log(fmt.Sprintf("    halt rate on a uniform tape     %.3f", rep.Survey.RandHalt))
	log(fmt.Sprintf("    fraction sensitive to the tape  %.3f", rep.Survey.Sensitive))
	infl := "    cell influence"
	effect := "    cell effect of a 1"
	for c := range rep.Survey.CellInfluence {
		infl += fmt.Sprintf("  %d:%.3f", c, rep.Survey.CellInfluence[c])
		effect += fmt.Sprintf("  %d:%+.3f", c, rep.Survey.CellSigned[c])
	}
	log(infl)
	log(effect)
	log("  feedforward stack from the zero vector. echo is the fraction of halt bits a one-bit input flip still changes:")
	for d := range rep.LayerHalt {
		log(fmt.Sprintf("    layer %d  halt %.3f  echo %.4f", d+1, rep.LayerHalt[d], rep.LayerEcho[d]))
	}
	log("  cycle length and halt rate count only the runs that repeated inside the step limit:")
	log(fmt.Sprintf("  recurrent, start at zero: closed %.3f  fixed %.3f  mean cycle %.2f  halt %.3f",
		rep.ZeroClosed, rep.ZeroFixed, rep.ZeroCycle, rep.ZeroHalt))
	log(fmt.Sprintf("  recurrent, random start:  closed %.3f  fixed %.3f  mean cycle %.2f  halt %.3f",
		rep.RandClosed, rep.RandFixed, rep.RandCycle, rep.RandHalt))
}

func loopMachine(states int) string {
	return constMachine(states, 1)
}

func haltNowMachine(states int) string {
	return constMachine(states, 0)
}

func constMachine(states, next int) string {
	mp := make(map[int][2]Transition, states)
	for s := 1; s <= states; s++ {
		tr := Transition{WriteSymbol: '0', MoveDir: 1, NextState: next}
		mp[s] = [2]Transition{tr, tr}
	}
	return EncodeMachine(mp)
}

// copyHeadMachine halts exactly when the symbol under the head is '1'.
// On '0' the head steps left onto blank cells and stays in state 1, so a
// later '1' is never read. On '1' the next state is 0, which the following
// step reports. MaxSteps must be at least 2.
func copyHeadMachine() string {
	stay := Transition{WriteSymbol: '0', MoveDir: -1, NextState: 1}
	stop := Transition{WriteSymbol: '1', MoveDir: 1, NextState: 0}
	return EncodeMachine(map[int][2]Transition{1: {stay, stop}})
}

func netFromArgs(args []string) error {
	fs := flag.NewFlagSet("net", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: oracle net [flags]\n\nBuild random networks of bounded halting oracles.\nEach machine's tape is the halt bits of the machines it reads.\nThe report measures tape sensitivity, how a flipped halt bit spreads, and whether iteration reaches a fixed point.\n\n")
		fs.PrintDefaults()
	}
	nodes := fs.Int("nodes", 32, "machines in each network")
	degree := fs.Int("degree", 4, "tape cells, one halt bit each")
	steps := fs.Int("oracle-steps", 64, "maximum simulator steps")
	depth := fs.Int("depth", 8, "feedforward layers")
	trials := fs.Int("trials", 40, "networks per state count")
	iters := fs.Int("iters", 64, "recurrent steps before giving up on a cycle")
	survey := fs.Int("survey", 2048, "machines sampled for tape sensitivity")
	seed := fs.Int64("seed", 1, "seed")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	log := func(msg string) {
		fmt.Fprintln(os.Stdout, msg)
	}
	for _, states := range []int{1, 2} {
		rep, err := analyzeNets(netConfig{
			Nodes:    *nodes,
			Degree:   *degree,
			States:   states,
			MaxSteps: *steps,
			Depth:    *depth,
			Trials:   *trials,
			Iters:    *iters,
			Survey:   *survey,
			Seed:     *seed,
		})
		if err != nil {
			return err
		}
		printNetReport(log, rep)
		log("")
	}
	return nil
}
