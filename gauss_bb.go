// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

// bbBound is the step limit used to read a run back. BB(2) halts at step 6,
// inside this bound, so that observation is the halt itself. A run still
// going at the bound is not a halting decision.
const bbBound = 32

// bbHaltStep and bbHaltOnes are the busy beaver's halting observation.
// The backward pass starts from this observation. It does not supervise
// the four rules one by one.
const (
	bbHaltStep = 6
	bbHaltOnes = 4
)

// bbRule is one transition of the 2-state busy beaver.
// move is -1 or +1. next 0 is halt.
type bbRule struct {
	write int
	move  int
	next  int
}

// bb2 is the 2-state 2-symbol busy beaver. The transition into halt is
// state 2 reading 1. Which initial state reaches that halt on a blank tape
// is what backpropagation solves for.
func bb2() [3][2]bbRule {
	var t [3][2]bbRule
	t[1][0] = bbRule{write: 1, move: 1, next: 2}
	t[1][1] = bbRule{write: 1, move: -1, next: 2}
	t[2][0] = bbRule{write: 1, move: -1, next: 1}
	t[2][1] = bbRule{write: 1, move: 1, next: 0}
	return t
}

func bb2Transitions() map[int][2]Transition {
	rules := bb2()
	m := make(map[int][2]Transition, 2)
	for s := 1; s <= 2; s++ {
		var pair [2]Transition
		for sym := 0; sym < 2; sym++ {
			w := byte('0')
			if rules[s][sym].write == 1 {
				w = '1'
			}
			pair[sym] = Transition{
				WriteSymbol: w,
				MoveDir:     rules[s][sym].move,
				NextState:   rules[s][sym].next,
			}
		}
		m[s] = pair
	}
	return m
}

// bbScore runs table the way HaltAt does, starting at start. A blank cell
// is 0. ones is the number of cells holding 1 when the run stops. haltStep
// is -1 when the bound is reached with the machine still running. State 0
// is observed on the step after the transition into halt.
func bbScore(table map[int][2]Transition, maxSteps, start int) (states []int, haltStep, ones int) {
	tape := make(map[int]byte)
	head := 0
	state := start
	haltStep = -1
	for step := 0; step < maxSteps; step++ {
		states = append(states, state)
		if state == 0 {
			haltStep = step
			break
		}
		pair, ok := table[state]
		if !ok {
			haltStep = step
			break
		}
		cur, exists := tape[head]
		if !exists {
			cur = '0'
		}
		rule := pair[1]
		if cur == '0' {
			rule = pair[0]
		}
		tape[head] = rule.WriteSymbol
		head += rule.MoveDir
		state = rule.NextState
	}
	for _, b := range tape {
		if b == '1' {
			ones++
		}
	}
	return states, haltStep, ones
}

// bbHaltCost is the squared distance from the busy beaver's halt. The
// final state wants to be 0, the halt wants to be observed at step 6, and
// the tape wants to hold four 1s. A run that is still going at the bound
// pays for the full bound.
func bbHaltCost(states []int, haltStep, ones int) float64 {
	end := -1.0
	if n := len(states); n > 0 {
		end = float64(states[n-1])
	}
	step := float64(haltStep)
	if haltStep < 0 {
		step = float64(bbBound)
	}
	dState := end
	dStep := step - float64(bbHaltStep)
	dOnes := float64(ones - bbHaltOnes)
	return dState*dState + dStep*dStep + dOnes*dOnes
}

// bbBranch is the forward run from one initial state on an all-zero tape.
type bbBranch struct {
	start    int
	states   []int
	haltStep int
	ones     int
	cost     float64
}

func bbBranchFrom(start int) bbBranch {
	states, step, ones := bbScore(bb2Transitions(), bbBound, start)
	return bbBranch{
		start:    start,
		states:   states,
		haltStep: step,
		ones:     ones,
		cost:     bbHaltCost(states, step, ones),
	}
}

// bbBackprop is the expected halt cost of an initial state on a blank tape,
// and the partial of that cost with respect to the logit of starting in
// state 1. The cost is computed at the halt of each branch. It is carried
// back to the logit by the chain rule through the branch probability.
// State 2 also reaches halt, but earlier and with two 1s, so its cost is
// higher and the partial points toward state 1.
func bbBackprop(logit float64) (loss, dlogit float64) {
	p := logistic(logit)
	c1 := bbBranchFrom(1).cost
	c2 := bbBranchFrom(2).cost
	loss = p*c1 + (1-p)*c2
	dlogit = (c1 - c2) * p * (1 - p)
	return loss, dlogit
}

type bbConfig struct {
	States int
	Iters  int
	Log    func(string)
}

func (cfg bbConfig) logf(format string, args ...any) {
	if cfg.Log != nil {
		cfg.Log(fmt.Sprintf(format, args...))
	}
}

// learnBB starts at the busy beaver's halting state and follows the
// partial of that halt cost back to the initial state. The tape at that
// initial state is all zeros. The logit starts at 0, an even chance of
// state 1 or state 2. A probability of at least 0.9 selects the state,
// and that state is run forward on the blank tape.
func learnBB(cfg bbConfig) error {
	if cfg.States != 2 {
		return fmt.Errorf("only BB(2) is learned")
	}
	if cfg.Iters <= 0 {
		cfg.Iters = 32
	}
	from1 := bbBranchFrom(1)
	from2 := bbBranchFrom(2)
	cfg.logf("gaussian BB(2)")
	cfg.logf("  backpropagation starts at the halting state")
	cfg.logf("  the initial tape is all zeros")
	cfg.logf("  the halt observation is state 0 at step %d with %d ones", bbHaltStep, bbHaltOnes)
	cfg.logf("  branch state 1  halt step %d  ones %d  cost %.4f", from1.haltStep, from1.ones, from1.cost)
	cfg.logf("  branch state 2  halt step %d  ones %d  cost %.4f", from2.haltStep, from2.ones, from2.cost)
	logit := 0.0
	var loss float64
	matched := -1
	state := 0
	for iter := 0; iter <= cfg.Iters; iter++ {
		var dlogit float64
		loss, dlogit = bbBackprop(logit)
		p := logistic(logit)
		state = 2
		if logit >= 0 {
			state = 1
		}
		sure := p >= 0.9 || p <= 0.1
		if iter == 0 || iter == cfg.Iters || iter%4 == 0 || (sure && iter > 0) {
			cfg.logf("iter %d  loss %.6f  p(state 1) %.2f", iter, loss, p)
		}
		if sure && iter > 0 {
			matched = iter
			break
		}
		if iter == cfg.Iters {
			break
		}
		logit -= dlogit
		if logit > 20 {
			logit = 20
		}
		if logit < -20 {
			logit = -20
		}
	}
	cfg.logf("initial state %d", state)
	cfg.logf("tape all zeros")
	states, haltStep, ones := bbScore(bb2Transitions(), bbBound, state)
	cfg.logf("states %s", formatStates(states))
	if haltStep < 0 {
		cfg.logf("still running at step %d", bbBound)
	} else {
		cfg.logf("halt step %d", haltStep)
	}
	cfg.logf("ones %d", ones)
	if matched < 0 {
		return fmt.Errorf("BB(2) did not find an initial state after %d steps", cfg.Iters)
	}
	if haltStep != bbHaltStep || ones != bbHaltOnes || len(states) == 0 || states[len(states)-1] != 0 {
		return fmt.Errorf("initial state %d halted at step %d with %d ones", state, haltStep, ones)
	}
	return nil
}
