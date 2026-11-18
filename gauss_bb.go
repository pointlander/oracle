// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
)

// bbTau is the width of the write and next-state softmaxes. The write
// symbols sit at 0 and 1. The next state sits at 0, 1, or 2. A width of
// 0.5 keeps those symbols apart. State 0 is halt.
const bbTau = 0.5

// bbBound is the step limit used to read the learned machine back. BB(2)
// halts at step 6, inside this bound, so that observation is the halt
// itself. A run still going at the bound is not a halting decision.
const bbBound = 32

// bbRule is one target transition of the 2-state busy beaver.
// move is -1 or +1. next 0 is halt.
type bbRule struct {
	write int
	move  int
	next  int
}

// bbParams holds the learned transition. write and next are Gaussian means.
// move is a logit. State 0 is halt and is not stored.
type bbParams struct {
	write [3][2]float64
	move  [3][2]float64
	next  [3][2]float64
}

// bb2 is the 2-state 2-symbol busy beaver. From a blank tape it writes four
// 1s and enters state 0 on the sixth transition. That transition is state 2
// reading 1. Halt is observed on the following step.
func bb2() [3][2]bbRule {
	var t [3][2]bbRule
	t[1][0] = bbRule{write: 1, move: 1, next: 2}
	t[1][1] = bbRule{write: 1, move: -1, next: 2}
	t[2][0] = bbRule{write: 1, move: -1, next: 1}
	t[2][1] = bbRule{write: 1, move: 1, next: 0}
	return t
}

// moveNLL is the logistic cross-entropy of a left or right step.
// dir > 0 is right. The partial is with respect to the logit.
func moveNLL(logit float64, dir int) (nll, dlogit float64) {
	p := logistic(logit)
	y := 0.0
	if dir > 0 {
		y = 1
	}
	a := logit
	if a < 0 {
		a = -a
	}
	nll = math.Log1p(math.Exp(-a))
	if logit > 0 {
		nll += logit
	}
	nll -= y * logit
	dlogit = p - y
	return nll, dlogit
}

// bbPartial is the mean cross-entropy of the four busy-beaver rules and the
// partial of that mean. Each rule is one term, so the halting rule is not
// drowned by the steps that repeat another rule. The write and next partials
// are classLoss derivatives. Divide by τ^2 on the way out of those.
func bbPartial(p bbParams) (mean float64, dw, dm, dn [3][2]float64) {
	rules := bb2()
	n := 0.0
	for s := 1; s <= 2; s++ {
		for sym := 0; sym < 2; sym++ {
			rule := rules[s][sym]
			nw, gw := classLoss(p.write[s][sym], bbTau, rule.write, 2)
			nm, gm := moveNLL(p.move[s][sym], rule.move)
			nn, gn := classLoss(p.next[s][sym], bbTau, rule.next, 3)
			mean += nw + nm + nn
			dw[s][sym] = gw
			dm[s][sym] = gm
			dn[s][sym] = gn
			n++
		}
	}
	mean /= n
	for s := 1; s <= 2; s++ {
		for sym := 0; sym < 2; sym++ {
			dw[s][sym] /= n
			dm[s][sym] /= n
			dn[s][sym] /= n
		}
	}
	return mean, dw, dm, dn
}

// bbMatched reports whether every hard read equals the busy beaver and every
// move puts probability at least 0.9 on its direction. The halting rule's
// next state has to be the nearest symbol to 0.
func bbMatched(p bbParams) bool {
	rules := bb2()
	for s := 1; s <= 2; s++ {
		for sym := 0; sym < 2; sym++ {
			rule := rules[s][sym]
			if classOf(p.write[s][sym], 2) != rule.write || discreteBit(p.write[s][sym]) != byte('0'+rule.write) {
				return false
			}
			if classOf(p.next[s][sym], 3) != rule.next {
				return false
			}
			prob := logistic(p.move[s][sym])
			if rule.move > 0 {
				if prob < 0.9 {
					return false
				}
			} else if prob > 0.1 {
				return false
			}
		}
	}
	return true
}

// transitions is the binary machine read off the hard decisions.
// A positive write weight writes 1. A non-negative move logit moves right.
// The next state is the nearest of 0, 1, and 2.
func (p bbParams) transitions() map[int][2]Transition {
	m := make(map[int][2]Transition, 2)
	for s := 1; s <= 2; s++ {
		var pair [2]Transition
		for sym := 0; sym < 2; sym++ {
			move := -1
			if p.move[s][sym] >= 0 {
				move = 1
			}
			pair[sym] = Transition{
				WriteSymbol: discreteBit(p.write[s][sym]),
				MoveDir:     move,
				NextState:   classOf(p.next[s][sym], 3),
			}
		}
		m[s] = pair
	}
	return m
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

// bbScore runs table the way HaltAt does. ones is the number of cells
// holding 1 when the run stops. haltStep is -1 when the bound is reached
// with the machine still running.
func bbScore(table map[int][2]Transition, maxSteps int) (states []int, haltStep, ones int) {
	tape := make(map[int]byte)
	head := 0
	state := 1
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

func formatDir(move int) string {
	if move > 0 {
		return "R"
	}
	return "L"
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

// learnBB follows the Gaussian partial of each busy-beaver rule. The write
// and the next state step by τ^2, which cancels 1/τ^2. The move logit steps
// by 1. Weights start at 0. The halting rule is state 2 reading 1: its next
// state is trained on 0.
func learnBB(cfg bbConfig) error {
	if cfg.States != 2 {
		return fmt.Errorf("only BB(2) is learned")
	}
	if cfg.Iters <= 0 {
		cfg.Iters = 32
	}
	cfg.logf("gaussian BB(2)")
	cfg.logf("  2 states, 2 symbols, blank tape")
	cfg.logf("  each transition learns a write, a move, and a next state")
	cfg.logf("  the next state is a Gaussian softmax over 0, 1, and 2")
	cfg.logf("  state 0 is halt, observed on the step after the transition")
	cfg.logf("  the halting transition is state 2 reading 1")
	var p bbParams
	var loss float64
	matched := -1
	for iter := 0; iter <= cfg.Iters; iter++ {
		var dw, dm, dn [3][2]float64
		loss, dw, dm, dn = bbPartial(p)
		sure := bbMatched(p)
		if iter == 0 || iter == cfg.Iters || iter%4 == 0 || (sure && iter > 0) {
			cfg.logf("iter %d  loss %.6f", iter, loss)
		}
		if sure && iter > 0 {
			matched = iter
			break
		}
		if iter == cfg.Iters {
			break
		}
		for s := 1; s <= 2; s++ {
			for sym := 0; sym < 2; sym++ {
				// bbPartial divides by the four rules. Multiply back so the
				// step is one rule's partial, (E[symbol] - target).
				p.write[s][sym] -= dw[s][sym] * 4 * bbTau * bbTau
				p.next[s][sym] -= dn[s][sym] * 4 * bbTau * bbTau
				p.move[s][sym] -= dm[s][sym] * 4
				if p.move[s][sym] > 20 {
					p.move[s][sym] = 20
				}
				if p.move[s][sym] < -20 {
					p.move[s][sym] = -20
				}
			}
		}
	}
	table := p.transitions()
	for s := 1; s <= 2; s++ {
		for sym := 0; sym < 2; sym++ {
			rule := table[s][sym]
			cfg.logf("  state %d read %d  write %c  move %s  next %d", s, sym, rule.WriteSymbol, formatDir(rule.MoveDir), rule.NextState)
		}
	}
	code := EncodeMachine(table)
	cfg.logf("encoding %s", code)
	states, haltStep, ones := bbScore(table, bbBound)
	cfg.logf("states %s", formatStates(states))
	if haltStep < 0 {
		cfg.logf("still running at step %d", bbBound)
	} else {
		cfg.logf("halt step %d", haltStep)
	}
	cfg.logf("ones %d", ones)
	if matched < 0 {
		return fmt.Errorf("BB(2) did not learn the halting state after %d steps", cfg.Iters)
	}
	if haltStep != 6 || ones != 4 {
		return fmt.Errorf("BB(2) halted at step %d with %d ones", haltStep, ones)
	}
	return nil
}
