// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

// bb5MaxSteps is one step past the halt observation of the 5-state busy
// beaver. The halt is state 0 at the start of that step. A run that is
// still going there has not been shown to halt.
const bb5MaxSteps = 47176871

// bbStateTau is the width of the state softmax. The six symbols are the
// halt state and the five active states.
const bbStateTau = 0.5

// bb5Rules is the 5-state 2-symbol busy beaver. State 1 is A. The only
// transition into state 0 is state 5 reading 0, which writes 1 and moves
// right. Two rules write 1, move right, and enter state 2, so the tape
// alone does not say which one fired. The state chain does.
func bb5Rules() [6][2]bbRule {
	var t [6][2]bbRule
	t[1][0] = bbRule{write: 1, move: 1, next: 2}
	t[1][1] = bbRule{write: 1, move: -1, next: 3}
	t[2][0] = bbRule{write: 1, move: 1, next: 3}
	t[2][1] = bbRule{write: 1, move: 1, next: 2}
	t[3][0] = bbRule{write: 1, move: 1, next: 4}
	t[3][1] = bbRule{write: 0, move: -1, next: 5}
	t[4][0] = bbRule{write: 1, move: -1, next: 1}
	t[4][1] = bbRule{write: 1, move: -1, next: 4}
	t[5][0] = bbRule{write: 1, move: 1, next: 0}
	t[5][1] = bbRule{write: 0, move: -1, next: 1}
	return t
}

// bb5NoHaltRules is the 5-state champion with the halt transition removed.
// State 5 reading 0 writes 1, moves right, and stays in state 5. No rule
// enters state 0, so the machine does not halt. The state chain still
// identifies each step, because each pair of states is one rule.
func bb5NoHaltRules() [6][2]bbRule {
	t := bb5Rules()
	t[5][0].next = 5
	return t
}

// rulesEnterHalt reports whether any transition's next state is 0.
func rulesEnterHalt(rules [6][2]bbRule) bool {
	for s := 1; s <= 5; s++ {
		for sym := 0; sym < 2; sym++ {
			if rules[s][sym].next == 0 {
				return true
			}
		}
	}
	return false
}

// bb5Transitions is bb5Rules in the simulator's binary encoding.
func bb5Transitions() map[int][2]Transition {
	return bbRulesTransitions(bb5Rules())
}

func bbRulesTransitions(rules [6][2]bbRule) map[int][2]Transition {
	m := make(map[int][2]Transition, 5)
	for s := 1; s <= 5; s++ {
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

// spanTape is a dense 0/1 tape. lo is the cell index of buf[0]. visLo and
// visHi are the cells the head has actually visited.
type spanTape struct {
	buf       []byte
	lo        int
	visLo     int
	visHi     int
	seen      bool
	allocated bool
}

func (t *spanTape) touch(i int) {
	if !t.seen || i < t.visLo {
		t.visLo = i
	}
	if !t.seen || i > t.visHi {
		t.visHi = i
	}
	t.seen = true
}

func (t *spanTape) ensure(i int) {
	if !t.allocated {
		t.buf = []byte{0}
		t.lo = i
		t.allocated = true
		return
	}
	if i >= t.lo && i < t.lo+len(t.buf) {
		return
	}
	if i < t.lo {
		extra := t.lo - i
		nlen := len(t.buf) + extra
		if nlen < 2*len(t.buf) {
			nlen = 2 * len(t.buf)
		}
		nb := make([]byte, nlen)
		copy(nb[nlen-len(t.buf):], t.buf)
		t.lo -= nlen - len(t.buf)
		t.buf = nb
		return
	}
	nlen := i - t.lo + 1
	if nlen < 2*len(t.buf) {
		nlen = 2 * len(t.buf)
	}
	if cap(t.buf) >= nlen {
		t.buf = t.buf[:nlen]
		return
	}
	nb := make([]byte, nlen)
	copy(nb, t.buf)
	t.buf = nb
}

func (t *spanTape) get(i int) byte {
	if !t.allocated || i < t.lo || i >= t.lo+len(t.buf) {
		return 0
	}
	return t.buf[i-t.lo]
}

func (t *spanTape) set(i int, v byte) {
	t.ensure(i)
	t.buf[i-t.lo] = v
}

func (t *spanTape) clone() spanTape {
	c := *t
	c.buf = append([]byte(nil), t.buf...)
	return c
}

func (t *spanTape) ones() int {
	if !t.seen {
		return 0
	}
	n := 0
	for i := t.visLo; i <= t.visHi; i++ {
		if t.get(i) == 1 {
			n++
		}
	}
	return n
}

func (t *spanTape) blank() bool {
	if !t.allocated {
		return true
	}
	for _, b := range t.buf {
		if b != 0 {
			return false
		}
	}
	return true
}

// bb5Run is a forward execution. chain[t] is the state at the start of
// step t. The halt state is not stored there; halted says the run entered
// state 0. state and head are the machine after the last transition.
type bb5Run struct {
	chain  []byte
	tape   spanTape
	head   int
	state  int
	steps  int
	halted bool
}

// runBB5 runs the 5-state machine from state 1 on a blank tape. maxSteps
// counts transitions. State 0 stops the run before another transition.
func runBB5(maxSteps int, logf func(int)) bb5Run {
	return runRules(bb5Rules(), maxSteps, logf)
}

// runRules runs one fixed table from state 1 on a blank tape.
func runRules(rules [6][2]bbRule, maxSteps int, logf func(int)) bb5Run {
	var run bb5Run
	if maxSteps > 0 {
		run.chain = make([]byte, 0, maxSteps)
	}
	state := 1
	head := 0
	finish := func(halted bool) bb5Run {
		run.halted = halted
		run.head = head
		run.state = state
		return run
	}
	for run.steps < maxSteps {
		if state == 0 {
			return finish(true)
		}
		run.chain = append(run.chain, byte(state))
		run.tape.touch(head)
		read := run.tape.get(head)
		rule := rules[state][int(read)]
		run.tape.set(head, byte(rule.write))
		head += rule.move
		state = rule.next
		run.steps++
		if logf != nil && run.steps%5000000 == 0 {
			logf(run.steps)
		}
	}
	return finish(state == 0)
}

// ruleInto returns the symbol state `from` reads in order to enter `to`.
// On this machine each pair of states is one rule.
func ruleInto(rules [6][2]bbRule, from, to int) (sym int, rule bbRule, ok bool) {
	if from < 1 || from > 5 {
		return 0, bbRule{}, false
	}
	for sym = 0; sym < 2; sym++ {
		if rules[from][sym].next == to {
			return sym, rules[from][sym], true
		}
	}
	return 0, bbRule{}, false
}

// bb5Undo walks the chain backward from state. means supplies the guessed
// previous state: the symbol nearest that state's learned mean. The tape
// must already hold the guessed bits. A matching write is turned back into
// the symbol the rule read. blank is every cell 0, the head on cell 0, and
// the state back at 1, after at least one step. An already blank tape at
// state 1 is not a halt.
func bb5Undo(rules [6][2]bbRule, tape spanTape, head int, chain []byte, means [6]float64, state int) (blank bool, steps int) {
	for t := len(chain) - 1; t >= 0; t-- {
		prev := classOf(means[int(chain[t])], 6)
		sym, rule, ok := ruleInto(rules, prev, state)
		if !ok {
			return false, steps
		}
		cell := head - rule.move
		if tape.get(cell) != byte(rule.write) {
			return false, steps
		}
		tape.set(cell, byte(sym))
		head = cell
		state = prev
		steps++
	}
	blank = len(chain) > 0 && tape.blank() && head == 0 && state == 1
	return blank, steps
}

// bb5BackwardFromHalt undoes a run that has entered state 0.
func bb5BackwardFromHalt(tape spanTape, head int, chain []byte, means [6]float64) (blank bool, steps int) {
	return bb5Undo(bb5Rules(), tape, head, chain, means, 0)
}

// chainCounts tallies the states on the chain.
func chainCounts(chain []byte) (count [6]int) {
	for _, s := range chain {
		if s < 6 {
			count[s]++
		}
	}
	return count
}

// chainLoss is the mean state cross-entropy along the chain and the partial
// of that mean. A state the chain never visits has a zero partial.
func chainLoss(means [6]float64, count [6]int) (mean float64, dmu [6]float64) {
	n := 0
	for _, c := range count {
		n += c
	}
	if n == 0 {
		return 0, dmu
	}
	nf := float64(n)
	for s := 0; s < 6; s++ {
		if count[s] == 0 {
			continue
		}
		nll, g := classLoss(means[s], bbStateTau, s, 6)
		w := float64(count[s]) / nf
		mean += w * nll
		dmu[s] = w * g
	}
	return mean, dmu
}

// fitChainStates moves each visited state's mean by its own partial. The
// logged loss is the mean over the chain. Dividing by the chain length would
// leave a state that appears once, including the step into halt, almost
// unmoved, so the step puts back that state's share.
func fitChainStates(count [6]int, iters int, logf func(int, float64, [6]float64)) ([6]float64, bool) {
	var means [6]float64
	for iter := 0; iter <= iters; iter++ {
		loss, dmu := chainLoss(means, count)
		sure := true
		for s := 1; s <= 5; s++ {
			if count[s] == 0 {
				continue
			}
			if classOf(means[s], 6) != s {
				sure = false
			}
		}
		if logf != nil && (iter == 0 || iter == iters || iter%4 == 0 || (sure && iter > 0)) {
			logf(iter, loss, means)
		}
		if sure && iter > 0 {
			return means, true
		}
		if iter == iters {
			break
		}
		n := 0
		for _, c := range count {
			n += c
		}
		nf := float64(n)
		for s := 1; s <= 5; s++ {
			if count[s] == 0 {
				continue
			}
			// dmu is the partial of the mean. Multiply by n/count so the
			// step is one visit's partial, (E[state] - s).
			means[s] -= dmu[s] * (nf / float64(count[s])) * bbStateTau * bbStateTau
		}
	}
	return means, false
}

func haltTapeFromSpan(tape spanTape) []haltGauss {
	if !tape.seen {
		return nil
	}
	out := make([]haltGauss, 0, tape.visHi-tape.visLo+1)
	for i := tape.visLo; i <= tape.visHi; i++ {
		out = append(out, haltGauss{at: i, bit: int(tape.get(i)), cell: unitGaussCell(0)})
	}
	return out
}

func guessTape(src spanTape, learned []haltGauss) spanTape {
	out := src.clone()
	for _, cell := range learned {
		out.set(cell.at, byte(gaussBit(cell.cell.mean)))
	}
	return out
}

func guessesMatch(src spanTape, learned []haltGauss) bool {
	for _, cell := range learned {
		if int(src.get(cell.at)) != gaussBit(cell.cell.mean) {
			return false
		}
	}
	return true
}

func (cfg bbConfig) fitGaussTape(tape []haltGauss) (int, error) {
	if len(tape) == 0 {
		return 0, fmt.Errorf("BB(5) halt tape is empty")
	}
	matched := -1
	for iter := 0; iter <= cfg.Iters; iter++ {
		loss, dmu := haltTapeLoss(tape)
		sure := haltTapeSure(tape)
		if iter == 0 || iter == cfg.Iters || iter%4 == 0 || (sure && iter > 0) {
			cfg.logf("tape iter %d  loss %.6f", iter, loss)
		}
		if sure && iter > 0 {
			matched = iter
			break
		}
		if iter == cfg.Iters {
			break
		}
		n := float64(len(tape))
		for i := range tape {
			tape[i].cell.mean -= dmu[i] * n
		}
	}
	if matched < 0 {
		return 0, fmt.Errorf("BB(5) did not learn the halting tape after %d steps", cfg.Iters)
	}
	return matched, nil
}

func tapeProbSummary(tape []haltGauss) (n1 int, p1 float64, n0 int, p0 float64) {
	for _, cell := range tape {
		p := gaussTailProb(cell.cell.mean, cell.cell.dev())
		if cell.bit == 1 {
			n1++
			p1 += p
		} else {
			n0++
			p0 += p
		}
	}
	if n1 > 0 {
		p1 /= float64(n1)
	}
	if n0 > 0 {
		p0 /= float64(n0)
	}
	return n1, p1, n0, p0
}

// learnBB5 fits the state chain and the Gaussian halt tape, guesses the
// bits, and runs the machine backward. A blank tape at the end is the
// forward machine reaching that halt from a blank tape.
func learnBB5(cfg bbConfig) error {
	cfg.logf("gaussian BB(5)")
	cfg.logf("  backpropagation starts at the halting state")
	cfg.logf("  the state chain and the halt tape are gaussian")
	cfg.logf("  bits are guessed from the tape and the machine is run backward")
	cfg.logf("  a blank tape at the end is a halt from a blank tape")
	cfg.logf("running forward")
	run := runBB5(bb5MaxSteps, func(steps int) {
		cfg.logf("  forward step %d", steps)
	})
	if !run.halted {
		return fmt.Errorf("BB(5) still running at step %d", bb5MaxSteps)
	}
	ones := run.tape.ones()
	cells := 0
	if run.tape.seen {
		cells = run.tape.visHi - run.tape.visLo + 1
	}
	cfg.logf("halt step %d", run.steps)
	cfg.logf("ones %d", ones)
	cfg.logf("cells %d", cells)
	cfg.logf("head %d", run.head)

	count := chainCounts(run.chain)
	cfg.logf("state chain")
	means, ok := fitChainStates(count, cfg.Iters, func(iter int, loss float64, means [6]float64) {
		cfg.logf("chain iter %d  loss %.6f", iter, loss)
		for s := 1; s <= 5; s++ {
			if count[s] == 0 {
				continue
			}
			cfg.logf("  state %d  mean %.4f  class %d  visits %d", s, means[s], classOf(means[s], 6), count[s])
		}
	})
	if !ok {
		return fmt.Errorf("BB(5) did not learn the state chain after %d steps", cfg.Iters)
	}

	learned := haltTapeFromSpan(run.tape)
	cfg.logf("halting tape gaussians")
	cfg.logf("  each visited cell is N(mean, 1)")
	cfg.logf("  the probability is the mass above 0")
	cfg.logf("  a sample above 0 reads as 1")
	if _, err := cfg.fitGaussTape(learned); err != nil {
		return err
	}
	n1, p1, n0, p0 := tapeProbSummary(learned)
	cfg.logf("  1-cells %d  mean p %.4f", n1, p1)
	cfg.logf("  0-cells %d  mean p %.4f", n0, p0)
	if !guessesMatch(run.tape, learned) {
		return fmt.Errorf("BB(5) gaussian guesses do not match the halt tape")
	}
	cfg.logf("guessed bits match")

	guessed := guessTape(run.tape, learned)
	blank, back := bb5BackwardFromHalt(guessed, run.head, run.chain, means)
	cfg.logf("backward steps %d", back)
	if blank {
		cfg.logf("blank tape yes")
		return nil
	}
	cfg.logf("blank tape no")
	return fmt.Errorf("BB(5) backward run did not reach a blank tape")
}

// learnBB5NoHalt runs the champion with no transition into halt. The forward
// run stops at the step bound still going. Gradient descent fits that state's
// chain and the Gaussian tape, and the guessed bits are run backward. A blank
// tape rewinds the run. It is not a halting state. Starting the same walk in
// state 0 does not reach a blank tape, because the machine never entered it.
func learnBB5NoHalt(cfg bbConfig) error {
	rules := bb5NoHaltRules()
	if rulesEnterHalt(rules) {
		return fmt.Errorf("BB(5) non-halting machine has a transition into state 0")
	}
	cfg.logf("gaussian BB(5)")
	cfg.logf("  this machine does not halt")
	cfg.logf("  the champion's halt rule stays in state 5")
	cfg.logf("  no transition enters state 0")
	cfg.logf("  backpropagation follows the state chain to the step bound")
	cfg.logf("  the tape there is gaussian")
	cfg.logf("  bits are guessed and the machine is run backward")
	cfg.logf("  a blank tape rewinds that run")
	cfg.logf("  it is not a halting state")
	cfg.logf("running forward")
	run := runRules(rules, bb5MaxSteps, func(steps int) {
		cfg.logf("  forward step %d", steps)
	})
	if run.halted {
		return fmt.Errorf("BB(5) non-halting machine entered state 0 at step %d", run.steps)
	}
	if run.steps != bb5MaxSteps {
		return fmt.Errorf("BB(5) non-halting machine stopped at step %d", run.steps)
	}
	ones := run.tape.ones()
	cells := 0
	if run.tape.seen {
		cells = run.tape.visHi - run.tape.visLo + 1
	}
	cfg.logf("still running at step %d", run.steps)
	cfg.logf("state %d", run.state)
	cfg.logf("ones %d", ones)
	cfg.logf("cells %d", cells)
	cfg.logf("head %d", run.head)

	count := chainCounts(run.chain)
	cfg.logf("state chain")
	means, ok := fitChainStates(count, cfg.Iters, func(iter int, loss float64, means [6]float64) {
		cfg.logf("chain iter %d  loss %.6f", iter, loss)
		for s := 1; s <= 5; s++ {
			if count[s] == 0 {
				continue
			}
			cfg.logf("  state %d  mean %.4f  class %d  visits %d", s, means[s], classOf(means[s], 6), count[s])
		}
	})
	if !ok {
		return fmt.Errorf("BB(5) did not learn the state chain after %d steps", cfg.Iters)
	}

	learned := haltTapeFromSpan(run.tape)
	cfg.logf("tape gaussians")
	cfg.logf("  each visited cell is N(mean, 1)")
	cfg.logf("  the probability is the mass above 0")
	cfg.logf("  a sample above 0 reads as 1")
	if _, err := cfg.fitGaussTape(learned); err != nil {
		return err
	}
	n1, p1, n0, p0 := tapeProbSummary(learned)
	cfg.logf("  1-cells %d  mean p %.4f", n1, p1)
	cfg.logf("  0-cells %d  mean p %.4f", n0, p0)
	if !guessesMatch(run.tape, learned) {
		return fmt.Errorf("BB(5) gaussian guesses do not match the tape")
	}
	cfg.logf("guessed bits match")

	guessed := guessTape(run.tape, learned)
	cfg.logf("backward from state %d", run.state)
	blank, back := bb5Undo(rules, guessed, run.head, run.chain, means, run.state)
	cfg.logf("backward steps %d", back)
	if blank {
		cfg.logf("blank tape yes")
	} else {
		cfg.logf("blank tape no")
		return fmt.Errorf("BB(5) non-halting run did not rewind to a blank tape")
	}

	cfg.logf("backward from the halting state")
	haltBlank, haltBack := bb5Undo(rules, guessTape(run.tape, learned), run.head, run.chain, means, 0)
	cfg.logf("backward steps %d", haltBack)
	if haltBlank {
		cfg.logf("blank tape yes")
		return fmt.Errorf("BB(5) non-halting machine reached a blank tape from state 0")
	}
	cfg.logf("blank tape no")
	return nil
}
