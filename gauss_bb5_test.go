// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"testing"
)

// bb5Machine is 1RB1LC_1RC1RB_1RD0LE_1LA1LD_1RZ0LA in this program's encoding.
const bb5Machine = "11000010100000111100001111000010110001000000010110000001100001001100000000000001"

func TestBB5EncodingMatchesTheChampion(t *testing.T) {
	code := EncodeMachine(bb5Transitions())
	if code != bb5Machine {
		t.Fatalf("encoding %s", code)
	}
	if got := (&UniversalBoundedSimulator{MaxSteps: 8}).HaltAt(code, ""); got != -1 {
		t.Fatalf("HaltAt %d", got)
	}
	rules := bb5Rules()
	seen := map[[2]int]int{}
	for s := 1; s <= 5; s++ {
		for sym := 0; sym < 2; sym++ {
			to := rules[s][sym].next
			key := [2]int{s, to}
			if _, ok := seen[key]; ok {
				t.Fatalf("states %d to %d are two rules", s, to)
			}
			seen[key] = sym
			got, rule, ok := ruleInto(rules, s, to)
			if !ok || got != sym || rule != rules[s][sym] {
				t.Fatalf("rule %d read %d", s, sym)
			}
		}
	}
	// The only transition into halt is state 5 reading 0.
	if rules[5][0].next != 0 || rules[5][0].write != 1 || rules[5][0].move != 1 {
		t.Fatalf("halt rule %+v", rules[5][0])
	}
}

func TestBB5PrefixAgreesWithTheSimulator(t *testing.T) {
	run := runBB5(8, nil)
	if run.halted || run.steps != 8 || run.head != 0 || run.state != 2 {
		t.Fatalf("halted %v steps %d head %d state %d", run.halted, run.steps, run.head, run.state)
	}
	want := []byte{1, 2, 3, 4, 1, 3, 5, 1}
	if len(run.chain) != len(want) {
		t.Fatalf("chain %v", run.chain)
	}
	for i, s := range want {
		if run.chain[i] != s {
			t.Fatalf("chain %v", run.chain)
		}
	}
	states, step, _ := bbScore(bb5Transitions(), 9, 1)
	if step != -1 || len(states) != 9 || states[8] != 2 {
		t.Fatalf("score step %d states %v", step, states)
	}
	for i, s := range want {
		if states[i] != int(s) {
			t.Fatalf("score %v", states)
		}
	}
	if run.tape.get(-1) != 1 || run.tape.get(0) != 0 || run.tape.get(1) != 0 || run.tape.get(2) != 1 || run.tape.get(3) != 1 {
		t.Fatalf("tape -1=%d 0=%d 1=%d 2=%d 3=%d", run.tape.get(-1), run.tape.get(0), run.tape.get(1), run.tape.get(2), run.tape.get(3))
	}
	if run.tape.ones() != 3 || run.tape.visHi-run.tape.visLo+1 != 5 {
		t.Fatalf("ones %d cells %d", run.tape.ones(), run.tape.visHi-run.tape.visLo+1)
	}
}

func TestBB5ChainLossMatchesFiniteDifference(t *testing.T) {
	count := [6]int{0, 3, 5, 1, 4, 2}
	means := [6]float64{0, 0.2, -0.4, 1.1, 0.7, 2.5}
	_, dmu := chainLoss(means, count)
	const eps = 1e-5
	for s := 1; s <= 5; s++ {
		up, down := means, means
		up[s] += eps
		down[s] -= eps
		lu, _ := chainLoss(up, count)
		ld, _ := chainLoss(down, count)
		diff := (lu - ld) / (2 * eps)
		if math.Abs(dmu[s]-diff) > 1e-6 {
			t.Fatalf("state %d partial %.6f finite difference %.6f", s, dmu[s], diff)
		}
	}
	if dmu[0] != 0 {
		t.Fatalf("halt partial %.6f", dmu[0])
	}
}

func TestBB5FitsTheStateChain(t *testing.T) {
	// State 2 is absent and state 5 appears once. Both must come out right:
	// an absent state stays put, and one visit is still a full step.
	count := [6]int{0, 8, 0, 3, 5, 1}
	means, ok := fitChainStates(count, 32, nil)
	if !ok {
		t.Fatal("chain did not classify")
	}
	for _, s := range []int{1, 3, 4, 5} {
		if classOf(means[s], 6) != s {
			t.Fatalf("state %d mean %.4f class %d", s, means[s], classOf(means[s], 6))
		}
	}
	if means[2] != 0 || classOf(means[2], 6) == 2 {
		t.Fatalf("absent state mean %.4f", means[2])
	}
}

func TestBB5HaltKeepsTheHeadAndUndoesToBlank(t *testing.T) {
	// One transition into halt. The head has moved, so a backward walk that
	// forgot it would restore the wrong cell.
	var rules [6][2]bbRule
	rules[1][0] = bbRule{write: 1, move: 1, next: 0}
	rules[1][1] = bbRule{write: 0, move: -1, next: 2}
	run := runRules(rules, 4, nil)
	if !run.halted || run.steps != 1 || run.head != 1 || run.state != 0 || len(run.chain) != 1 || run.chain[0] != 1 {
		t.Fatalf("halted %v steps %d head %d state %d chain %v", run.halted, run.steps, run.head, run.state, run.chain)
	}
	if run.tape.get(0) != 1 {
		t.Fatalf("written %d", run.tape.get(0))
	}
	var means [6]float64
	for s := 0; s < 6; s++ {
		means[s] = float64(s)
	}
	blank, steps := bb5Undo(rules, run.tape.clone(), run.head, run.chain, means, 0)
	if !blank || steps != 1 {
		t.Fatalf("blank %v steps %d", blank, steps)
	}
	// A blank tape that is already in state 1 did not come from a halt.
	var empty spanTape
	if blank, _ := bb5BackwardFromHalt(empty, 0, nil, means); blank {
		t.Fatal("empty chain counted as a halt")
	}
}

func TestBB5GaussianTapeUndoesThePrefix(t *testing.T) {
	const n = 8
	run := runBB5(n, nil)
	count := chainCounts(run.chain)
	means, ok := fitChainStates(count, 32, nil)
	if !ok {
		t.Fatal("chain did not classify")
	}
	for s := 1; s <= 5; s++ {
		if classOf(means[s], 6) != s {
			t.Fatalf("state %d class %d", s, classOf(means[s], 6))
		}
	}
	learned := haltTapeFromSpan(run.tape)
	if _, err := (bbConfig{Iters: 32}).fitGaussTape(learned); err != nil {
		t.Fatal(err)
	}
	if !guessesMatch(run.tape, learned) {
		t.Fatal("guesses differ")
	}
	guessed := guessTape(run.tape, learned)
	blank, steps := bb5Undo(bb5Rules(), guessed, run.head, run.chain, means, run.state)
	if !blank || steps != n {
		t.Fatalf("blank %v steps %d", blank, steps)
	}
	// The prefix has not halted. Starting the undo in state 0 is the wrong chain.
	if blank, _ := bb5BackwardFromHalt(guessed, run.head, run.chain, means); blank {
		t.Fatal("prefix counted as a halt")
	}
}

func TestBB5LearnedChainUndoesALongerPrefix(t *testing.T) {
	const n = 4096
	run := runBB5(n, nil)
	if run.halted || run.steps != n {
		t.Fatalf("halted %v steps %d", run.halted, run.steps)
	}
	means, ok := fitChainStates(chainCounts(run.chain), 32, nil)
	if !ok {
		t.Fatal("chain did not classify")
	}
	learned := haltTapeFromSpan(run.tape)
	if _, err := (bbConfig{Iters: 32}).fitGaussTape(learned); err != nil {
		t.Fatal(err)
	}
	guessed := guessTape(run.tape, learned)
	blank, steps := bb5Undo(bb5Rules(), guessed, run.head, run.chain, means, run.state)
	if !blank || steps != n {
		t.Fatalf("blank %v steps %d ones %d", blank, steps, run.tape.ones())
	}
}
