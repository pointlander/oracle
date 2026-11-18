// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"strings"
	"testing"
)

func TestBB2HaltsAtSix(t *testing.T) {
	table := bb2Transitions()
	code := EncodeMachine(table)
	if got := (&UniversalBoundedSimulator{MaxSteps: bbBound}).HaltAt(code, ""); got != 6 {
		t.Fatalf("HaltAt %d encoding %s", got, code)
	}
	states, step, ones := bbScore(table, bbBound, 1)
	if step != 6 || ones != 4 || formatStates(states) != "1 2 1 2 1 2 0" {
		t.Fatalf("step %d ones %d states %s", step, ones, formatStates(states))
	}
	// The halting rule is the only transition into state 0.
	halts := 0
	for s := 1; s <= 2; s++ {
		for sym := 0; sym < 2; sym++ {
			if table[s][sym].NextState == 0 {
				halts++
				if s != 2 || sym != 1 {
					t.Fatalf("halt from state %d read %d", s, sym)
				}
			}
		}
	}
	if halts != 1 {
		t.Fatalf("halt rules %d", halts)
	}
}

func TestBB2State2HaltsEarlier(t *testing.T) {
	// The same machine started in state 2 also reaches halt, on a zero
	// tape, but not the busy beaver's halt. That gap is the gradient.
	states, step, ones := bbScore(bb2Transitions(), bbBound, 2)
	if step != 3 || ones != 2 || formatStates(states) != "2 1 2 0" {
		t.Fatalf("step %d ones %d states %s", step, ones, formatStates(states))
	}
	if c1, c2 := bbBranchFrom(1).cost, bbBranchFrom(2).cost; c1 != 0 || c2 <= c1 {
		t.Fatalf("cost state1 %.4f state2 %.4f", c1, c2)
	}
}

func TestBBBackpropMatchesFiniteDifference(t *testing.T) {
	const logit = 0.2
	_, dlogit := bbBackprop(logit)
	const eps = 1e-4
	up, _ := bbBackprop(logit + eps)
	down, _ := bbBackprop(logit - eps)
	diff := (up - down) / (2 * eps)
	if math.Abs(dlogit-diff) > 1e-6 {
		t.Fatalf("partial %.6f finite difference %.6f", dlogit, diff)
	}
	// The halt cost of state 2 is higher, so the partial is negative and
	// the step moves the logit toward state 1.
	if dlogit >= 0 {
		t.Fatalf("partial %.6f", dlogit)
	}
}

func TestGaussLearnsBB2InitialState(t *testing.T) {
	var lines []string
	err := learnBB(bbConfig{States: 2, Iters: 32, Log: func(s string) { lines = append(lines, s) }})
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Join(lines, "\n")
	if !strings.Contains(text, "initial state 1") || !strings.Contains(text, "halt step 6") || !strings.Contains(text, "ones 4") {
		t.Fatalf("log:\n%s", text)
	}
	if !strings.Contains(text, "states 1 2 1 2 1 2 0") {
		t.Fatalf("forward run missing:\n%s", text)
	}
}
