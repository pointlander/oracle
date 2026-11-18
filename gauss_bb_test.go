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
	states, step, ones := bbScore(table, bbBound)
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

func TestBBHaltPartialMatchesFiniteDifference(t *testing.T) {
	var p bbParams
	p.next[2][1] = 0.3
	p.write[2][1] = 0.4
	p.move[2][1] = -0.2
	_, dw, dm, dn := bbPartial(p)
	const eps = 1e-4
	partial := func(set func(*bbParams, float64), at float64) float64 {
		up, down := p, p
		set(&up, at+eps)
		set(&down, at-eps)
		mu, _, _, _ := bbPartial(up)
		md, _, _, _ := bbPartial(down)
		return (mu - md) / (2 * eps)
	}
	if d := partial(func(q *bbParams, v float64) { q.next[2][1] = v }, p.next[2][1]); math.Abs(dn[2][1]-d) > 1e-6 {
		t.Fatalf("halt next partial %.6f finite difference %.6f", dn[2][1], d)
	}
	if d := partial(func(q *bbParams, v float64) { q.write[2][1] = v }, p.write[2][1]); math.Abs(dw[2][1]-d) > 1e-6 {
		t.Fatalf("halt write partial %.6f finite difference %.6f", dw[2][1], d)
	}
	if d := partial(func(q *bbParams, v float64) { q.move[2][1] = v }, p.move[2][1]); math.Abs(dm[2][1]-d) > 1e-6 {
		t.Fatalf("halt move partial %.6f finite difference %.6f", dm[2][1], d)
	}
	// The halt target is 0 and this mean sits above 0, so the partial
	// of the loss is positive and the step moves the mean down.
	if dn[2][1] <= 0 {
		t.Fatalf("halt partial %.6f", dn[2][1])
	}
}

func TestGaussLearnsBB2Halt(t *testing.T) {
	var lines []string
	err := learnBB(bbConfig{States: 2, Iters: 32, Log: func(s string) { lines = append(lines, s) }})
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Join(lines, "\n")
	if !strings.Contains(text, "halt step 6") || !strings.Contains(text, "ones 4") {
		t.Fatalf("log:\n%s", text)
	}
	if !strings.Contains(text, "state 2 read 1  write 1  move R  next 0") {
		t.Fatalf("halting rule missing:\n%s", text)
	}
}
