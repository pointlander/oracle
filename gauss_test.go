// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestGaussBitAndRunningDeviation(t *testing.T) {
	if gaussBit(0.1) != 1 || gaussBit(0) != 0 || gaussBit(-0.2) != 0 {
		t.Fatal("threshold")
	}
	var c gaussCell
	c.observe(1)
	c.observe(3)
	if c.n != 2 || c.mean != 2 || math.Abs(c.dev()-1) > 1e-12 {
		t.Fatalf("mean %v dev %v n %d", c.mean, c.dev(), c.n)
	}
	var same gaussCell
	same.observe(2)
	same.observe(2)
	if same.mean != 2 || same.dev() != 0 {
		t.Fatalf("constant write mean %v dev %v", same.mean, same.dev())
	}
	started := pointMass(1)
	started.observe(-1)
	if started.mean != 0 || math.Abs(started.dev()-1) > 1e-12 {
		t.Fatalf("mixed input mean %v dev %v", started.mean, started.dev())
	}
}

func TestGaussBlankSamplesAreAboutHalfOnes(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	ones := 0
	const n = 4000
	for range n {
		if gaussBit(sampleGauss(nil, rng)) == 1 {
			ones++
		}
	}
	frac := float64(ones) / n
	if frac < 0.45 || frac > 0.55 {
		t.Fatalf("blank ones %.3f", frac)
	}
	pos := pointMass(1)
	neg := pointMass(-1)
	for range 20 {
		if gaussBit(sampleGauss(pos, rng)) != 1 {
			t.Fatal("positive point mass")
		}
		if gaussBit(sampleGauss(neg, rng)) != 0 {
			t.Fatal("negative point mass")
		}
	}
}

func TestGaussHaltFollowsTheStepBound(t *testing.T) {
	halt := &gaussMachine{state: make([][2]gaussTransition, 2)}
	halt.state[1][0] = gaussTransition{Write: 5, Move: 1, Next: 0}
	halt.state[1][1] = gaussTransition{Write: -5, Move: -1, Next: 0}
	rng := rand.New(rand.NewSource(1))
	run := runGauss(halt, 8, nil, rng)
	if !run.Halted || run.HaltStep != 1 {
		t.Fatalf("halt %+v", run.Halted)
	}
	cell := run.cells[0]
	if cell == nil || cell.n != 1 || (cell.mean != 5 && cell.mean != -5) || cell.dev() != 0 {
		t.Fatalf("first write %+v", cell)
	}

	loop := &gaussMachine{state: make([][2]gaussTransition, 2)}
	loop.state[1][0] = gaussTransition{Write: 1, Move: 1, Next: 1}
	loop.state[1][1] = gaussTransition{Write: 1, Move: -1, Next: 1}
	run = runGauss(loop, 8, nil, rng)
	if run.Halted || run.Reads != 8 {
		t.Fatalf("loop halted %v reads %d", run.Halted, run.Reads)
	}
}

func TestGaussReturnWritesTheSameWeightTwice(t *testing.T) {
	// Step out and back, then halt. Cell 0 receives the same weight twice.
	m := &gaussMachine{state: make([][2]gaussTransition, 4)}
	for sym := 0; sym < 2; sym++ {
		m.state[1][sym] = gaussTransition{Write: 2, Move: 1, Next: 2}
		m.state[2][sym] = gaussTransition{Write: 2, Move: -1, Next: 3}
		m.state[3][sym] = gaussTransition{Write: 2, Move: 1, Next: 0}
	}
	run := runGauss(m, 8, nil, rand.New(rand.NewSource(1)))
	if !run.Halted || run.HaltStep != 3 {
		t.Fatalf("halted %v step %d", run.Halted, run.HaltStep)
	}
	cell := run.cells[0]
	if cell.n != 2 || cell.mean != 2 || cell.dev() != 0 {
		t.Fatalf("cell0 n %d mean %v dev %v", cell.n, cell.mean, cell.dev())
	}
}

func TestGaussPointMassSetsTheFirstRead(t *testing.T) {
	// Symbol 1 halts. Symbol 0 stays in state 1 and steps right.
	m := &gaussMachine{state: make([][2]gaussTransition, 2)}
	m.state[1][0] = gaussTransition{Write: 0, Move: 1, Next: 1}
	m.state[1][1] = gaussTransition{Write: 0, Move: 1, Next: 0}
	pos := runGauss(m, 4, func(cells map[int]*gaussCell) {
		cells[0] = pointMass(1)
	}, rand.New(rand.NewSource(1)))
	if !pos.Halted || pos.HaltStep != 1 || pos.FirstRead != 1 {
		t.Fatalf("positive halted %v step %d read %d", pos.Halted, pos.HaltStep, pos.FirstRead)
	}
	neg := runGauss(m, 2, func(cells map[int]*gaussCell) {
		cells[0] = pointMass(-1)
	}, rand.New(rand.NewSource(1)))
	if neg.Halted || neg.FirstRead != 0 {
		t.Fatalf("negative halted %v read %d", neg.Halted, neg.FirstRead)
	}
}

func TestGaussAnalyzeIsDeterministic(t *testing.T) {
	cfg := gaussConfig{States: 1, MaxSteps: 16, Survey: 64, Repeats: 4, Input: 4, Seed: 1}
	a, err := analyzeGauss(cfg)
	if err != nil {
		t.Fatal(err)
	}
	b, err := analyzeGauss(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if a.BlankHalt != b.BlankHalt || a.Unstable != b.Unstable || a.PosHalt != b.PosHalt {
		t.Fatalf("reports differ %+v %+v", a.BlankHalt, b.BlankHalt)
	}
	if math.Abs(a.AlwaysHalt+a.NeverHalt+a.Unstable-1) > 1e-12 {
		t.Fatalf("repeat classes %.4f %.4f %.4f", a.AlwaysHalt, a.NeverHalt, a.Unstable)
	}
	if a.NegFirst1 != 0 || a.PosFirst1 != 1 {
		t.Fatalf("first reads -1 %.3f +1 %.3f", a.NegFirst1, a.PosFirst1)
	}
	if _, err := analyzeGauss(gaussConfig{States: 3, MaxSteps: 8, Survey: 1, Repeats: 1, Input: 1}); err == nil {
		t.Fatal("expected 3 states to be rejected")
	}
}

func TestGaussHelp(t *testing.T) {
	if err := gaussFromArgs([]string{"-h"}); err != nil {
		t.Fatal(err)
	}
}
