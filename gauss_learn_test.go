// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"strconv"
	"strings"
	"testing"
)

func TestSymbolLossMatchesFiniteDifference(t *testing.T) {
	const mu = 10.0
	const tau = 4.0
	const target = 12
	_, analytic := symbolLoss(mu, tau, target)
	const eps = 1e-4
	lp, _ := symbolLoss(mu+eps, tau, target)
	lm, _ := symbolLoss(mu-eps, tau, target)
	diff := (lp - lm) / (2 * eps)
	if math.Abs(analytic-diff) > 1e-4 {
		t.Fatalf("partial %.6f finite difference %.6f", analytic, diff)
	}
}

func TestParseMoves(t *testing.T) {
	right, err := parseMoves("", 3)
	if err != nil || len(right) != 3 || right[0] != 1 || right[2] != 1 {
		t.Fatalf("default %v %v", right, err)
	}
	both, err := parseMoves("rL<>+-", 0)
	want := []int{1, -1, -1, 1, 1, -1}
	if err != nil || len(both) != len(want) {
		t.Fatal(err)
	}
	for i := range want {
		if both[i] != want[i] {
			t.Fatalf("moves %v", both)
		}
	}
	if _, err := parseMoves("RX", 1); err == nil {
		t.Fatal("expected a rejected move")
	}
}

func TestWriteGradientSharesTheMean(t *testing.T) {
	// Positions written: 0, 1, 0. Cell 0 averages weights 0 and 2.
	weights := []float64{4, 8, 8}
	moves := []int{1, -1, 1}
	target := []byte{20, 30}
	_, grad, tape, err := lossAndWriteGrad(weights, moves, target, 4, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, dmu0 := symbolLoss(6, 4, 20)
	_, dmu1 := symbolLoss(8, 4, 30)
	if math.Abs(grad[0]-dmu0/2) > 1e-9 || math.Abs(grad[2]-dmu0/2) > 1e-9 {
		t.Fatalf("shares %.6f %.6f want %.6f", grad[0], grad[2], dmu0/2)
	}
	if math.Abs(grad[1]-dmu1) > 1e-9 {
		t.Fatalf("grad %.6f dmu %.6f", grad[1], dmu1)
	}
	if tape != string([]byte{nearestByte(6), nearestByte(8)}) {
		t.Fatalf("tape %q", tape)
	}
	// A write on cell -1 is outside the target window.
	off := []float64{6, 100}
	_, grad, _, err = lossAndWriteGrad(off, []int{-1, 1}, []byte{20}, 4, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, dmu := symbolLoss(6, 4, 20)
	if math.Abs(grad[0]-dmu) > 1e-9 || grad[1] != 0 {
		t.Fatalf("window grad %.6f off-window %.6f", grad[0], grad[1])
	}
}

func TestTwoWritePartialMatchesFiniteDifference(t *testing.T) {
	weights := []float64{4, 8, 8}
	moves := []int{1, -1, 1}
	target := []byte{20, 30}
	const tau = 4.0
	_, grad, _, err := lossAndWriteGrad(weights, moves, target, tau, 0)
	if err != nil {
		t.Fatal(err)
	}
	const eps = 1e-4
	for i := range weights {
		up := append([]float64(nil), weights...)
		dn := append([]float64(nil), weights...)
		up[i] += eps
		dn[i] -= eps
		lp, _, _, err := lossAndWriteGrad(up, moves, target, tau, 0)
		if err != nil {
			t.Fatal(err)
		}
		lm, _, _, err := lossAndWriteGrad(dn, moves, target, tau, 0)
		if err != nil {
			t.Fatal(err)
		}
		diff := (lp - lm) / (2 * eps) * float64(len(target))
		if math.Abs(grad[i]-diff) > 1e-4 {
			t.Fatalf("weight %d partial %.6f finite difference %.6f", i, grad[i], diff)
		}
	}
}

func TestMoveTreePartialMatchesFiniteDifference(t *testing.T) {
	weights := []float64{3, -2}
	logits := []float64{0.4, -0.2}
	target := []byte{20, 30}
	const tau = 4.0
	_, writeGrad, moveGrad, _, _, err := treeLoss(weights, logits, target, tau, 0)
	if err != nil {
		t.Fatal(err)
	}
	const eps = 1e-4
	total := func(w, a []float64) float64 {
		t.Helper()
		mean, _, _, _, _, err := treeLoss(w, a, target, tau, 0)
		if err != nil {
			t.Fatal(err)
		}
		return mean * float64(len(target))
	}
	for i := range weights {
		up := append([]float64(nil), weights...)
		dn := append([]float64(nil), weights...)
		up[i] += eps
		dn[i] -= eps
		diff := (total(up, logits) - total(dn, logits)) / (2 * eps)
		if math.Abs(writeGrad[i]-diff) > 1e-3 {
			t.Fatalf("write %d partial %.6f finite difference %.6f", i, writeGrad[i], diff)
		}
	}
	for i := range logits {
		up := append([]float64(nil), logits...)
		dn := append([]float64(nil), logits...)
		up[i] += eps
		dn[i] -= eps
		diff := (total(weights, up) - total(weights, dn)) / (2 * eps)
		if math.Abs(moveGrad[i]-diff) > 1e-3 {
			t.Fatalf("move %d partial %.6f finite difference %.6f", i, moveGrad[i], diff)
		}
	}
	if math.Abs(moveGrad[len(moveGrad)-1]) > 1e-9 {
		t.Fatalf("last move partial %.6f", moveGrad[len(moveGrad)-1])
	}
}

func TestMoveTreeAtZeroHasNoDirectionPartial(t *testing.T) {
	weights := []float64{0, 0}
	logits := []float64{0, 0}
	_, _, moveGrad, mode, _, err := treeLoss(weights, logits, []byte("AB"), 8, 0)
	if err != nil {
		t.Fatal(err)
	}
	if mode != "RR" {
		t.Fatalf("mode %s", mode)
	}
	for i, g := range moveGrad {
		if math.Abs(g) > 1e-9 {
			t.Fatalf("move %d partial %.6f", i, g)
		}
	}
}

func TestTapeCellsRoundsTheFraction(t *testing.T) {
	if tapeCells(12, 0.8) != 10 || tapeCells(16, 0.8) != 13 || tapeCells(2, 0.8) != 2 || tapeCells(1, 0.8) != 1 {
		t.Fatalf("cells %d %d %d %d", tapeCells(12, 0.8), tapeCells(16, 0.8), tapeCells(2, 0.8), tapeCells(1, 0.8))
	}
}

func TestWrappedWriteMatchesFiniteDifference(t *testing.T) {
	// Three steps on two cells. Positions are 0, 1, 0. The third emission
	// averages the first and third weights.
	weights := []float64{4, 8, 6}
	moves := []int{1, 1, 1}
	target := []byte{20, 30, 40}
	const tau = 4.0
	const cells = 2
	_, grad, tape, err := lossAndWriteGrad(weights, moves, target, tau, cells)
	if err != nil {
		t.Fatal(err)
	}
	_, d0 := symbolLoss(4, tau, 20)
	_, d1 := symbolLoss(8, tau, 30)
	_, d2 := symbolLoss(5, tau, 40)
	want := []float64{d0 + d2/2, d1, d2 / 2}
	for i := range want {
		if math.Abs(grad[i]-want[i]) > 1e-9 {
			t.Fatalf("grad %d %.6f want %.6f", i, grad[i], want[i])
		}
	}
	if tape != string([]byte{nearestByte(4), nearestByte(8), nearestByte(5)}) {
		t.Fatalf("tape %q", tape)
	}
	const eps = 1e-4
	for i := range weights {
		up := append([]float64(nil), weights...)
		dn := append([]float64(nil), weights...)
		up[i] += eps
		dn[i] -= eps
		lp, _, _, err := lossAndWriteGrad(up, moves, target, tau, cells)
		if err != nil {
			t.Fatal(err)
		}
		lm, _, _, err := lossAndWriteGrad(dn, moves, target, tau, cells)
		if err != nil {
			t.Fatal(err)
		}
		diff := (lp - lm) / (2 * eps) * float64(len(target))
		if math.Abs(grad[i]-diff) > 1e-3 {
			t.Fatalf("weight %d partial %.6f finite difference %.6f", i, grad[i], diff)
		}
	}
}

func TestGaussLearnsHelloWorld(t *testing.T) {
	err := learnGaussText(gaussLearnConfig{
		Text:  "Hello World!",
		Iters: 32,
		Temp:  8,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestOneUnrollRejectsMoreThanABlock(t *testing.T) {
	n := maxMoveUnroll + 1
	weights := make([]float64, n)
	logits := make([]float64, n)
	target := make([]byte, n)
	_, _, _, _, _, err := treeLoss(weights, logits, target, 8, 0)
	if err == nil {
		t.Fatal("expected one unroll to reject more than a block")
	}
}

func TestGaussLearnsTextLongerThanOneBlock(t *testing.T) {
	// 18 symbols stay one machine. The full move tree stops at 16.
	err := learnGaussText(gaussLearnConfig{
		Text:  "Hello World! Again",
		Iters: 16,
		Size:  1,
		Temp:  8,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestGaussLearnsHelloWorldBothWays(t *testing.T) {
	// Eleven steps right reach cell 11. Eleven steps left return through
	// cells 10..0. The last step does not write. Both directions occur,
	// and every target cell is written at least once.
	moves := "RRRRRRRRRRRLLLLLLLLLLLR"
	parsed, err := parseMoves(moves, 0)
	if err != nil {
		t.Fatal(err)
	}
	weights := make([]float64, len(parsed))
	cells, _, err := runLearn(weights, parsed)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len("Hello World!"); i++ {
		if cells[i] == nil || cells[i].n == 0 {
			t.Fatalf("cell %d unvisited", i)
		}
	}
	err = learnGaussText(gaussLearnConfig{
		Text:  "Hello World!",
		Moves: moves,
		Iters: 16,
		Size:  1,
		Temp:  8,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestDiscreteWriteEmitsTheCharacter(t *testing.T) {
	w0 := []float64{72, 101, 108}
	w1 := []float64{33, 63, 88}
	m0 := []int{1, 1, 1}
	m1 := []int{-1, -1, -1}
	writes, tape := runDiscrete(w0, w1, m0, m1, 0)
	if writes != "Hel" || tape != "Hel" {
		t.Fatalf("writes %q tape %q", writes, tape)
	}
	// Two cells. The third step wraps onto cell 0, reads H, and takes rule 1.
	writes, tape = runDiscrete(w0, w1, m0, []int{1, 1, -1}, 2)
	if writes != "HeX?" || tape != "X?" {
		t.Fatalf("wrapped writes %q tape %q", writes, tape)
	}
	table := machineTable(w0, w1, m0, m1)
	if table[1][0].WriteSymbol != 'H' || table[1][1].WriteSymbol != '!' || table[1][0].MoveDir != 1 || table[1][1].MoveDir != -1 {
		t.Fatalf("pair %+v %+v", table[1][0], table[1][1])
	}
	if table[1][0].NextState == table[1][1].NextState || table[2][0].NextState == table[2][1].NextState {
		t.Fatal("read 0 and read 1 go to the same state")
	}
	one := machineTable([]float64{'A'}, []float64{'B'}, []int{1}, []int{-1})
	if one[1][0].NextState != 0 || one[1][1].NextState != 1 || one[1][0].WriteSymbol != 'A' || one[1][1].WriteSymbol != 'B' {
		t.Fatalf("one state %+v %+v", one[1][0], one[1][1])
	}
	states := discreteStates(w0, w1, m0, m1)
	if formatStates(states) != "1 2 3 0" {
		t.Fatalf("states %v", states)
	}
	// Step back onto H. Rule 1 returns to the previous state.
	states = discreteStates(w0, w1, []int{1, -1, 1}, []int{-1, 1, -1})
	if formatStates(states) != "1 2 3 2" {
		t.Fatalf("states %v", states)
	}
	zeros := make([]int, 64)
	long := discreteStates(make([]float64, 64), make([]float64, 64), zeros, zeros)
	if len(long) != 65 || long[0] != 1 || long[63] != 64 || long[64] != 0 {
		t.Fatalf("long states %d", len(long))
	}
	if machineTable(nil, nil, nil, nil) != nil {
		t.Fatal("empty machine")
	}
	if machineTable([]float64{1}, nil, []int{1}, []int{1}) != nil {
		t.Fatal("length mismatch")
	}
	if machineTable(make([]float64, 64), make([]float64, 64), zeros, zeros) == nil {
		t.Fatal("64 states still run")
	}
}

func TestLogDiscreteRunsTheConvertedMachine(t *testing.T) {
	w0 := []float64{72, 101, 108}
	w1 := []float64{33, 63, 88}
	m0 := []int{1, 1, 1}
	m1 := []int{-1, -1, -1}
	var lines []string
	cfg := gaussLearnConfig{Log: func(s string) { lines = append(lines, s) }}
	got := cfg.logDiscrete(w0, w1, m0, m1, 0)
	writes, tape := runDiscrete(w0, w1, m0, m1, 0)
	if got != writes || got != "Hel" {
		t.Fatalf("writes %q", got)
	}
	text := strings.Join(lines, "\n")
	states := formatStateChars(discreteStates(w0, w1, m0, m1), ruleSymbols(w0))
	if !strings.Contains(text, "writes "+strconv.Quote(writes)) || !strings.Contains(text, "tape "+strconv.Quote(tape)) || !strings.Contains(text, "states "+states) || strings.Contains(text, "states 1 ") || !strings.Contains(text, "write 'H'") || !strings.Contains(text, "write '!'") {
		t.Fatalf("log\n%s", text)
	}
	lines = nil
	got = cfg.logDiscrete(w0, w1, m0, []int{1, 1, -1}, 2)
	if got != "HeX?" || !strings.Contains(strings.Join(lines, "\n"), "tape "+strconv.Quote("X?")) {
		t.Fatalf("wrapped writes %q\n%s", got, strings.Join(lines, "\n"))
	}
}

func TestLearnedMachinePrintsTheCharacter(t *testing.T) {
	var lines []string
	err := learnGaussText(gaussLearnConfig{
		Text:  "A",
		Moves: "R",
		Iters: 8,
		Size:  1,
		Temp:  8,
		Log:   func(s string) { lines = append(lines, s) },
	})
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(lines, "\n")
	if !strings.Contains(got, "discrete machine") || !strings.Contains(got, "state 'A'") || !strings.Contains(got, "next '\\x00'") || !strings.Contains(got, "write 'A'") || !strings.Contains(got, "writes \"A\"") || !strings.Contains(got, "tape \"A\"") || !strings.Contains(got, "states \"A\\x00\"") || !strings.Contains(got, "text \"A\"") {
		t.Fatalf("log\n%s", got)
	}
}

func TestRuleWriteEmitsTheWeightOnARevisit(t *testing.T) {
	// Three steps on two cells land on 0, 1, 0. The third step reads a
	// non-zero byte and uses the read-1 weight. The emission is that weight.
	w0 := []float64{4, 8, 0}
	w1 := []float64{0, 0, 6}
	moves := []int{1, 1, 1}
	target := []byte{20, 30, 40}
	const tau = 4.0
	const cells = 2
	_, g0, g1, tape, fired0, fired1, err := symbolPathRules(w0, w1, moves, moves, target, tau, cells)
	if err != nil {
		t.Fatal(err)
	}
	_, d0 := symbolLoss(4, tau, 20)
	_, d1 := symbolLoss(8, tau, 30)
	_, d2 := symbolLoss(6, tau, 40)
	if math.Abs(g0[0]-d0) > 1e-9 || math.Abs(g0[1]-d1) > 1e-9 || g0[2] != 0 {
		t.Fatalf("g0 %.6f %.6f %.6f", g0[0], g0[1], g0[2])
	}
	if g1[0] != 0 || g1[1] != 0 || math.Abs(g1[2]-d2) > 1e-9 {
		t.Fatalf("g1 %.6f %.6f %.6f", g1[0], g1[1], g1[2])
	}
	if tape != string([]byte{nearestByte(4), nearestByte(8), nearestByte(6)}) {
		t.Fatalf("tape %q", tape)
	}
	if !fired0[0] || !fired0[1] || fired0[2] || !fired1[2] {
		t.Fatalf("fired %v %v", fired0, fired1)
	}
	const eps = 1e-4
	total := func(a, b []float64) float64 {
		t.Helper()
		sum, _, _, _, _, _, err := symbolPathRules(a, b, moves, moves, target, tau, cells)
		if err != nil {
			t.Fatal(err)
		}
		return sum
	}
	for i := range w0 {
		up := append([]float64(nil), w0...)
		dn := append([]float64(nil), w0...)
		up[i] += eps
		dn[i] -= eps
		diff := (total(up, w1) - total(dn, w1)) / (2 * eps)
		if math.Abs(g0[i]-diff) > 1e-3 {
			t.Fatalf("w0 %d partial %.6f finite difference %.6f", i, g0[i], diff)
		}
		up = append([]float64(nil), w1...)
		dn = append([]float64(nil), w1...)
		up[i] += eps
		dn[i] -= eps
		diff = (total(w0, up) - total(w0, dn)) / (2 * eps)
		if math.Abs(g1[i]-diff) > 1e-3 {
			t.Fatalf("w1 %d partial %.6f finite difference %.6f", i, g1[i], diff)
		}
	}
}

func TestMoveBranchAtZeroHasNoDirectionPartial(t *testing.T) {
	_, _, _, dg0, dg1, mode, _, _, _, _, err := moveBranchLoss(
		[]float64{0, 0}, []float64{0, 0}, []float64{0, 0}, []float64{0, 0}, []byte("AB"), 8, 0)
	if err != nil {
		t.Fatal(err)
	}
	if mode != "RR" {
		t.Fatalf("mode %s", mode)
	}
	for i := range dg0 {
		if math.Abs(dg0[i]) > 1e-9 || math.Abs(dg1[i]) > 1e-9 {
			t.Fatalf("move %d partial %.6f %.6f", i, dg0[i], dg1[i])
		}
	}
}

func TestMoveBranchFlipsOneUndecidedLogit(t *testing.T) {
	w0 := []float64{10, 40}
	w1 := []float64{0, 0}
	a0 := []float64{0.2, 0.4}
	a1 := []float64{0, 0}
	target := []byte{10, 40}
	const tau = 8.0
	_, _, _, dg0, dg1, _, _, _, fired0, _, err := moveBranchLoss(w0, w1, a0, a1, target, tau, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !fired0[0] || !fired0[1] {
		t.Fatal("rule 0 should fire")
	}
	if math.Abs(dg0[1]) > 1e-12 {
		t.Fatalf("last move partial %.6f", dg0[1])
	}
	if math.Abs(dg1[0]) > 1e-9 {
		t.Fatalf("unread rule partial %.6f", dg1[0])
	}
	m0 := modeMoves(a0)
	m1 := modeMoves(a1)
	base, _, _, _, _, _, err := symbolPathRules(w0, w1, m0, m1, target, tau, 0)
	if err != nil {
		t.Fatal(err)
	}
	alt := append([]int(nil), m0...)
	alt[0] = -m0[0]
	flipped, _, _, _, _, _, err := symbolPathRules(w0, w1, alt, m1, target, tau, 0)
	if err != nil {
		t.Fatal(err)
	}
	p := logistic(a0[0])
	lRight, lLeft := base, flipped
	if m0[0] < 0 {
		lRight, lLeft = flipped, base
	}
	want := (lRight - lLeft) * p * (1 - p)
	if math.Abs(dg0[0]-want) > 1e-6 {
		t.Fatalf("partial %.6f want %.6f", dg0[0], want)
	}
	a0[0] = 5
	_, _, _, dg0, _, _, _, _, _, _, err = moveBranchLoss(w0, w1, a0, a1, target, tau, 0)
	if err != nil {
		t.Fatal(err)
	}
	if dg0[0] != 0 {
		t.Fatalf("decided partial %.6f", dg0[0])
	}
}

func TestRecordSymbolsKeepsTheBytes(t *testing.T) {
	for _, s := range []string{"Hello World!", "Abcdefg", "Abcdefgh"} {
		if recordSymbols(s) != s {
			t.Fatalf("text %q", s)
		}
	}
	weights := []float64{72, 101, 108, 108, 111}
	moves := []int{1, 1, 1, 1, 1}
	got := recordSymbols(emissionText(weights, moves, 0))
	if got != "Hello" {
		t.Fatalf("emission %q", got)
	}
}
