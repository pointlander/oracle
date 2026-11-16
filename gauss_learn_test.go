// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
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
	_, grad, tape, err := lossAndWriteGrad(weights, moves, target, 4)
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
	_, grad, _, err = lossAndWriteGrad(off, []int{-1, 1}, []byte{20}, 4)
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
	_, grad, _, err := lossAndWriteGrad(weights, moves, target, tau)
	if err != nil {
		t.Fatal(err)
	}
	const eps = 1e-4
	for i := range weights {
		up := append([]float64(nil), weights...)
		dn := append([]float64(nil), weights...)
		up[i] += eps
		dn[i] -= eps
		lp, _, _, err := lossAndWriteGrad(up, moves, target, tau)
		if err != nil {
			t.Fatal(err)
		}
		lm, _, _, err := lossAndWriteGrad(dn, moves, target, tau)
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
	_, writeGrad, moveGrad, _, _, err := treeLoss(weights, logits, target, tau)
	if err != nil {
		t.Fatal(err)
	}
	const eps = 1e-4
	total := func(w, a []float64) float64 {
		t.Helper()
		mean, _, _, _, _, err := treeLoss(w, a, target, tau)
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
	_, _, moveGrad, mode, _, err := treeLoss(weights, logits, []byte("AB"), 8)
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

func TestGaussLearnsHelloWorld(t *testing.T) {
	err := learnGaussText(gaussLearnConfig{
		Text:  "Hello World!",
		Iters: 16,
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
	_, _, _, _, _, err := treeLoss(weights, logits, target, 8)
	if err == nil {
		t.Fatal("expected one unroll to reject more than a block")
	}
}

func TestGaussLearnsTextLongerThanOneBlock(t *testing.T) {
	// 18 symbols: one full block of 16 and a second block of 2.
	err := learnGaussText(gaussLearnConfig{
		Text:  "Hello World! Again",
		Iters: 16,
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
		Temp:  8,
	})
	if err != nil {
		t.Fatal(err)
	}
}
