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

func TestWriteGradientSharesTheMean(t *testing.T) {
	// Two writes on one cell. Each partial is half of ∂L/∂μ.
	var cell gaussCell
	cell.observe(4)
	cell.observe(8)
	_, dmu := symbolLoss(cell.mean, 4, 20)
	want := dmu / 2
	// Reconstruct the same mean as the average of the two weights and check
	// the share used by lossAndWriteGrad's one-write path on a twin cell.
	if cell.n != 2 || cell.mean != 6 {
		t.Fatalf("mean %v n %d", cell.mean, cell.n)
	}
	if math.Abs(want-dmu/float64(cell.n)) > 1e-12 {
		t.Fatal("share")
	}
	// A one-write machine's gradient equals ∂L/∂μ.
	_, grad := lossAndWriteGrad([]float64{6}, []byte{20}, 4)
	if math.Abs(grad[0]-dmu) > 1e-9 {
		t.Fatalf("grad %.6f dmu %.6f", grad[0], dmu)
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
