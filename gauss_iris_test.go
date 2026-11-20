// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"strings"
	"testing"
)

func TestClassLossMatchesFiniteDifference(t *testing.T) {
	const mu = 0.4
	const tau = 0.5
	const y = 2
	_, analytic := classLoss(mu, tau, y, 3)
	const eps = 1e-4
	lp, _ := classLoss(mu+eps, tau, y, 3)
	lm, _ := classLoss(mu-eps, tau, y, 3)
	diff := (lp - lm) / (2 * eps)
	if math.Abs(analytic-diff) > 1e-4 {
		t.Fatalf("partial %.6f finite difference %.6f", analytic, diff)
	}
}

func TestIrisWeightPartialMatchesFiniteDifference(t *testing.T) {
	rows := []irisRow{{
		x: [irisFeatures]float64{1, -1, 0.5, 2},
		y: 1,
	}}
	// Measurements 1, -1, 0.5, 2. Byte 0 selects write 0.
	const bias = 10.0
	w0 := []float64{9, 9, 9, 9}
	w1 := []float64{0.2, -0.3, 0.4, 0.1}
	const tau = irisTau
	gb, g0, g1, _ := irisGrad(bias, w0, w1, rows, tau, 3)
	// Fired rules are 1, 0, 1, 1. The other write of each step is unread.
	if g0[0] != 0 || g1[1] != 0 || g0[2] != 0 || g0[3] != 0 {
		t.Fatalf("unread g0 %v g1 %v", g0, g1)
	}
	const eps = 1e-4
	_, _, _, lp := irisGrad(bias+eps, w0, w1, rows, tau, 3)
	_, _, _, lm := irisGrad(bias-eps, w0, w1, rows, tau, 3)
	diff := (lp - lm) / (2 * eps)
	if math.Abs(gb-diff) > 1e-4 {
		t.Fatalf("bias partial %.6f finite difference %.6f", gb, diff)
	}
	up0, dn0 := append([]float64(nil), w0...), append([]float64(nil), w0...)
	up0[1] += eps
	dn0[1] -= eps
	_, _, _, lp = irisGrad(bias, up0, w1, rows, tau, 3)
	_, _, _, lm = irisGrad(bias, dn0, w1, rows, tau, 3)
	diff = (lp - lm) / (2 * eps)
	if math.Abs(g0[1]-diff) > 1e-4 {
		t.Fatalf("weight partial %.6f finite difference %.6f", g0[1], diff)
	}
}

func TestIrisProgramReadsTheTape(t *testing.T) {
	const bias = 10.0
	x := [irisFeatures]float64{1, -1, 0.5, 2}
	tape := irisInscribe(bias, x)
	if len(tape) != irisTapeCells || tape[0] != bias || tape[1] != 1 || tape[3] != 0.5 {
		t.Fatalf("tape %v", tape)
	}
	w0 := []float64{9, 9, 9, 9}
	w1 := []float64{0.2, -0.3, 0.4, 0.1}
	sum, rule := irisProgram(bias, w0, w1, x)
	// Measurement bytes are 1, 0, 1, 2, so the fired writes are w1, w0, w1, w1.
	want := bias + x[0] + x[1] + x[2] + x[3] + w1[0] + w0[1] + w1[2] + w1[3]
	if math.Abs(sum-want) > 1e-9 {
		t.Fatalf("sum %.6f want %.6f", sum, want)
	}
	got := []int{1, 0, 1, 1}
	for i := range got {
		if rule[i] != got[i] {
			t.Fatalf("rule %v", rule)
		}
	}
}

func TestIrisLossIsTheTextSymbol(t *testing.T) {
	rows := []irisRow{{y: 1}}
	w0 := make([]float64, irisFeatures)
	w1 := make([]float64, irisFeatures)
	_, _, _, loss := irisGrad(0, w0, w1, rows, irisTau, 3)
	nll, _ := symbolLoss(0, irisTau, irisClassByte(1, 3))
	if math.Abs(loss-nll) > 1e-9 {
		t.Fatalf("loss %.6f symbol %.6f", loss, nll)
	}
	if irisClassByte(0, 3) != 112 || irisClassByte(1, 3) != 128 || irisClassByte(2, 3) != 144 {
		t.Fatalf("bytes %d %d %d", irisClassByte(0, 3), irisClassByte(1, 3), irisClassByte(2, 3))
	}
	if irisClassOf(128, 3) != 1 || irisClassOf(144, 3) != 2 || irisClassOf(120, 3) != 0 {
		t.Fatalf("class of 128 %d", irisClassOf(128, 3))
	}
}

func TestParseIrisSkipsBlankLines(t *testing.T) {
	text := "1,2,3,4,Iris-setosa\n\n5,6,7,8,Iris-versicolor\n"
	rows, names, err := parseIris(text)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || len(names) != 2 || rows[1].y != 1 || rows[1].x[3] != 8 {
		t.Fatalf("rows %d names %v", len(rows), names)
	}
}

func TestIrisSeparatesOneFeature(t *testing.T) {
	// Class 0 sits on the left of the first feature. Class 1 sits on the right.
	var train, test []irisRow
	for i := 0; i < 8; i++ {
		train = append(train, irisRow{x: [irisFeatures]float64{-2, 0, 0, 0}, y: 0})
		train = append(train, irisRow{x: [irisFeatures]float64{2, 0, 0, 0}, y: 1})
		test = append(test, irisRow{x: [irisFeatures]float64{-2, 0, 0, 0}, y: 0})
		test = append(test, irisRow{x: [irisFeatures]float64{2, 0, 0, 0}, y: 1})
	}
	standardizeIris(train, test)
	bias, w0, w1, err := fitIris(train, test, 2, 32, irisTau, nil)
	if err != nil {
		t.Fatal(err)
	}
	if irisCorrect(test, bias, w0, w1, 2) != len(test) {
		t.Fatalf("test %d/%d", irisCorrect(test, bias, w0, w1, 2), len(test))
	}
}

func TestGaussClassifiesIris(t *testing.T) {
	var lines []string
	err := learnIris(irisConfig{
		Path:  "iris.zip",
		Iters: 32,
		Seed:  1,
		Log:   func(s string) { lines = append(lines, s) },
	})
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(lines, "\n")
	if !strings.Contains(got, "Iris-setosa") || !strings.Contains(got, "Iris-versicolor") || !strings.Contains(got, "Iris-virginica") {
		t.Fatalf("log\n%s", got)
	}
	rows, _, err := loadIris("iris.zip")
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 150 {
		t.Fatalf("rows %d", len(rows))
	}
	train, test, err := splitIris(rows, 1)
	if err != nil {
		t.Fatal(err)
	}
	standardizeIris(train, test)
	bias, w0, w1, err := fitIris(train, test, 3, 32, irisTau, nil)
	if err != nil {
		t.Fatal(err)
	}
	hit := irisCorrect(test, bias, w0, w1, 3)
	if hit*10 < len(test)*9 {
		t.Fatalf("test %d/%d", hit, len(test))
	}
}
