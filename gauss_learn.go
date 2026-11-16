// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"strconv"
)

// symbolLoss is the cross-entropy of a target byte under a Gaussian softmax.
// Logits are -(b-μ)^2 / (2 τ^2) for byte codes b = 0..255. The partial with
// respect to the cell mean is (E[b] - target) / τ^2. A hard nearest-byte read
// is constant between integers, so this softmax is the differentiable symbol.
func symbolLoss(mu, tau float64, target int) (nll, dmu float64) {
	scale := 1 / (2 * tau * tau)
	var z [256]float64
	maxZ := math.Inf(-1)
	for b := 0; b < 256; b++ {
		d := float64(b) - mu
		z[b] = -d * d * scale
		if z[b] > maxZ {
			maxZ = z[b]
		}
	}
	sum := 0.0
	moment := 0.0
	for b := 0; b < 256; b++ {
		e := math.Exp(z[b] - maxZ)
		sum += e
		moment += e * float64(b)
	}
	expect := moment / sum
	nll = -z[target] + maxZ + math.Log(sum)
	dmu = (expect - float64(target)) / (tau * tau)
	return nll, dmu
}

func nearestByte(mu float64) byte {
	r := math.Round(mu)
	if r < 0 {
		r = 0
	}
	if r > 255 {
		r = 255
	}
	return byte(r)
}

// writeRight runs a 256-symbol machine with one state per target byte.
// Every read symbol in a state shares that state's write weight. The head
// writes the weight into the current cell and moves right. The last state halts.
// Each cell mean is the average of the weights written there.
func writeRight(weights []float64) []gaussCell {
	cells := make([]gaussCell, len(weights))
	for state := 1; state <= len(weights); state++ {
		head := state - 1
		cells[head].observe(weights[head])
	}
	return cells
}

// lossAndWriteGrad returns the mean symbol cross-entropy and the partial of
// the total loss with respect to each write weight. A cell written n times
// gives each of those weights the share (1/n) ∂L/∂μ.
func lossAndWriteGrad(weights []float64, target []byte, tau float64) (float64, []float64) {
	cells := writeRight(weights)
	grad := make([]float64, len(weights))
	loss := 0.0
	for i, y := range target {
		nll, dmu := symbolLoss(cells[i].mean, tau, int(y))
		loss += nll
		share := dmu / float64(cells[i].n)
		grad[i] += share
	}
	return loss / float64(len(target)), grad
}

func decodeCells(cells []gaussCell) string {
	out := make([]byte, len(cells))
	for i, c := range cells {
		out[i] = nearestByte(c.mean)
	}
	return string(out)
}

// gaussLearnConfig trains write weights of a right-moving 256-symbol machine.
// LR <= 0 uses Temp², which cancels the 1/τ² in the symbol partial.
type gaussLearnConfig struct {
	Text  string
	Iters int
	LR    float64
	Temp  float64
	Log   func(string)
}

func (cfg *gaussLearnConfig) apply() error {
	if cfg.Text == "" {
		return fmt.Errorf("text must be non-empty")
	}
	if cfg.Iters <= 0 {
		cfg.Iters = 16
	}
	if cfg.Temp <= 0 {
		return fmt.Errorf("temperature must be positive")
	}
	if cfg.LR <= 0 {
		cfg.LR = cfg.Temp * cfg.Temp
	}
	return nil
}

func (cfg gaussLearnConfig) logf(format string, args ...any) {
	if cfg.Log != nil {
		cfg.Log(fmt.Sprintf(format, args...))
	}
}

// learnGaussText follows ∂L/∂write for the bytes of cfg.Text.
// The learned tape is the nearest byte to each cell mean.
func learnGaussText(cfg gaussLearnConfig) error {
	if err := cfg.apply(); err != nil {
		return err
	}
	target := []byte(cfg.Text)
	weights := make([]float64, len(target))
	cfg.logf("256-symbol gaussian machine")
	cfg.logf("  target %s", strconv.Quote(cfg.Text))
	cfg.logf("  %d states, alphabet 256, move right, one write weight per state", len(target))
	cfg.logf("  loss is the cross-entropy of a Gaussian softmax over byte codes")
	cfg.logf("  each step subtracts the learning rate times the partial of that loss")
	var loss float64
	tape := ""
	for iter := 0; iter <= cfg.Iters; iter++ {
		var grad []float64
		loss, grad = lossAndWriteGrad(weights, target, cfg.Temp)
		tape = decodeCells(writeRight(weights))
		if iter == 0 || iter == cfg.Iters || iter%4 == 0 || tape == cfg.Text {
			cfg.logf("iter %d  loss %.6f  tape %s", iter, loss, strconv.Quote(tape))
		}
		if tape == cfg.Text && iter > 0 {
			break
		}
		if iter == cfg.Iters {
			break
		}
		for i := range weights {
			weights[i] -= cfg.LR * grad[i]
		}
	}
	if tape != cfg.Text {
		return fmt.Errorf("tape %s after %d steps", strconv.Quote(tape), cfg.Iters)
	}
	return nil
}
