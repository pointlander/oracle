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

// parseMoves reads a string of left and right steps. L, l, <, and - move left.
// R, r, >, and + move right. An empty string is n right steps, one per target byte.
func parseMoves(s string, n int) ([]int, error) {
	if s == "" {
		if n < 1 {
			return nil, fmt.Errorf("text must be non-empty")
		}
		out := make([]int, n)
		for i := range out {
			out[i] = 1
		}
		return out, nil
	}
	out := make([]int, 0, len(s))
	for _, r := range s {
		switch r {
		case 'R', 'r', '>', '+':
			out = append(out, 1)
		case 'L', 'l', '<', '-':
			out = append(out, -1)
		default:
			return nil, fmt.Errorf("move %q is not L or R", string(r))
		}
	}
	return out, nil
}

func formatMoves(moves []int) string {
	b := make([]byte, len(moves))
	for i, m := range moves {
		b[i] = 'R'
		if m < 0 {
			b[i] = 'L'
		}
	}
	return string(b)
}

// runLearn runs a 256-symbol machine with one write weight per state.
// Every read symbol in a state shares that state's write weight. The head
// starts on cell 0, writes the weight, then moves one cell left or right.
// writers lists the states that wrote each cell, in order.
func runLearn(weights []float64, moves []int) (map[int]*gaussCell, map[int][]int, error) {
	if len(weights) != len(moves) {
		return nil, nil, fmt.Errorf("weights and moves differ in length")
	}
	if len(weights) == 0 {
		return nil, nil, fmt.Errorf("moves must be non-empty")
	}
	cells := make(map[int]*gaussCell)
	writers := make(map[int][]int)
	head := 0
	for state, w := range weights {
		move := moves[state]
		if move != -1 && move != 1 {
			return nil, nil, fmt.Errorf("move %d is not -1 or 1", move)
		}
		cell := cells[head]
		if cell == nil {
			cell = &gaussCell{}
			cells[head] = cell
		}
		cell.observe(w)
		writers[head] = append(writers[head], state)
		head += move
	}
	return cells, writers, nil
}

// lossAndWriteGrad returns the mean symbol cross-entropy, the partial of the
// total loss with respect to each write weight, and the nearest-byte tape on
// cells 0..len(target)-1. A cell written n times gives each of those weights
// the share (1/n) ∂L/∂μ. A write outside that window has partial 0. An
// unwritten target cell is read as mean 0 and sends no partial.
func lossAndWriteGrad(weights []float64, moves []int, target []byte, tau float64) (float64, []float64, string, error) {
	cells, writers, err := runLearn(weights, moves)
	if err != nil {
		return 0, nil, "", err
	}
	grad := make([]float64, len(weights))
	out := make([]byte, len(target))
	loss := 0.0
	for i, y := range target {
		mu := 0.0
		n := 0
		if c := cells[i]; c != nil && c.n > 0 {
			mu = c.mean
			n = c.n
		}
		nll, dmu := symbolLoss(mu, tau, int(y))
		loss += nll
		out[i] = nearestByte(mu)
		if n == 0 {
			continue
		}
		share := dmu / float64(n)
		for _, state := range writers[i] {
			grad[state] += share
		}
	}
	return loss / float64(len(target)), grad, string(out), nil
}

// gaussLearnConfig trains write weights of a 256-symbol machine.
// Moves is a string of L and R. Empty is one right move per target byte.
// LR <= 0 uses Temp², which cancels the 1/τ² in the symbol partial.
type gaussLearnConfig struct {
	Text  string
	Moves string
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
	moves, err := parseMoves(cfg.Moves, len(target))
	if err != nil {
		return err
	}
	weights := make([]float64, len(moves))
	cfg.logf("256-symbol gaussian machine")
	cfg.logf("  target %s", strconv.Quote(cfg.Text))
	cfg.logf("  %d states, alphabet 256, moves %s", len(moves), formatMoves(moves))
	cfg.logf("  the head starts on cell 0, writes, then moves left or right")
	cfg.logf("  loss is the cross-entropy of a Gaussian softmax over byte codes")
	cfg.logf("  each step subtracts the learning rate times the partial of that loss")
	var loss float64
	tape := ""
	for iter := 0; iter <= cfg.Iters; iter++ {
		var grad []float64
		loss, grad, tape, err = lossAndWriteGrad(weights, moves, target, cfg.Temp)
		if err != nil {
			return err
		}
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
