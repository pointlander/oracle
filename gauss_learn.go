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

// maxMoveUnroll is the deepest left/right tree expanded while learning moves.
// A text of this length has 2^maxMoveUnroll root-to-leaf paths.
const maxMoveUnroll = 16

// defaultMoveLR is the step size on move logits. It is separate from the
// write learning rate, which is scaled to cancel 1/τ².
const defaultMoveLR = 1

// gaussLearnConfig trains write weights of a 256-symbol machine.
// Moves is a string of L and R. Empty learns a move logit per state by
// unrolling both branches. LR <= 0 uses Temp², which cancels the 1/τ² in
// the symbol partial. MoveLR <= 0 uses defaultMoveLR.
type gaussLearnConfig struct {
	Text   string
	Moves  string
	Iters  int
	LR     float64
	MoveLR float64
	Temp   float64
	Log    func(string)
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
	if cfg.MoveLR <= 0 {
		cfg.MoveLR = defaultMoveLR
	}
	return nil
}

func (cfg gaussLearnConfig) logf(format string, args ...any) {
	if cfg.Log != nil {
		cfg.Log(fmt.Sprintf(format, args...))
	}
}

// pathSum is the total symbol cross-entropy of one root-to-leaf path, and it
// adds that path's write partials, already multiplied by prob, into writeGrad.
func pathSum(weights []float64, moves []int, target []byte, tau, prob float64, writeGrad []float64) float64 {
	pos := make([]int, len(weights))
	head := 0
	for t := range weights {
		pos[t] = head
		head += moves[t]
	}
	sum := 0.0
	for i, y := range target {
		mu := 0.0
		n := 0
		for t := range pos {
			if pos[t] == i {
				mu += weights[t]
				n++
			}
		}
		if n > 0 {
			mu /= float64(n)
		}
		nll, dmu := symbolLoss(mu, tau, int(y))
		sum += nll
		if n == 0 {
			continue
		}
		share := prob * dmu / float64(n)
		for t := range pos {
			if pos[t] == i {
				writeGrad[t] += share
			}
		}
	}
	return sum
}

// treeLoss unrolls the machine the way an RNN is unrolled, except each step
// branches left and right. A path probability is the product of its branch
// probabilities. The loss is the expected total symbol cross-entropy. The
// write partial is the probability-weighted average of the path partials. The
// move-logit partial at a step is (E[loss|right] - E[loss|left]) p (1-p).
// The move after the last write does not change the tape, so its partial is 0.
func treeLoss(weights, logits []float64, target []byte, tau float64) (mean float64, writeGrad, moveGrad []float64, mode string, tape string, err error) {
	if len(weights) != len(logits) || len(weights) == 0 {
		return 0, nil, nil, "", "", fmt.Errorf("weights and move logits differ in length")
	}
	if len(weights) > maxMoveUnroll {
		return 0, nil, nil, "", "", fmt.Errorf("move tree unrolls at most %d steps", maxMoveUnroll)
	}
	steps := len(weights)
	p := make([]float64, steps)
	for t, a := range logits {
		p[t] = logistic(a)
	}
	writeGrad = make([]float64, steps)
	rightLoss := make([]float64, steps)
	leftLoss := make([]float64, steps)
	rightMass := make([]float64, steps)
	leftMass := make([]float64, steps)
	moves := make([]int, steps)
	expect := 0.0
	nPath := 1 << steps
	for mask := 0; mask < nPath; mask++ {
		logp := 0.0
		dead := false
		for t := 0; t < steps; t++ {
			if mask>>t&1 == 1 {
				moves[t] = 1
				if p[t] == 0 {
					dead = true
					break
				}
				logp += math.Log(p[t])
			} else {
				moves[t] = -1
				if p[t] == 1 {
					dead = true
					break
				}
				logp += math.Log(1 - p[t])
			}
		}
		if dead || logp < -745 {
			continue
		}
		prob := math.Exp(logp)
		total := pathSum(weights, moves, target, tau, prob, writeGrad)
		expect += prob * total
		for t := 0; t < steps; t++ {
			if moves[t] > 0 {
				rightLoss[t] += prob * total
				rightMass[t] += prob
			} else {
				leftLoss[t] += prob * total
				leftMass[t] += prob
			}
		}
	}
	moveGrad = make([]float64, steps)
	for t := 0; t < steps; t++ {
		if rightMass[t] == 0 || leftMass[t] == 0 {
			continue
		}
		dLdp := rightLoss[t]/rightMass[t] - leftLoss[t]/leftMass[t]
		pt := p[t]
		moveGrad[t] = dLdp * pt * (1 - pt)
	}
	modeMoves := make([]int, steps)
	for t, a := range logits {
		modeMoves[t] = -1
		if a >= 0 {
			modeMoves[t] = 1
		}
	}
	_, _, tape, err = lossAndWriteGrad(weights, modeMoves, target, tau)
	if err != nil {
		return 0, nil, nil, "", "", err
	}
	n := float64(len(target))
	return expect / n, writeGrad, moveGrad, formatMoves(modeMoves), tape, nil
}

func formatProbs(logits []float64) string {
	b := make([]byte, 0, len(logits)*5)
	for i, a := range logits {
		if i > 0 {
			b = append(b, ' ')
		}
		b = append(b, fmt.Sprintf("%.2f", logistic(a))...)
	}
	return string(b)
}

// learnMoveTree follows the partials of the expected loss through the unrolled
// move tree. The reported tape is the nearest-byte read along the mode path.
// Steps other than the last must put at least probability 0.9 on that path's
// direction. The last move has partial 0.
func learnMoveTree(cfg gaussLearnConfig) error {
	target := []byte(cfg.Text)
	if len(target) > maxMoveUnroll {
		return fmt.Errorf("move tree unrolls at most %d steps", maxMoveUnroll)
	}
	weights := make([]float64, len(target))
	logits := make([]float64, len(target))
	cfg.logf("256-symbol gaussian machine")
	cfg.logf("  target %s", strconv.Quote(cfg.Text))
	cfg.logf("  %d states, alphabet 256, both moves unrolled at every state", len(target))
	cfg.logf("  loss is the expected cross-entropy over the move tree")
	cfg.logf("  each step follows the partial of that loss through the writes and the move logits")
	var loss float64
	tape := ""
	mode := ""
	var err error
	for iter := 0; iter <= cfg.Iters; iter++ {
		var writeGrad, moveGrad []float64
		loss, writeGrad, moveGrad, mode, tape, err = treeLoss(weights, logits, target, cfg.Temp)
		if err != nil {
			return err
		}
		sure := movesDecided(logits)
		if iter == 0 || iter == cfg.Iters || iter%4 == 0 || (tape == cfg.Text && sure) {
			cfg.logf("iter %d  loss %.6f  moves %s  p %s  tape %s", iter, loss, mode, formatProbs(logits), strconv.Quote(tape))
		}
		if tape == cfg.Text && sure && iter > 0 {
			break
		}
		if iter == cfg.Iters {
			break
		}
		for i := range weights {
			weights[i] -= cfg.LR * writeGrad[i]
			logits[i] -= cfg.MoveLR * moveGrad[i]
			if logits[i] > 20 {
				logits[i] = 20
			}
			if logits[i] < -20 {
				logits[i] = -20
			}
		}
	}
	if tape != cfg.Text || !movesDecided(logits) {
		return fmt.Errorf("tape %s moves %s after %d steps", strconv.Quote(tape), mode, cfg.Iters)
	}
	return nil
}

// movesDecided reports whether every move except the last puts probability
// at least 0.9 on its mode direction. One step has no later write, so the
// only move is free.
func movesDecided(logits []float64) bool {
	if len(logits) <= 1 {
		return true
	}
	for _, a := range logits[:len(logits)-1] {
		p := logistic(a)
		if p < 0.9 && p > 0.1 {
			return false
		}
	}
	return true
}

// learnGaussText follows ∂L/∂write for the bytes of cfg.Text.
// With no fixed moves, it also follows ∂L/∂move through the unrolled tree.
// The learned tape is the nearest byte to each cell mean.
func learnGaussText(cfg gaussLearnConfig) error {
	if err := cfg.apply(); err != nil {
		return err
	}
	if cfg.Moves == "" {
		return learnMoveTree(cfg)
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
