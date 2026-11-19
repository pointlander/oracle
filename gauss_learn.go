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

func movesFromMode(mode string) []int {
	out := make([]int, len(mode))
	for i, r := range mode {
		out[i] = 1
		if r == 'L' {
			out[i] = -1
		}
	}
	return out
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
// total loss with respect to each write weight, and the nearest-byte string.
// cells <= 0 or cells >= len(target) reads finished cells 0..len(target)-1.
// A shorter tape wraps, and each step emits the running mean on its cell.
func lossAndWriteGrad(weights []float64, moves []int, target []byte, tau float64, cells int) (float64, []float64, string, error) {
	sum, grad, tape, err := symbolPath(weights, moves, target, tau, cells)
	if err != nil {
		return 0, nil, "", err
	}
	return sum / float64(len(target)), grad, tape, nil
}

// maxMoveUnroll is the deepest left/right tree expanded while learning moves.
// A text of this length has 2^maxMoveUnroll root-to-leaf paths.
const maxMoveUnroll = 16

// defaultMoveLR is the step size on move logits. It is separate from the
// write learning rate, which is scaled to cancel 1/τ².
const defaultMoveLR = 1

// gaussLearnConfig trains write weights of a 256-symbol machine.
// Each state has a write and a move for read 0 and for read 1.
// Moves is a string of L and R. Empty learns both move logits from the
// mode path, expanding the other branch only while that move is undecided.
// LR <= 0 uses Temp², which cancels the 1/τ² in the symbol partial.
// MoveLR <= 0 uses defaultMoveLR. Size <= 0 uses defaultTapeSize.
type gaussLearnConfig struct {
	Text   string
	Moves  string
	Iters  int
	LR     float64
	MoveLR float64
	Size   float64
	Temp   float64
	Log    func(string)
}

func (cfg *gaussLearnConfig) apply() error {
	if cfg.Text == "" {
		return fmt.Errorf("text must be non-empty")
	}
	if cfg.Iters <= 0 {
		cfg.Iters = 32
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
	if cfg.Size == 0 {
		cfg.Size = defaultTapeSize
	}
	if cfg.Size <= 0 || cfg.Size > 1 {
		return fmt.Errorf("tape size must be in (0, 1]")
	}
	return nil
}

func (cfg gaussLearnConfig) logf(format string, args ...any) {
	if cfg.Log != nil {
		cfg.Log(fmt.Sprintf(format, args...))
	}
}

func (cfg gaussLearnConfig) logTape(n int) {
	cells := tapeCells(n, cfg.Size)
	if cells < n {
		cfg.logf("  the head wraps inside %d cells, and each symbol is the mean after that write", cells)
	}
}

func (cfg gaussLearnConfig) logLoss(n int) {
	cfg.logf("  each state has a write and a move for read 0 and for read 1")
	cfg.logf("  a blank cell or a zero byte reads 0, and any other byte reads 1")
	if tapeCells(n, cfg.Size) < n {
		cfg.logf("  the logged loss is the cross-entropy along the mode path")
		cfg.logf("  each step emits the weight it wrote, and the cell mean selects the rule")
	} else {
		cfg.logf("  the logged loss is the cross-entropy of the finished cells along the mode path")
	}
	cfg.logf("  an undecided move takes one step on the other branch")
}

// defaultTapeSize is the tape length as a fraction of the symbols being learned.
const defaultTapeSize = 0.8

// tapeCells is the nearest integer to frac times n, at least 1 and at most n.
func tapeCells(n int, frac float64) int {
	if n < 1 {
		return 0
	}
	c := int(math.Round(frac * float64(n)))
	if c < 1 {
		c = 1
	}
	if c > n {
		c = n
	}
	return c
}

func wrapCell(head, cells int) int {
	h := head % cells
	if h < 0 {
		h += cells
	}
	return h
}

// symbolPath is the total symbol cross-entropy of one path. When cells is
// shorter than the target, the head wraps inside that tape and step i emits
// the mean of the writes on its cell so far. Otherwise cells 0..len(target)-1
// are read after every write, and a write outside that window has partial 0.
func symbolPath(weights []float64, moves []int, target []byte, tau float64, cells int) (sum float64, grad []float64, tape string, err error) {
	if len(weights) != len(moves) {
		return 0, nil, "", fmt.Errorf("weights and moves differ in length")
	}
	if len(weights) == 0 {
		return 0, nil, "", fmt.Errorf("moves must be non-empty")
	}
	for _, m := range moves {
		if m != -1 && m != 1 {
			return 0, nil, "", fmt.Errorf("move %d is not -1 or 1", m)
		}
	}
	limited := cells > 0 && cells < len(target)
	pos := make([]int, len(weights))
	head := 0
	for t, m := range moves {
		if limited {
			pos[t] = wrapCell(head, cells)
			head = pos[t] + m
		} else {
			pos[t] = head
			head += m
		}
	}
	grad = make([]float64, len(weights))
	if !limited {
		out := make([]byte, len(target))
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
			out[i] = nearestByte(mu)
			if n == 0 {
				continue
			}
			share := dmu / float64(n)
			for t := range pos {
				if pos[t] == i {
					grad[t] += share
				}
			}
		}
		return sum, grad, string(out), nil
	}
	out := make([]byte, len(target))
	steps := len(weights)
	if steps > len(target) {
		steps = len(target)
	}
	for i := 0; i < steps; i++ {
		mu := 0.0
		n := 0
		for t := 0; t <= i; t++ {
			if pos[t] == pos[i] {
				mu += weights[t]
				n++
			}
		}
		mu /= float64(n)
		nll, dmu := symbolLoss(mu, tau, int(target[i]))
		sum += nll
		out[i] = nearestByte(mu)
		share := dmu / float64(n)
		for t := 0; t <= i; t++ {
			if pos[t] == pos[i] {
				grad[t] += share
			}
		}
	}
	for i := steps; i < len(target); i++ {
		nll, _ := symbolLoss(0, tau, int(target[i]))
		sum += nll
	}
	return sum, grad, string(out), nil
}

// pathSum adds one path's write partials, multiplied by prob, into writeGrad.
func pathSum(weights []float64, moves []int, target []byte, tau, prob float64, cells int, writeGrad []float64) float64 {
	sum, grad, _, err := symbolPath(weights, moves, target, tau, cells)
	if err != nil {
		return 0
	}
	for i := range grad {
		writeGrad[i] += prob * grad[i]
	}
	return sum
}

// treeLoss unrolls the machine the way an RNN is unrolled, except each step
// branches left and right. A path probability is the product of its branch
// probabilities. The loss is the expected total symbol cross-entropy. The
// write partial is the probability-weighted average of the path partials. The
// move-logit partial at a step is (E[loss|right] - E[loss|left]) p (1-p).
// The move after the last write does not change the tape, so its partial is 0.
func treeLoss(weights, logits []float64, target []byte, tau float64, cells int) (mean float64, writeGrad, moveGrad []float64, mode string, tape string, err error) {
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
		total := pathSum(weights, moves, target, tau, prob, cells, writeGrad)
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
	_, _, tape, err = lossAndWriteGrad(weights, modeMoves, target, tau, cells)
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

// ruleWrite is one write on a cell. rule is 0 or 1.
type ruleWrite struct {
	step int
	rule int
}

// symbolPathRules is the total symbol cross-entropy when each state has a
// write and a move for read 0 and for read 1. Before a write, a cell with no
// observations, or whose mean rounds to byte 0, selects rule 0. Any other
// byte selects rule 1. That choice is constant, so it has no partial. On a
// short tape each step emits the weight it just wrote, so the discrete write
// is that character. The cell mean only selects the rule. Otherwise cells
// 0..len(target)-1 are read after every write, and a write outside that
// window has partial 0. The returned gradients are partials of the total.
func symbolPathRules(w0, w1 []float64, m0, m1 []int, target []byte, tau float64, cells int) (sum float64, g0, g1 []float64, tape string, fired0, fired1 []bool, err error) {
	return walkRules(w0, w1, m0, m1, target, tau, cells, false)
}

// meanPathSum is the total loss of the running cell means on a short tape.
// Move logits follow this gap, because a move changes which writes share a
// cell. The write weights themselves follow the character each rule emits.
func meanPathSum(w0, w1 []float64, m0, m1 []int, target []byte, tau float64, cells int) (float64, error) {
	sum, _, _, _, _, _, err := walkRules(w0, w1, m0, m1, target, tau, cells, true)
	return sum, err
}

func walkRules(w0, w1 []float64, m0, m1 []int, target []byte, tau float64, cells int, useMean bool) (sum float64, g0, g1 []float64, tape string, fired0, fired1 []bool, err error) {
	n := len(w0)
	if n == 0 || len(w1) != n || len(m0) != n || len(m1) != n {
		return 0, nil, nil, "", nil, nil, fmt.Errorf("weights and moves differ in length")
	}
	if len(target) == 0 {
		return 0, nil, nil, "", nil, nil, fmt.Errorf("text must be non-empty")
	}
	for t := 0; t < n; t++ {
		if m0[t] != -1 && m0[t] != 1 {
			return 0, nil, nil, "", nil, nil, fmt.Errorf("move %d is not -1 or 1", m0[t])
		}
		if m1[t] != -1 && m1[t] != 1 {
			return 0, nil, nil, "", nil, nil, fmt.Errorf("move %d is not -1 or 1", m1[t])
		}
	}
	limited := cells > 0 && cells < len(target)
	g0 = make([]float64, n)
	g1 = make([]float64, n)
	fired0 = make([]bool, n)
	fired1 = make([]bool, n)
	writers := make(map[int][]ruleWrite)
	sumW := make(map[int]float64)
	count := make(map[int]int)
	steps := n
	if limited && steps > len(target) {
		steps = len(target)
	}
	out := make([]byte, len(target))
	head := 0
	for t := 0; t < steps; t++ {
		pos := head
		if limited {
			pos = wrapCell(head, cells)
		}
		rule := 0
		if count[pos] > 0 && nearestByte(sumW[pos]/float64(count[pos])) != 0 {
			rule = 1
		}
		w := w0[t]
		move := m0[t]
		if rule == 1 {
			w = w1[t]
			move = m1[t]
			fired1[t] = true
		} else {
			fired0[t] = true
		}
		sumW[pos] += w
		count[pos]++
		writers[pos] = append(writers[pos], ruleWrite{step: t, rule: rule})
		if limited {
			if useMean {
				c := count[pos]
				mu := sumW[pos] / float64(c)
				nll, dmu := symbolLoss(mu, tau, int(target[t]))
				sum += nll
				out[t] = nearestByte(mu)
				share := dmu / float64(c)
				for _, wr := range writers[pos] {
					if wr.rule == 0 {
						g0[wr.step] += share
					} else {
						g1[wr.step] += share
					}
				}
			} else {
				nll, dmu := symbolLoss(w, tau, int(target[t]))
				sum += nll
				out[t] = nearestByte(w)
				if rule == 0 {
					g0[t] += dmu
				} else {
					g1[t] += dmu
				}
			}
		}
		head = pos + move
	}
	if limited {
		for i := steps; i < len(target); i++ {
			nll, _ := symbolLoss(0, tau, int(target[i]))
			sum += nll
		}
		return sum, g0, g1, string(out), fired0, fired1, nil
	}
	for i, y := range target {
		mu := 0.0
		c := count[i]
		if c > 0 {
			mu = sumW[i] / float64(c)
		}
		nll, dmu := symbolLoss(mu, tau, int(y))
		sum += nll
		out[i] = nearestByte(mu)
		if c == 0 {
			continue
		}
		share := dmu / float64(c)
		for _, wr := range writers[i] {
			if wr.rule == 0 {
				g0[wr.step] += share
			} else {
				g1[wr.step] += share
			}
		}
	}
	return sum, g0, g1, string(out), fired0, fired1, nil
}

func modeMoves(logits []float64) []int {
	out := make([]int, len(logits))
	for i, a := range logits {
		out[i] = -1
		if a >= 0 {
			out[i] = 1
		}
	}
	return out
}

func clipLogit(a float64) float64 {
	if a > 20 {
		return 20
	}
	if a < -20 {
		return -20
	}
	return a
}

// moveStep applies one move-logit update. The loss gap of a one-byte miss is
// on the order of 1/τ², so the step is scaled by τ², the same cancellation
// the write learning rate uses. The step is at most 1, so a single comparison
// cannot carry a fair coin across the 0.9 decision.
func moveStep(logit, grad, lr, tau float64) float64 {
	step := lr * grad * tau * tau
	if step > 1 {
		return clipLogit(logit - 1)
	}
	if step < -1 {
		return clipLogit(logit + 1)
	}
	return clipLogit(logit - step)
}

// logitOpen reports a move that is still undecided. A probability of at
// least 0.9, or at most 0.1, has settled on its mode direction.
func logitOpen(a float64) bool {
	p := logistic(a)
	return p < 0.9 && p > 0.1
}

// moveBranchLoss trains the mode path. A logit at least 0 moves right.
// Write partials come from that path. On a short tape the logged loss is the
// character each rule wrote, and the move gap is the cell-mean loss. For each
// step except the last, and for each read, an undecided logit is also flipped
// on its own. Its partial is (loss right − loss left) p (1−p). A decided
// logit is left alone. The last move does not change a later write.
func moveBranchLoss(w0, w1, a0, a1 []float64, target []byte, tau float64, cells int) (mean float64, g0, g1, dg0, dg1 []float64, mode0, mode1, tape string, fired0, fired1 []bool, err error) {
	if len(w0) != len(a0) || len(w0) != len(a1) || len(w0) == 0 {
		return 0, nil, nil, nil, nil, "", "", "", nil, nil, fmt.Errorf("weights and move logits differ in length")
	}
	m0 := modeMoves(a0)
	m1 := modeMoves(a1)
	var sum float64
	sum, g0, g1, tape, fired0, fired1, err = symbolPathRules(w0, w1, m0, m1, target, tau, cells)
	if err != nil {
		return 0, nil, nil, nil, nil, "", "", "", nil, nil, err
	}
	moveBase := sum
	useMean := cells > 0 && cells < len(target)
	if useMean {
		moveBase, err = meanPathSum(w0, w1, m0, m1, target, tau, cells)
		if err != nil {
			return 0, nil, nil, nil, nil, "", "", "", nil, nil, err
		}
	}
	dg0 = make([]float64, len(a0))
	dg1 = make([]float64, len(a1))
	for t := 0; t < len(a0)-1; t++ {
		if logitOpen(a0[t]) {
			dg0[t] = flippedMoveGrad(w0, w1, m0, m1, target, tau, cells, moveBase, useMean, 0, t, a0[t])
		}
		if logitOpen(a1[t]) {
			dg1[t] = flippedMoveGrad(w0, w1, m0, m1, target, tau, cells, moveBase, useMean, 1, t, a1[t])
		}
	}
	n := float64(len(target))
	return sum / n, g0, g1, dg0, dg1, formatMoves(m0), formatMoves(m1), tape, fired0, fired1, nil
}

// flippedMoveGrad is the partial of flipping one read's move off the mode.
// The other moves stay on the mode path.
func flippedMoveGrad(w0, w1 []float64, m0, m1 []int, target []byte, tau float64, cells int, base float64, useMean bool, rule, step int, logit float64) float64 {
	alt0 := append([]int(nil), m0...)
	alt1 := append([]int(nil), m1...)
	mode := alt0[step]
	if rule == 1 {
		mode = alt1[step]
		alt1[step] = -mode
	} else {
		alt0[step] = -mode
	}
	var flipped float64
	var err error
	if useMean {
		flipped, err = meanPathSum(w0, w1, alt0, alt1, target, tau, cells)
	} else {
		flipped, _, _, _, _, _, err = symbolPathRules(w0, w1, alt0, alt1, target, tau, cells)
	}
	if err != nil {
		return 0
	}
	lRight, lLeft := base, flipped
	if mode < 0 {
		lRight, lLeft = flipped, base
	}
	p := logistic(logit)
	return (lRight - lLeft) * p * (1 - p)
}

// rulesDecided reports whether every move that fired, except the move after
// the last write, puts probability at least 0.9 on its mode direction. A
// rule that never fired stays free. One step has no later write.
func rulesDecided(a0, a1 []float64, fired0, fired1 []bool) bool {
	if len(a0) <= 1 {
		return true
	}
	for t := 0; t < len(a0)-1; t++ {
		if t < len(fired0) && fired0[t] && logitOpen(a0[t]) {
			return false
		}
		if t < len(fired1) && fired1[t] && logitOpen(a1[t]) {
			return false
		}
	}
	return true
}

// discreteNext is the state a learned transition enters. Read 0 advances,
// and the last state halts. Read 1 returns to the previous state, and the
// first state halts. A one-state machine stays on a 1, so that entry is not
// the halt entry.
func discreteNext(state, n, read int) int {
	if read == 0 {
		if state >= n {
			return 0
		}
		return state + 1
	}
	if n == 1 {
		return 1
	}
	if state <= 1 {
		return 0
	}
	return state - 1
}

// learnedPair is the two transitions of one learned state. Each read writes
// the nearest byte of its own weight and moves as that read learned. The
// character is the write. It is not packed into the 6-bit binary encoding.
// The two entries go to different states.
func learnedPair(w0, w1 float64, move0, move1, state, n int) [2]Transition {
	return [2]Transition{
		{WriteSymbol: nearestByte(w0), MoveDir: move0, NextState: discreteNext(state, n, 0)},
		{WriteSymbol: nearestByte(w1), MoveDir: move1, NextState: discreteNext(state, n, 1)},
	}
}

// runDiscrete runs the learned table. A missing cell or a zero byte takes
// rule 0. Any other byte takes rule 1. writes is the characters in order.
// tape is the cells afterwards. A short tape is cells 0..cells-1 and the
// head wraps. Otherwise the window is cells 0..n-1. A cell outside that
// window is still remembered, and a cell that was never written stays 0.
func runDiscrete(w0, w1 []float64, m0, m1 []int, cells int) (writes, tape string) {
	n := len(w0)
	if n == 0 || len(w1) != n || len(m0) != n || len(m1) != n {
		return "", ""
	}
	writes, tape, _ = runConverted(machineTable(w0, w1, m0, m1), n, cells)
	return writes, tape
}

// machineTable is the character transition table. The write is the nearest
// byte of that rule's weight. The table is not round-tripped through the
// binary encoding, because a character does not fit in one write bit.
func machineTable(w0, w1 []float64, m0, m1 []int) map[int][2]Transition {
	n := len(w0)
	if n == 0 || len(w1) != n || len(m0) != n || len(m1) != n {
		return nil
	}
	table := make(map[int][2]Transition, n)
	for i := 0; i < n; i++ {
		table[i+1] = learnedPair(w0[i], w1[i], m0[i], m1[i], i+1, n)
	}
	return table
}

// discreteStates runs the character machine on a blank tape and returns the
// state at the start of each step. A non-zero byte sends the machine to a
// different state from a 0. State 0 is observed on the step after the halt
// transition. The bound is one step past the chain. A run still going there
// has not halted.
func discreteStates(w0, w1 []float64, m0, m1 []int) []int {
	table := machineTable(w0, w1, m0, m1)
	if table == nil {
		return nil
	}
	return traceTransitions(table, len(w0)+1)
}

// traceTransitions follows the table. A missing cell or a zero byte selects
// rule 0. State 0, or a state missing from the table, stops the trace. A run
// that is still going at maxSteps returns the states it reached. The bound
// is not a halting decider.
func traceTransitions(table map[int][2]Transition, maxSteps int) []int {
	tape := make(map[int]byte)
	head := 0
	state := 1
	states := make([]int, 0, maxSteps)
	for step := 0; step < maxSteps; step++ {
		states = append(states, state)
		if state == 0 {
			return states
		}
		transitions, exists := table[state]
		if !exists {
			return states
		}
		current, cellExists := tape[head]
		if !cellExists {
			current = 0
		}
		rule := transitions[1]
		if current == 0 {
			rule = transitions[0]
		}
		tape[head] = rule.WriteSymbol
		head += rule.MoveDir
		state = rule.NextState
	}
	return states
}

func formatStates(states []int) string {
	b := make([]byte, 0, len(states)*3)
	for i, s := range states {
		if i > 0 {
			b = append(b, ' ')
		}
		b = append(b, strconv.Itoa(s)...)
	}
	return string(b)
}

// formatStateChar is one learned state. The state is that symbol. State 0
// has no symbol, so it is a zero byte.
func formatStateChar(symbols string, state int) string {
	if state >= 1 && state <= len(symbols) {
		return strconv.QuoteRune(rune(symbols[state-1]))
	}
	return strconv.QuoteRune(0)
}

// formatStateChars prints each visited state as its learned symbol. State 0
// has no symbol, so it is a zero byte. The result is quoted.
func formatStateChars(states []int, symbols string) string {
	buf := make([]byte, len(states))
	for i, s := range states {
		if s >= 1 && s <= len(symbols) {
			buf[i] = symbols[s-1]
		}
	}
	return strconv.Quote(string(buf))
}

// emissionText is the nearest-byte string along this path. The loss target
// does not change that read, so the target bytes are zeros.
func emissionText(weights []float64, moves []int, cells int) string {
	_, _, tape, err := symbolPath(weights, moves, make([]byte, len(weights)), 1, cells)
	if err != nil {
		return ""
	}
	return tape
}

// recordSymbols saves each symbol byte in a buffer. The bytes are not written
// on a tape.
func recordSymbols(symbols string) string {
	buf := make([]byte, 0, len(symbols))
	for i := 0; i < len(symbols); i++ {
		buf = append(buf, symbols[i])
	}
	return string(buf)
}

// runConverted runs a discrete machine. A missing cell or a zero byte selects
// rule 0. Any other byte selects rule 1. A shorter tape wraps inside
// cells 0..cells-1. Otherwise every write is remembered, and the reported
// tape is cells 0..n-1. writes is the symbol each transition wrote. states
// is the state at the start of each step, including the halt state when the
// run enters it.
func runConverted(table map[int][2]Transition, n, cells int) (writes, tape string, states []int) {
	if table == nil || n < 1 {
		return "", "", nil
	}
	limited := cells > 0 && cells < n
	width := n
	if limited {
		width = cells
	}
	mem := make(map[int]byte)
	bits := make([]byte, 0, n)
	head := 0
	state := 1
	states = make([]int, 0, n+1)
	for step := 0; step < n+1; step++ {
		states = append(states, state)
		if state == 0 {
			break
		}
		transitions, ok := table[state]
		if !ok {
			break
		}
		pos := head
		if limited {
			pos = wrapCell(head, cells)
		}
		current := byte(0)
		if b, ok := mem[pos]; ok {
			current = b
		}
		rule := transitions[1]
		if current == 0 {
			rule = transitions[0]
		}
		bits = append(bits, rule.WriteSymbol)
		mem[pos] = rule.WriteSymbol
		head = pos + rule.MoveDir
		state = rule.NextState
	}
	cell := make([]byte, width)
	for i := range cell {
		cell[i] = mem[i]
	}
	return string(bits), string(cell), states
}

// ruleSymbols is the blank-tape spelling of each state: the nearest byte of
// the read-0 weight. State 0 has no symbol.
func ruleSymbols(w0 []float64) string {
	b := make([]byte, len(w0))
	for i, w := range w0 {
		b[i] = nearestByte(w)
	}
	return string(b)
}

// logDiscrete converts the learned weights to a discrete machine and runs
// that machine. Each rule writes its learned character. The symbol bytes are
// recorded in a buffer and are not written on the tape. A state is named by
// the read-0 character.
func (cfg gaussLearnConfig) logDiscrete(w0, w1 []float64, m0, m1 []int, cells int) string {
	table := machineTable(w0, w1, m0, m1)
	if table == nil {
		return ""
	}
	names := ruleSymbols(w0)
	target := []byte(cfg.Text)
	if len(target) == 0 {
		target = make([]byte, len(w0))
	}
	emitted := ""
	if _, _, _, tape, _, _, err := symbolPathRules(w0, w1, m0, m1, target, 1, cells); err == nil {
		emitted = tape
	}
	symbols := recordSymbols(emitted)
	cfg.logf("discrete machine")
	cfg.logf("  a state is its learned character")
	cfg.logf("  read 0 writes that character, moves as learned, and goes to the next character")
	cfg.logf("  read 1 writes its character, moves as learned, and goes to the previous character")
	cfg.logf("  a zero byte is blank")
	for s := 1; s <= len(w0); s++ {
		for read := 0; read < 2; read++ {
			tr := table[s][read]
			move := byte('R')
			if tr.MoveDir < 0 {
				move = 'L'
			}
			cfg.logf("  state %s  read %d  write %s  move %c  next %s", formatStateChar(names, s), read, strconv.QuoteRune(rune(tr.WriteSymbol)), move, formatStateChar(names, tr.NextState))
		}
	}
	writes, tape, states := runConverted(table, len(w0), cells)
	cfg.logf("  writes %s", strconv.Quote(writes))
	cfg.logf("  tape %s", strconv.Quote(tape))
	cfg.logf("  states %s", formatStateChars(states, names))
	if len(states) == len(w0)+1 && states[len(states)-1] != 0 {
		cfg.logf("  still running")
	}
	cfg.logf("  the learned symbol bytes are recorded in a buffer")
	cfg.logf("  text %s", strconv.Quote(symbols))
	return writes
}

// learnMoveTree follows the mode path of one machine for the whole string.
// An undecided move also takes a single step on the other branch. The
// reported states are the nearest-byte read along the mode path. After that
// string matches, and the moves that fired have settled, the weights convert
// to a discrete machine that writes the learned characters.
func learnMoveTree(cfg gaussLearnConfig) error {
	target := []byte(cfg.Text)
	n := len(target)
	w0 := make([]float64, n)
	w1 := make([]float64, n)
	a0 := make([]float64, n)
	a1 := make([]float64, n)
	cells := tapeCells(n, cfg.Size)
	cfg.logf("256-symbol gaussian machine")
	cfg.logf("  target %s", strconv.Quote(cfg.Text))
	cfg.logf("  %d states, alphabet 256, tape cells %d", n, cells)
	cfg.logf("  one machine for the whole string")
	cfg.logf("  the mode path is trained, and the other move is expanded only while that move is undecided")
	cfg.logTape(n)
	cfg.logLoss(n)
	var loss float64
	tape := ""
	mode0, mode1 := "", ""
	for iter := 0; iter <= cfg.Iters; iter++ {
		var g0, g1, dg0, dg1 []float64
		var fired0, fired1 []bool
		var err error
		loss, g0, g1, dg0, dg1, mode0, mode1, tape, fired0, fired1, err = moveBranchLoss(w0, w1, a0, a1, target, cfg.Temp, cells)
		if err != nil {
			return err
		}
		sure := rulesDecided(a0, a1, fired0, fired1)
		if iter == 0 || iter == cfg.Iters || iter%4 == 0 || (tape == cfg.Text && sure) {
			cfg.logf("iter %d  loss %.6f  moves %s / %s  p %s / %s  states %s", iter, loss, mode0, mode1, formatProbs(a0), formatProbs(a1), strconv.Quote(tape))
		}
		if tape == cfg.Text && sure && iter > 0 {
			cfg.logDiscrete(w0, w1, movesFromMode(mode0), movesFromMode(mode1), cells)
			return nil
		}
		if iter == cfg.Iters {
			break
		}
		for i := range w0 {
			w0[i] -= cfg.LR * g0[i]
			w1[i] -= cfg.LR * g1[i]
			a0[i] = moveStep(a0[i], dg0[i], cfg.MoveLR, cfg.Temp)
			a1[i] = moveStep(a1[i], dg1[i], cfg.MoveLR, cfg.Temp)
		}
	}
	return fmt.Errorf("states %s moves %s / %s after %d steps", strconv.Quote(tape), mode0, mode1, cfg.Iters)
}

// learnGaussText follows the partial of the symbol loss for the bytes of cfg.Text.
// Each state has a write for the symbol it reads. With no fixed moves, it also
// follows the partial of an undecided move. The learned states are the nearest
// byte along the mode path. After that string matches, the weights convert to
// a discrete machine that writes those characters.
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
	w0 := make([]float64, len(moves))
	w1 := make([]float64, len(moves))
	cells := tapeCells(len(target), cfg.Size)
	cfg.logf("256-symbol gaussian machine")
	cfg.logf("  target %s", strconv.Quote(cfg.Text))
	cfg.logf("  %d states, alphabet 256, tape cells %d, moves %s", len(moves), cells, formatMoves(moves))
	cfg.logTape(len(target))
	cfg.logf("  the head starts on cell 0, writes the rule for the symbol there, then moves")
	cfg.logf("  both reads use the fixed move")
	cfg.logf("  loss is the cross-entropy of a Gaussian softmax over byte codes")
	cfg.logf("  each step subtracts the learning rate times the partial of that loss")
	var loss float64
	tape := ""
	for iter := 0; iter <= cfg.Iters; iter++ {
		var sum float64
		var g0, g1 []float64
		sum, g0, g1, tape, _, _, err = symbolPathRules(w0, w1, moves, moves, target, cfg.Temp, cells)
		if err != nil {
			return err
		}
		loss = sum / float64(len(target))
		if iter == 0 || iter == cfg.Iters || iter%4 == 0 || tape == cfg.Text {
			cfg.logf("iter %d  loss %.6f  states %s", iter, loss, strconv.Quote(tape))
		}
		if tape == cfg.Text && iter > 0 {
			break
		}
		if iter == cfg.Iters {
			break
		}
		for i := range w0 {
			w0[i] -= cfg.LR * g0[i]
			w1[i] -= cfg.LR * g1[i]
		}
	}
	if tape != cfg.Text {
		return fmt.Errorf("states %s after %d steps", strconv.Quote(tape), cfg.Iters)
	}
	cfg.logDiscrete(w0, w1, moves, moves, cells)
	return nil
}
