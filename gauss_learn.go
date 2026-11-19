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
// Moves is a string of L and R. Empty learns a move logit per state by
// unrolling both branches. LR <= 0 uses Temp², which cancels the 1/τ² in
// the symbol partial. MoveLR <= 0 uses defaultMoveLR. Size <= 0 uses
// defaultTapeSize.
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
	if tapeCells(n, cfg.Size) < n {
		cfg.logf("  the logged loss is the cross-entropy along the mode path")
		cfg.logf("  writes follow that partial; move logits follow the expected loss over the tree")
		return
	}
	cfg.logf("  loss is the expected cross-entropy over the move tree")
	cfg.logf("  each step follows the partial of that loss through the writes and the move logits")
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

// discreteBit is the binary write of a learned weight. A positive weight
// writes 1. Every other weight, including 0, writes 0.
func discreteBit(w float64) byte {
	if w > 0 {
		return '1'
	}
	return '0'
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

// discretePair is the two transitions of one learned state. Read 0 writes
// the sign of the weight and moves as learned. Read 1 writes the other bit
// and moves the other way. The two entries go to different states.
func discretePair(w float64, move, state, n int) [2]Transition {
	bit := discreteBit(w)
	other := byte('1')
	if bit == '1' {
		other = '0'
	}
	return [2]Transition{
		{WriteSymbol: bit, MoveDir: move, NextState: discreteNext(state, n, 0)},
		{WriteSymbol: other, MoveDir: -move, NextState: discreteNext(state, n, 1)},
	}
}

// encodeDiscrete is the binary machine with one state per learned weight.
// The 0 entry and the 1 entry go to different states. A machine with no
// states, or with more than 63, does not fit the 6-bit next-state field.
func encodeDiscrete(weights []float64, moves []int) (string, bool) {
	n := len(weights)
	if n == 0 || n != len(moves) || n > 63 {
		return "", false
	}
	mp := make(map[int][2]Transition, n)
	for i := 0; i < n; i++ {
		mp[i+1] = discretePair(weights[i], moves[i], i+1, n)
	}
	return EncodeMachine(mp), true
}

// runDiscrete runs the learned table. A 0 uses the sign, the learned move,
// and the next state. A 1 uses the other bit, the other move, and the other
// state. writes is those bits in order. tape is the cells afterwards. A
// short tape is cells 0..cells-1 and the head wraps. Otherwise the window is
// cells 0..n-1. A cell outside that window is still remembered, and a cell
// that was never written stays 0.
func runDiscrete(weights []float64, moves []int, cells int) (writes, tape string) {
	n := len(weights)
	if n == 0 || n != len(moves) {
		return "", ""
	}
	writes, tape, _ = runConverted(machineTable(weights, moves), n, cells)
	return writes, tape
}

// machineTable is the binary transition table. When the machine fits in the
// 6-bit next-state field, the table is decoded from that encoding, so a run
// tests the round trip. A longer machine uses the same rules directly.
func machineTable(weights []float64, moves []int) map[int][2]Transition {
	n := len(weights)
	if n == 0 || n != len(moves) {
		return nil
	}
	if code, ok := encodeDiscrete(weights, moves); ok {
		if table := decodeMachine(code); table != nil {
			return table
		}
	}
	table := make(map[int][2]Transition, n)
	for i := 0; i < n; i++ {
		table[i+1] = discretePair(weights[i], moves[i], i+1, n)
	}
	return table
}

// decodeMachine reads an encoding the way HaltAt does. A blank cell is '0'.
// An empty or partial encoding does not decode.
func decodeMachine(code string) map[int][2]Transition {
	if len(code) == 0 || len(code)%16 != 0 {
		return nil
	}
	numStates := len(code) / 16
	table := make(map[int][2]Transition, numStates)
	for s := 1; s <= numStates; s++ {
		offset := (s - 1) * 16
		t0, ok0 := decodeTransition(code[offset : offset+8])
		t1, ok1 := decodeTransition(code[offset+8 : offset+16])
		if !ok0 || !ok1 {
			return nil
		}
		table[s] = [2]Transition{t0, t1}
	}
	return table
}

// discreteStates runs the binary machine on a blank tape and returns the
// state at the start of each step. A 1 sends the machine to a different
// state from a 0. State 0 is observed on the step after the halt transition.
// The bound is one step past the chain. A run still going there has not halted.
func discreteStates(weights []float64, moves []int) []int {
	table := machineTable(weights, moves)
	if table == nil {
		return nil
	}
	return traceTransitions(table, len(weights)+1)
}

// traceTransitions follows HaltAt. State 0, or a state missing from the
// table, stops the trace. A run that is still going at maxSteps returns the
// states it reached. The bound is not a halting decider.
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
			current = '0'
		}
		rule := transitions[1]
		if current == '0' {
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

// runConverted runs a discrete machine. A shorter tape wraps inside
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
		current := byte('0')
		if b, ok := mem[pos]; ok {
			current = b
		}
		rule := transitions[1]
		if current == '0' {
			rule = transitions[0]
		}
		bits = append(bits, rule.WriteSymbol)
		mem[pos] = rule.WriteSymbol
		head = pos + rule.MoveDir
		state = rule.NextState
	}
	cell := make([]byte, width)
	for i := range cell {
		b, ok := mem[i]
		if !ok {
			b = '0'
		}
		cell[i] = b
	}
	return string(bits), string(cell), states
}

// logDiscrete converts the learned weights to a discrete machine and runs
// that machine. The symbol bytes are recorded in a buffer and are not written
// on the tape.
func (cfg gaussLearnConfig) logDiscrete(weights []float64, moves []int, cells int) string {
	table := machineTable(weights, moves)
	if table == nil {
		return ""
	}
	cfg.logf("discrete machine")
	cfg.logf("  read 0 writes the sign, moves as learned, and goes to the next state")
	cfg.logf("  read 1 writes the other bit, moves the other way, and goes to the previous state")
	for s := 1; s <= len(weights); s++ {
		for read := 0; read < 2; read++ {
			tr := table[s][read]
			move := byte('R')
			if tr.MoveDir < 0 {
				move = 'L'
			}
			cfg.logf("  state %d  read %d  write %c  move %c  next %d", s, read, tr.WriteSymbol, move, tr.NextState)
		}
	}
	writes, tape, states := runConverted(table, len(weights), cells)
	cfg.logf("  writes %s", writes)
	cfg.logf("  tape %s", tape)
	cfg.logf("  states %s", formatStates(states))
	if len(states) == len(weights)+1 && states[len(states)-1] != 0 {
		cfg.logf("  still running")
	}
	cfg.logf("  the learned symbol bytes are recorded in a buffer")
	cfg.logf("  text %s", strconv.Quote(recordSymbols(emissionText(weights, moves, cells))))
	return writes
}

// learnMoveTree follows the partials of the expected loss through the unrolled
// move tree. Text longer than one unroll is trained in successive blocks of
// maxMoveUnroll symbols. Each block starts on its first symbol. The block
// tapes are joined in order. The reported states are the nearest-byte read along
// each block's mode path. After a block matches, its weights convert to a
// binary machine. Several blocks also join those write bits.
func learnMoveTree(cfg gaussLearnConfig) error {
	target := []byte(cfg.Text)
	if len(target) <= maxMoveUnroll {
		cfg.logf("256-symbol gaussian machine")
		cfg.logf("  target %s", strconv.Quote(cfg.Text))
		cfg.logf("  %d states, alphabet 256, tape cells %d, both moves unrolled at every state", len(target), tapeCells(len(target), cfg.Size))
		cfg.logTape(len(target))
		cfg.logLoss(len(target))
		_, err := trainMoveBlock(cfg, target, 0)
		return err
	}
	cfg.logf("256-symbol gaussian machine")
	cfg.logf("  target %s", strconv.Quote(cfg.Text))
	cfg.logf("  %d symbols, alphabet 256, blocks of %d", len(target), maxMoveUnroll)
	cfg.logf("  each block unrolls both moves from its first symbol")
	cfg.logTape(maxMoveUnroll)
	cfg.logLoss(maxMoveUnroll)
	var joined []byte
	blocks := 0
	for start := 0; start < len(target); start += maxMoveUnroll {
		end := start + maxMoveUnroll
		if end > len(target) {
			end = len(target)
		}
		cfg.logf("block [%d:%d]  tape cells %d", start, end, tapeCells(end-start, cfg.Size))
		writes, err := trainMoveBlock(cfg, target[start:end], start)
		if err != nil {
			return err
		}
		joined = append(joined, writes...)
		blocks++
	}
	cfg.logf("states %s", strconv.Quote(cfg.Text))
	if blocks > 1 {
		cfg.logf("discrete writes %s", string(joined))
	}
	return nil
}

// trainMoveBlock learns one unroll. start is the block's index in the full
// text and is used only in the error. Steps other than the last must put at
// least probability 0.9 on the mode direction. The last move has partial 0.
// On a match it prints the binary machine and returns that machine's write bits.
func trainMoveBlock(cfg gaussLearnConfig, target []byte, start int) (string, error) {
	text := string(target)
	weights := make([]float64, len(target))
	logits := make([]float64, len(target))
	cells := tapeCells(len(target), cfg.Size)
	var loss float64
	tape := ""
	mode := ""
	var err error
	for iter := 0; iter <= cfg.Iters; iter++ {
		var writeGrad, moveGrad []float64
		loss, writeGrad, moveGrad, mode, tape, err = treeLoss(weights, logits, target, cfg.Temp, cells)
		if err != nil {
			return "", err
		}
		// On a short tape every path can retarget a cell, so a uniform mixture
		// keeps the move partial near 0. Writes follow the mode path. Move
		// logits still follow the expected loss over the tree.
		if cells < len(target) {
			loss, writeGrad, tape, err = lossAndWriteGrad(weights, movesFromMode(mode), target, cfg.Temp, cells)
			if err != nil {
				return "", err
			}
		}
		sure := movesDecided(logits)
		if iter == 0 || iter == cfg.Iters || iter%4 == 0 || (tape == text && sure) {
			cfg.logf("iter %d  loss %.6f  moves %s  p %s  states %s", iter, loss, mode, formatProbs(logits), strconv.Quote(tape))
		}
		if tape == text && sure && iter > 0 {
			return cfg.logDiscrete(weights, movesFromMode(mode), cells), nil
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
	return "", fmt.Errorf("block [%d:%d] states %s moves %s after %d steps", start, start+len(target), strconv.Quote(tape), mode, cfg.Iters)
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
// The learned tape is the nearest byte to each cell mean. After that tape
// matches, the weights convert to a binary machine that writes 0 or 1.
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
	cfg.logf("  %d states, alphabet 256, tape cells %d, moves %s", len(moves), tapeCells(len(target), cfg.Size), formatMoves(moves))
	cfg.logTape(len(target))
	cfg.logf("  the head starts on cell 0, writes, then moves left or right")
	cfg.logf("  loss is the cross-entropy of a Gaussian softmax over byte codes")
	cfg.logf("  each step subtracts the learning rate times the partial of that loss")
	var loss float64
	tape := ""
	for iter := 0; iter <= cfg.Iters; iter++ {
		var grad []float64
		loss, grad, tape, err = lossAndWriteGrad(weights, moves, target, cfg.Temp, tapeCells(len(target), cfg.Size))
		if err != nil {
			return err
		}
		if iter == 0 || iter == cfg.Iters || iter%4 == 0 || tape == cfg.Text {
			cfg.logf("iter %d  loss %.6f  states %s", iter, loss, strconv.Quote(tape))
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
		return fmt.Errorf("states %s after %d steps", strconv.Quote(tape), cfg.Iters)
	}
	cfg.logDiscrete(weights, moves, tapeCells(len(target), cfg.Size))
	return nil
}
