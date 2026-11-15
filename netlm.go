// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"flag"
	"fmt"
	"math/bits"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
)

// symbolSpec is one bounded machine per alphabet symbol.
// Every machine reads the previous byte, most significant bit on cell 0.
// The symbol whose machine does not halt is the next byte.
// If several machines are still running at MaxSteps, the smallest byte wins.
// If every machine halts, the step emits no symbol.
type symbolSpec struct {
	Alphabet []byte
	States   int
	MaxSteps int
}

func (s symbolSpec) per() int {
	return s.States * 2 * 3
}

func (s symbolSpec) nGenes() int {
	return len(s.Alphabet) * s.per()
}

func (s symbolSpec) validate() error {
	if len(s.Alphabet) < 1 {
		return fmt.Errorf("alphabet must be non-empty")
	}
	if s.States < 1 || s.States > 2 {
		return fmt.Errorf("states must be 1 or 2")
	}
	if s.MaxSteps < 2 {
		return fmt.Errorf("oracle steps must be at least 2")
	}
	return nil
}

func collectAlphabet(data []byte) []byte {
	var seen [256]bool
	out := make([]byte, 0)
	for _, b := range data {
		if !seen[b] {
			seen[b] = true
			out = append(out, b)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// byteTape is the previous byte, most significant bit first.
func byteTape(b byte) string {
	buf := make([]byte, 8)
	for i := 0; i < 8; i++ {
		if b&(1<<(7-i)) != 0 {
			buf[i] = '1'
		} else {
			buf[i] = '0'
		}
	}
	return string(buf)
}

func machineFromGenes(g []int, states int) string {
	mp := make(map[int][2]Transition, states)
	p := 0
	for s := 1; s <= states; s++ {
		var pair [2]Transition
		for sym := 0; sym < 2; sym++ {
			write := byte('0')
			if g[p] != 0 {
				write = '1'
			}
			move := -1
			if g[p+1] != 0 {
				move = 1
			}
			pair[sym] = Transition{WriteSymbol: write, MoveDir: move, NextState: g[p+2]}
			p += 3
		}
		mp[s] = pair
	}
	return EncodeMachine(mp)
}

func decodeSymbols(g []int, spec symbolSpec) []string {
	ensure(len(g) == spec.nGenes(), "genome %d != %d", len(g), spec.nGenes())
	out := make([]string, len(spec.Alphabet))
	per := spec.per()
	for i := range spec.Alphabet {
		out[i] = machineFromGenes(g[i*per:(i+1)*per], spec.States)
	}
	return out
}

// survivingSymbol runs one oracle per symbol. The winner is a machine that is
// still running at the step bound. The alphabet is sorted, so the first
// survivor is the smallest byte. ok is false when every machine halts.
func survivingSymbol(sim *UniversalBoundedSimulator, machines []string, alphabet []byte, prev byte) (sym byte, ok bool) {
	tape := byteTape(prev)
	for i, m := range machines {
		if sim.HaltAt(m, tape) < 0 {
			return alphabet[i], true
		}
	}
	return 0, false
}

func symbolDetail(sim *UniversalBoundedSimulator, machines []string, alphabet, data []byte) (acc float64, correct, total, missed int) {
	if len(data) < 2 {
		return 0, 0, 0, 0
	}
	total = len(data) - 1
	for i := 0; i < total; i++ {
		sym, ok := survivingSymbol(sim, machines, alphabet, data[i])
		if !ok {
			missed++
			continue
		}
		if sym == data[i+1] {
			correct++
		}
	}
	return float64(correct) / float64(total), correct, total, missed
}

func continueSymbols(sim *UniversalBoundedSimulator, machines []string, alphabet, prompt []byte, count int, fallback byte) []byte {
	out := make([]byte, count)
	var prev byte
	if len(prompt) > 0 {
		prev = prompt[len(prompt)-1]
	}
	for i := 0; i < count; i++ {
		sym, ok := survivingSymbol(sim, machines, alphabet, prev)
		if !ok {
			sym = fallback
		}
		out[i] = sym
		prev = sym
	}
	return out
}

func majorityByte(data []byte) (byte, float64) {
	if len(data) < 2 {
		return 0, 0
	}
	var count [256]int
	for i := 1; i < len(data); i++ {
		count[data[i]]++
	}
	mode := 0
	for b := 1; b < 256; b++ {
		if count[b] > count[mode] {
			mode = b
		}
	}
	return byte(mode), float64(count[mode]) / float64(len(data)-1)
}

func applyUnigram(data []byte, mode byte) float64 {
	if len(data) < 2 {
		return 0
	}
	correct := 0
	for i := 1; i < len(data); i++ {
		if data[i] == mode {
			correct++
		}
	}
	return float64(correct) / float64(len(data)-1)
}

// bigramTable maps each previous byte to its most common successor in data.
// Ties keep the smaller byte. Missing contexts use fallback.
func bigramTable(data []byte, fallback byte) [256]byte {
	var cond [256][256]int
	for i := 0; i+1 < len(data); i++ {
		cond[data[i]][data[i+1]]++
	}
	var pred [256]byte
	for prev := 0; prev < 256; prev++ {
		best := int(fallback)
		bestN := cond[prev][fallback]
		for b := 0; b < 256; b++ {
			if cond[prev][b] > bestN || (cond[prev][b] == bestN && b < best) {
				best = b
				bestN = cond[prev][b]
			}
		}
		if bestN == 0 {
			pred[prev] = fallback
			continue
		}
		pred[prev] = byte(best)
	}
	return pred
}

func applyBigram(data []byte, pred [256]byte) float64 {
	if len(data) < 2 {
		return 0
	}
	correct := 0
	total := len(data) - 1
	for i := 0; i < total; i++ {
		if pred[data[i]] == data[i+1] {
			correct++
		}
	}
	return float64(correct) / float64(total)
}

// A 2-symbol program is one transition per state and tape symbol.
// Each transition is a write bit, a move bit, and a next state in 0..states.
// State 0 halts. machineCount is [4*(states+1)]^(2*states).
func machineCount(states int) int {
	radix := 4 * (states + 1)
	n := 1
	for tr := 0; tr < states*2; tr++ {
		n *= radix
	}
	return n
}

// genesFromIndex decodes program id into write, move, next triples.
// Transition 0 is state 1 reading 0. The id is a mixed-radix integer,
// least significant transition first, so id 0 halts at once.
func genesFromIndex(id, states int) []int {
	radix := 4 * (states + 1)
	base := states + 1
	g := make([]int, states*2*3)
	for tr := 0; tr < states*2; tr++ {
		digit := id % radix
		id /= radix
		next := digit % base
		digit /= base
		g[tr*3] = digit / 2
		g[tr*3+1] = digit % 2
		g[tr*3+2] = next
	}
	return g
}

// loadProgram writes the same transitions genesFromIndex encodes.
// Slots are indexed by state*2+symbol. State 0 is unused.
func loadProgram(id, states int, write []byte, move, next []int) {
	radix := 4 * (states + 1)
	base := states + 1
	for s := 1; s <= states; s++ {
		for sym := 0; sym < 2; sym++ {
			digit := id % radix
			id /= radix
			i := s*2 + sym
			next[i] = digit % base
			digit /= base
			if digit%2 == 0 {
				move[i] = -1
			} else {
				move[i] = 1
			}
			if digit/2 == 0 {
				write[i] = '0'
			} else {
				write[i] = '1'
			}
		}
	}
}

// stillRunning reports whether the loaded program is still running after
// maxSteps. HaltAt observes state 0 at the start of the next step, and this
// loop does too. The tape buffer is blank '0' outside the input.
func stillRunning(maxSteps int, write []byte, move, next []int, buf []byte, origin int, tape string) bool {
	for i := range buf {
		buf[i] = '0'
	}
	copy(buf[origin:], tape)
	head := origin
	state := 1
	for step := 0; step < maxSteps; step++ {
		if state == 0 {
			return false
		}
		sym := 0
		if buf[head] == '1' {
			sym = 1
		}
		i := state*2 + sym
		buf[head] = write[i]
		head += move[i]
		state = next[i]
	}
	return true
}

func survivesID(states, id, maxSteps int, tape string) bool {
	write := make([]byte, (states+1)*2)
	move := make([]int, (states+1)*2)
	next := make([]int, (states+1)*2)
	loadProgram(id, states, write, move, next)
	buf := make([]byte, 2*maxSteps+16)
	return stillRunning(maxSteps, write, move, next, buf, maxSteps, tape)
}

// tmBehavior is one distinct survival mask on the 256 previous-byte tapes.
// ids holds up to one program per alphabet symbol, smallest ids first.
type tmBehavior struct {
	ids  []int
	mask [4]uint64
	pop  int
}

func maskPop(mask [4]uint64) int {
	n := 0
	for _, w := range mask {
		n += bits.OnesCount64(w)
	}
	return n
}

func maskKey(mask [4]uint64) string {
	var buf [32]byte
	for i, w := range mask {
		for k := 0; k < 8; k++ {
			buf[i*8+k] = byte(w >> (8 * k))
		}
	}
	return string(buf[:])
}

func catalogBehaviors(states, maxSteps, limit int) []tmBehavior {
	n := machineCount(states)
	var (
		mu     sync.Mutex
		locals []map[string][]int
	)
	parallelN(n, func(a, b int) {
		local := make(map[string][]int)
		write := make([]byte, (states+1)*2)
		move := make([]int, (states+1)*2)
		next := make([]int, (states+1)*2)
		buf := make([]byte, 2*maxSteps+16)
		var tapes [256]string
		for t := 0; t < 256; t++ {
			tapes[t] = byteTape(byte(t))
		}
		for id := a; id < b; id++ {
			loadProgram(id, states, write, move, next)
			var mask [4]uint64
			for t := 0; t < 256; t++ {
				if stillRunning(maxSteps, write, move, next, buf, maxSteps, tapes[t]) {
					mask[t>>6] |= 1 << uint(t&63)
				}
			}
			key := maskKey(mask)
			ids := local[key]
			if len(ids) < limit {
				local[key] = append(ids, id)
			}
		}
		mu.Lock()
		locals = append(locals, local)
		mu.Unlock()
	})
	merged := make(map[string][]int)
	for _, local := range locals {
		for key, ids := range local {
			merged[key] = append(merged[key], ids...)
		}
	}
	beh := make([]tmBehavior, 0, len(merged))
	for key, ids := range merged {
		sort.Ints(ids)
		if len(ids) > limit {
			ids = append([]int(nil), ids[:limit]...)
		} else {
			ids = append([]int(nil), ids...)
		}
		var mask [4]uint64
		raw := []byte(key)
		for i := range mask {
			var w uint64
			for k := 0; k < 8; k++ {
				w |= uint64(raw[i*8+k]) << (8 * k)
			}
			mask[i] = w
		}
		beh = append(beh, tmBehavior{ids: ids, mask: mask, pop: maskPop(mask)})
	}
	sort.Slice(beh, func(i, j int) bool {
		if beh[i].pop != beh[j].pop {
			return beh[i].pop < beh[j].pop
		}
		for w := range beh[i].mask {
			if beh[i].mask[w] != beh[j].mask[w] {
				return beh[i].mask[w] < beh[j].mask[w]
			}
		}
		return false
	})
	return beh
}

// tmSearch assigns each symbol a behavior. choice holds a behavior index.
// winner[tape] is the smallest symbol still running on that previous byte, or -1.
// live[tape] is the set of symbols still running there.
type tmSearch struct {
	beh     []tmBehavior
	choice  []int
	used    []int
	winner  []int
	live    [256][4]uint64
	counts  [][]int
	correct int
	total   int
	hook    func(pass int, order string)
}

func (st *tmSearch) setWinner(tape, sym int) {
	old := st.winner[tape]
	if old == sym {
		return
	}
	if old >= 0 {
		st.correct -= st.counts[tape][old]
	}
	if sym >= 0 {
		st.correct += st.counts[tape][sym]
	}
	st.winner[tape] = sym
}

func nextLive(live [4]uint64) int {
	for w, x := range live {
		if x != 0 {
			return w<<6 + bits.TrailingZeros64(x)
		}
	}
	return -1
}

// toggle flips one symbol on one tape and repairs the winner.
func (st *tmSearch) toggle(sym, tape int) {
	word := sym >> 6
	bit := uint(sym & 63)
	st.live[tape][word] ^= 1 << bit
	on := st.live[tape][word]&(1<<bit) != 0
	if on {
		if st.winner[tape] < 0 || sym < st.winner[tape] {
			st.setWinner(tape, sym)
		}
		return
	}
	if st.winner[tape] == sym {
		st.setWinner(tape, nextLive(st.live[tape]))
	}
}

// setSymbol moves one symbol to another behavior and updates winners
// only on tapes where the two masks differ.
func (st *tmSearch) setSymbol(sym, b int) {
	old := st.choice[sym]
	if old == b {
		return
	}
	st.used[old]--
	st.used[b]++
	st.choice[sym] = b
	for w := 0; w < 4; w++ {
		diff := st.beh[old].mask[w] ^ st.beh[b].mask[w]
		for diff != 0 {
			bit := bits.TrailingZeros64(diff)
			st.toggle(sym, w<<6+bit)
			diff &^= 1 << uint(bit)
		}
	}
}

// init places every symbol on a distinct always-halt program.
// Extra symbols, when that mask runs out of programs, are the largest bytes.
func (st *tmSearch) init() error {
	if len(st.beh) == 0 || st.beh[0].pop != 0 {
		return fmt.Errorf("no always-halt program")
	}
	n := len(st.choice)
	st.used[0] = n
	cap0 := len(st.beh[0].ids)
	if cap0 >= n {
		return nil
	}
	bi := 1
	filled := 0
	for sym := n - 1; sym >= cap0; sym-- {
		for bi < len(st.beh) && filled >= len(st.beh[bi].ids) {
			bi++
			filled = 0
		}
		if bi >= len(st.beh) {
			return fmt.Errorf("not enough distinct programs for %d symbols", n)
		}
		st.setSymbol(sym, bi)
		filled++
	}
	return nil
}

// improve tries every behavior for one symbol. The mask that adds the most
// correct next bytes wins. An equal score keeps the mask that survives on
// fewer tapes, so a rare byte does not stay running everywhere.
func (st *tmSearch) improve(sym int) bool {
	cur := st.choice[sym]
	bestB := cur
	bestC := st.correct
	bestPop := st.beh[cur].pop
	for b := range st.beh {
		if b != cur && st.used[b] >= len(st.beh[b].ids) {
			continue
		}
		st.setSymbol(sym, b)
		pop := st.beh[b].pop
		if st.correct > bestC || (st.correct == bestC && pop < bestPop) {
			bestC = st.correct
			bestPop = pop
			bestB = b
		}
		st.setSymbol(sym, cur)
	}
	if bestB == cur {
		return false
	}
	st.setSymbol(sym, bestB)
	return true
}

func (st *tmSearch) sweep(order []int) bool {
	changed := false
	for _, sym := range order {
		if st.improve(sym) {
			changed = true
		}
	}
	return changed
}

func (st *tmSearch) recount() int {
	n := 0
	for tape, sym := range st.winner {
		if sym >= 0 {
			n += st.counts[tape][sym]
		}
	}
	return n
}

func assignPrograms(choice []int, beh []tmBehavior, states int) ([]int, error) {
	cursor := make([]int, len(beh))
	seen := make(map[int]bool, len(choice))
	per := states * 2 * 3
	genome := make([]int, len(choice)*per)
	for sym, b := range choice {
		if cursor[b] >= len(beh[b].ids) {
			return nil, fmt.Errorf("behavior %d ran out of programs", b)
		}
		id := beh[b].ids[cursor[b]]
		cursor[b]++
		if seen[id] {
			return nil, fmt.Errorf("program %d assigned twice", id)
		}
		seen[id] = true
		copy(genome[sym*per:(sym+1)*per], genesFromIndex(id, states))
	}
	return genome, nil
}

func predictCounts(data, alphabet []byte, winner []int) (correct, total, missed int) {
	if len(data) < 2 {
		return 0, 0, 0
	}
	total = len(data) - 1
	for i := 0; i < total; i++ {
		sym := winner[data[i]]
		if sym < 0 {
			missed++
			continue
		}
		if alphabet[sym] == data[i+1] {
			correct++
		}
	}
	return correct, total, missed
}

// confirmMasks checks the fast simulator against HaltAt on the tapes that
// occur as a previous byte. The reported accuracy uses those masks.
func confirmMasks(states, maxSteps int, machines []string, choice []int, beh []tmBehavior, tapes []byte) error {
	sim := &UniversalBoundedSimulator{MaxSteps: maxSteps}
	var seen [256]bool
	for _, t := range tapes {
		seen[t] = true
	}
	for sym, m := range machines {
		mask := beh[choice[sym]].mask
		for t := 0; t < 256; t++ {
			if !seen[byte(t)] {
				continue
			}
			want := mask[t>>6]&(1<<uint(t&63)) != 0
			got := sim.HaltAt(m, byteTape(byte(t))) < 0
			if got != want {
				return fmt.Errorf("symbol %d tape %d survival %v, simulator %v", sym, t, want, got)
			}
		}
	}
	return nil
}

// NetLMConfig searches one distinct 2-symbol machine per alphabet symbol.
// States 0 tries 1-state and 2-state programs and keeps the higher accuracy.
type NetLMConfig struct {
	Data     []byte
	DataPath string
	States   int
	MaxSteps int
	Bytes    int
	ValBytes int
	GenBytes int
	OutDir   string
	Log      func(string)
}

// NetLMResult is the chosen assignment and its next-byte accuracy.
type NetLMResult struct {
	TrainBytes  int     `json:"train_bytes"`
	ValBytes    int     `json:"val_bytes"`
	FitBytes    int     `json:"fit_bytes"`
	ScoreBytes  int     `json:"score_bytes"`
	Alphabet    int     `json:"alphabet"`
	Programs    int     `json:"programs"`
	Behaviors   int     `json:"behaviors"`
	States      int     `json:"states"`
	OracleSteps int     `json:"oracle_steps"`
	Genes       int     `json:"genes"`
	FitAcc      float64 `json:"fit_acc"`
	ValAcc      float64 `json:"val_acc"`
	FitUnigram  float64 `json:"fit_unigram"`
	ValUnigram  float64 `json:"val_unigram"`
	FitBigram   float64 `json:"fit_bigram"`
	ValBigram   float64 `json:"val_bigram"`
	FitMiss     float64 `json:"fit_miss"`
	ValMiss     float64 `json:"val_miss"`
	Mode        byte    `json:"mode"`
	Prompt      string  `json:"prompt"`
	Sample      string  `json:"sample"`
	Genome      []int   `json:"genome"`
}

func (cfg *NetLMConfig) logf(format string, args ...any) {
	if cfg.Log != nil {
		cfg.Log(fmt.Sprintf(format, args...))
	}
}

func (cfg *NetLMConfig) applyDefaults() {
	if cfg.States < 0 {
		cfg.States = 0
	}
	if cfg.MaxSteps <= 0 {
		cfg.MaxSteps = 8
	}
	if cfg.Bytes <= 0 {
		cfg.Bytes = 20000
	}
	if cfg.ValBytes <= 0 {
		cfg.ValBytes = 2000
	}
	if cfg.GenBytes < 0 {
		cfg.GenBytes = 200
	}
}

var errTooManySymbols = errors.New("alphabet exceeds the number of programs")

func newSymbolSearch(alphabet []byte, fit []byte, states, maxSteps int) (*tmSearch, error) {
	programs := machineCount(states)
	if len(alphabet) > programs {
		return nil, fmt.Errorf("%w: %d symbols, %d programs", errTooManySymbols, len(alphabet), programs)
	}
	beh := catalogBehaviors(states, maxSteps, len(alphabet))
	counts := make([][]int, 256)
	for i := range counts {
		counts[i] = make([]int, len(alphabet))
	}
	index := make([]int, 256)
	for i := range index {
		index[i] = -1
	}
	for i, b := range alphabet {
		index[b] = i
	}
	total := 0
	if len(fit) >= 2 {
		total = len(fit) - 1
		for i := 0; i < total; i++ {
			counts[fit[i]][index[fit[i+1]]]++
		}
	}
	winner := make([]int, 256)
	for i := range winner {
		winner[i] = -1
	}
	st := &tmSearch{
		beh:    beh,
		choice: make([]int, len(alphabet)),
		used:   make([]int, len(beh)),
		winner: winner,
		counts: counts,
		total:  total,
	}
	if err := st.init(); err != nil {
		return nil, err
	}
	if st.recount() != st.correct {
		return nil, fmt.Errorf("initial score drifted")
	}
	return st, nil
}

func (st *tmSearch) assign() error {
	n := len(st.choice)
	freq := make([]int, n)
	for tape := range st.counts {
		for sym, c := range st.counts[tape] {
			freq[sym] += c
		}
	}
	byFreq := make([]int, n)
	byByte := make([]int, n)
	for i := range byByte {
		byByte[i] = i
		byFreq[i] = i
	}
	sort.SliceStable(byFreq, func(i, j int) bool {
		return freq[byFreq[i]] > freq[byFreq[j]]
	})
	// Frequency order lets a common symbol claim a tape before a smaller
	// byte can take it. Byte order then gives the smaller byte a chance to
	// release any tape a larger symbol predicts better. Each move raises the
	// number of correct bytes, or keeps that number and survives on fewer tapes.
	for pass := 1; pass <= 24; pass++ {
		moved := st.sweep(byFreq)
		if st.hook != nil {
			st.hook(pass, "frequency")
		}
		moved = st.sweep(byByte) || moved
		if st.hook != nil {
			st.hook(pass, "byte")
		}
		if !moved || st.correct == st.total {
			break
		}
	}
	if st.recount() != st.correct {
		return fmt.Errorf("search score drifted")
	}
	return nil
}

func (st *tmSearch) logPasses(cfg *NetLMConfig) {
	st.hook = func(pass int, order string) {
		acc := 0.0
		if st.total > 0 {
			acc = float64(st.correct) / float64(st.total)
		}
		cfg.logf("  pass %d %s %.4f", pass, order, acc)
	}
}

func searchSymbols(cfg *NetLMConfig, train, val, fit, score []byte, states int, mode byte, fitUni float64, biTab [256]byte) (*NetLMResult, error) {
	alphabet := collectAlphabet(fit)
	spec := symbolSpec{Alphabet: alphabet, States: states, MaxSteps: cfg.MaxSteps}
	if err := spec.validate(); err != nil {
		return nil, err
	}
	programs := machineCount(states)
	if len(alphabet) > programs {
		return nil, fmt.Errorf("%w: %d symbols, %d programs", errTooManySymbols, len(alphabet), programs)
	}
	cfg.logf("%d-state catalog, %d programs", states, programs)
	st, err := newSymbolSearch(alphabet, fit, states, cfg.MaxSteps)
	if err != nil {
		return nil, err
	}
	cfg.logf("%d-state search", states)
	cfg.logf("  programs %d, behaviors %d", machineCount(states), len(st.beh))
	st.logPasses(cfg)
	if err := st.assign(); err != nil {
		return nil, err
	}
	inUse := 0
	for _, n := range st.used {
		if n > 0 {
			inUse++
		}
	}
	cfg.logf("  behaviors in use %d", inUse)
	genome, err := assignPrograms(st.choice, st.beh, states)
	if err != nil {
		return nil, err
	}
	machines := decodeSymbols(genome, spec)
	observed := make([]byte, 0, len(alphabet))
	var seen [256]bool
	for _, b := range fit {
		if !seen[b] {
			seen[b] = true
			observed = append(observed, b)
		}
	}
	for _, b := range score {
		if !seen[b] {
			seen[b] = true
			observed = append(observed, b)
		}
	}
	if err := confirmMasks(states, cfg.MaxSteps, machines, st.choice, st.beh, observed); err != nil {
		return nil, err
	}
	fitCorrect, fitTotal, fitMissN := predictCounts(fit, alphabet, st.winner)
	if fitCorrect != st.correct {
		return nil, fmt.Errorf("predict %d disagrees with search %d", fitCorrect, st.correct)
	}
	res := &NetLMResult{
		TrainBytes:  len(train),
		ValBytes:    len(val),
		FitBytes:    len(fit),
		ScoreBytes:  len(score),
		Alphabet:    len(alphabet),
		Programs:    machineCount(states),
		Behaviors:   len(st.beh),
		States:      states,
		OracleSteps: cfg.MaxSteps,
		Genes:       len(genome),
		Mode:        mode,
		Genome:      genome,
		FitUnigram:  fitUni,
		FitBigram:   applyBigram(fit, biTab),
	}
	if fitTotal > 0 {
		res.FitAcc = float64(fitCorrect) / float64(fitTotal)
		res.FitMiss = float64(fitMissN) / float64(fitTotal)
	}
	if len(score) >= 2 {
		correct, total, miss := predictCounts(score, alphabet, st.winner)
		res.ValAcc = float64(correct) / float64(total)
		res.ValMiss = float64(miss) / float64(total)
		res.ValUnigram = applyUnigram(score, mode)
		res.ValBigram = applyBigram(score, biTab)
	}
	promptN := 32
	if promptN > len(fit) {
		promptN = len(fit)
	}
	res.Prompt = string(fit[:promptN])
	if cfg.GenBytes > 0 && promptN > 0 {
		sim := &UniversalBoundedSimulator{MaxSteps: cfg.MaxSteps}
		res.Sample = string(continueSymbols(sim, machines, alphabet, fit[:promptN], cfg.GenBytes, mode))
	}
	cfg.logf("fit acc %.4f  unigram %.4f  bigram %.4f  all halted %.4f", res.FitAcc, res.FitUnigram, res.FitBigram, res.FitMiss)
	cfg.logf("val acc %.4f  unigram %.4f  bigram %.4f  all halted %.4f", res.ValAcc, res.ValUnigram, res.ValBigram, res.ValMiss)
	if res.Sample != "" {
		cfg.logf("prompt %s", strconv.Quote(res.Prompt))
		cfg.logf("sample %s", strconv.Quote(res.Sample))
	}
	return res, nil
}

// NetLM searches one distinct 2-symbol machine per symbol.
// Fitness is the fraction of next bytes equal to the symbol whose machine
// is still running at the step bound.
func NetLM(cfg NetLMConfig) (*NetLMResult, error) {
	cfg.applyDefaults()
	if cfg.States > 2 {
		return nil, fmt.Errorf("states must be 1 or 2, or 0 to try both")
	}
	data := cfg.Data
	if data == nil {
		if cfg.DataPath == "" {
			return nil, fmt.Errorf("no training data")
		}
		b, err := os.ReadFile(cfg.DataPath)
		if err != nil {
			return nil, err
		}
		data = b
	}
	train, val, err := split90(data)
	if err != nil {
		return nil, err
	}
	fit := train
	if cfg.Bytes < len(fit) {
		fit = fit[:cfg.Bytes]
	}
	if len(fit) < 2 {
		return nil, fmt.Errorf("need at least 2 training bytes, got %d", len(fit))
	}
	score := val
	if cfg.ValBytes < len(score) {
		score = score[:cfg.ValBytes]
	}
	mode, fitUni := majorityByte(fit)
	biTab := bigramTable(fit, mode)
	cfg.logf("symbol halting oracles")
	cfg.logf("  train %d bytes, val %d bytes, fitness %d bytes, val score %d bytes", len(train), len(val), len(fit), len(score))
	cfg.logf("  alphabet %d, %d-step bound", len(collectAlphabet(fit)), cfg.MaxSteps)
	cfg.logf("  each symbol gets a different 2-symbol machine")
	cfg.logf("  the previous byte is the tape; a machine that does not halt generates its symbol")
	cfg.logf("  unigram %.4f (%q)  bigram %.4f on the fitness bytes", fitUni, mode, applyBigram(fit, biTab))

	states := []int{cfg.States}
	if cfg.States == 0 {
		states = []int{1, 2}
	}
	var best *NetLMResult
	ran := 0
	for _, st := range states {
		res, err := searchSymbols(&cfg, train, val, fit, score, st, mode, fitUni, biTab)
		if err != nil {
			if cfg.States == 0 && errors.Is(err, errTooManySymbols) {
				cfg.logf("%d-state skipped: %v", st, err)
				continue
			}
			return nil, err
		}
		ran++
		if best == nil || res.FitAcc > best.FitAcc || (res.FitAcc == best.FitAcc && res.ValAcc > best.ValAcc) {
			best = res
		}
	}
	if best == nil {
		return nil, fmt.Errorf("no state count covers the alphabet")
	}
	if ran > 1 {
		cfg.logf("kept %d-state machines, fit %.4f, val %.4f", best.States, best.FitAcc, best.ValAcc)
	}
	if cfg.OutDir != "" {
		path := filepath.Join(cfg.OutDir, "netlm.json")
		if err := writeResult(path, best); err != nil {
			return nil, err
		}
		cfg.logf("wrote %s", path)
	}
	return best, nil
}

func netLMFromArgs(args []string) error {
	fs := flag.NewFlagSet("netlm", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: oracle netlm [flags]\n\nTrain one halting oracle per symbol.\nEach symbol is a different 2-symbol Turing machine. The machine reads the previous byte.\nThe symbol whose machine does not halt is generated. The simulator is step-bounded.\n\n")
		fs.PrintDefaults()
	}
	data := fs.String("data", "100.txt.utf-8", "UTF-8 text file")
	states := fs.Int("states", 0, "states in each symbol machine; 0 tries 1 and 2")
	steps := fs.Int("oracle-steps", 8, "maximum simulator steps")
	nbytes := fs.Int("bytes", 20000, "training-byte prefix used for fitness")
	valBytes := fs.Int("val-bytes", 2000, "validation-byte prefix used for the report")
	genBytes := fs.Int("gen-bytes", 200, "closed-loop bytes to print after the prompt")
	out := fs.String("out", "checkpoints", "directory for netlm.json; empty skips the write")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	logFn := func(msg string) {
		fmt.Fprintln(os.Stdout, msg)
		_ = os.Stdout.Sync()
	}
	_, err := NetLM(NetLMConfig{
		DataPath: *data,
		States:   *states,
		MaxSteps: *steps,
		Bytes:    *nbytes,
		ValBytes: *valBytes,
		GenBytes: *genBytes,
		OutDir:   *out,
		Log:      logFn,
	})
	return err
}
