// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
	"sort"
)

// MCTSConfig is a per-byte Monte Carlo tree search over the model's next-byte distribution.
// Each simulation expands the Width most likely bytes, looks ahead Horizon bytes,
// and scores the path by the average probability the model assigned to those bytes.
// The committed byte is the root child with the most visits.
type MCTSConfig struct {
	Sims    int
	Width   int
	Horizon int
	PUCT    float64
	Temp    float64
}

type mctsNode struct {
	tok      int
	prior    float64
	prob     float64
	visits   int
	valueSum float64
	children []*mctsNode
}

func (n *mctsNode) q() float64 {
	if n.visits == 0 {
		return 0
	}
	return n.valueSum / float64(n.visits)
}

// bestChild is the PUCT choice. Unvisited nodes have Q = 0, so the prior term
// orders the first visit. A parent with no completed simulations uses a visit
// count of 1 so that term is not zero.
func (n *mctsNode) bestChild(c float64) *mctsNode {
	parentN := n.visits
	if parentN < 1 {
		parentN = 1
	}
	sqrtN := math.Sqrt(float64(parentN))
	best := n.children[0]
	bestScore := math.Inf(-1)
	for _, ch := range n.children {
		score := ch.q() + c*ch.prior*sqrtN/(1+float64(ch.visits))
		if score > bestScore || (score == bestScore && ch.tok < best.tok) {
			best = ch
			bestScore = score
		}
	}
	return best
}

func (n *mctsNode) chosen() *mctsNode {
	var best *mctsNode
	for _, ch := range n.children {
		if ch.visits == 0 {
			continue
		}
		if best == nil || ch.visits > best.visits ||
			(ch.visits == best.visits && ch.q() > best.q()) ||
			(ch.visits == best.visits && ch.q() == best.q() && ch.tok < best.tok) {
			best = ch
		}
	}
	return best
}

func byteDist(logits []float32, temp float64) []float64 {
	probs := make([]float64, len(logits))
	if len(logits) == 0 {
		return probs
	}
	if temp <= 0 {
		best := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		probs[best] = 1
		return probs
	}
	maxv := float64(logits[0])
	for _, v := range logits[1:] {
		if float64(v) > maxv {
			maxv = float64(v)
		}
	}
	var sum float64
	for i, v := range logits {
		e := math.Exp((float64(v) - maxv) / temp)
		probs[i] = e
		sum += e
	}
	if sum == 0 {
		u := 1 / float64(len(probs))
		for i := range probs {
			probs[i] = u
		}
		return probs
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

func sampleDist(probs []float64, rng *rand.Rand) int {
	r := rng.Float64()
	var c float64
	for i, p := range probs {
		c += p
		if r <= c {
			return i
		}
	}
	return len(probs) - 1
}

func topChildren(probs []float64, width int) []*mctsNode {
	if width < 1 {
		width = 1
	}
	idx := make([]int, len(probs))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool {
		if probs[idx[i]] == probs[idx[j]] {
			return idx[i] < idx[j]
		}
		return probs[idx[i]] > probs[idx[j]]
	})
	if width > len(idx) {
		width = len(idx)
	}
	var sum float64
	for i := 0; i < width; i++ {
		sum += probs[idx[i]]
	}
	kids := make([]*mctsNode, 0, width)
	for i := 0; i < width; i++ {
		p := probs[idx[i]]
		if p == 0 || sum == 0 {
			continue
		}
		kids = append(kids, &mctsNode{tok: idx[i], prior: p / sum, prob: p})
	}
	return kids
}

// mctsSimulate walks one PUCT path, expands a leaf, and backs up the mean
// probability of the lookahead bytes.
func mctsSimulate(root *mctsNode, prefix []byte, cfg MCTSConfig, rng *rand.Rand, logitsOf func([]byte) []float32) {
	node := root
	path := []*mctsNode{root}
	buf := append([]byte{}, prefix...)
	for len(node.children) > 0 && len(path)-1 < cfg.Horizon {
		node = node.bestChild(cfg.PUCT)
		path = append(path, node)
		buf = append(buf, byte(node.tok))
	}
	if len(path)-1 < cfg.Horizon && len(node.children) == 0 {
		node.children = topChildren(byteDist(logitsOf(buf), cfg.Temp), cfg.Width)
		if len(node.children) > 0 {
			node = node.bestChild(cfg.PUCT)
			path = append(path, node)
			buf = append(buf, byte(node.tok))
		}
	}
	acc := 0.0
	count := 0
	for _, nd := range path[1:] {
		acc += nd.prob
		count++
	}
	for count < cfg.Horizon {
		probs := byteDist(logitsOf(buf), cfg.Temp)
		if len(probs) == 0 {
			break
		}
		tok := sampleDist(probs, rng)
		acc += probs[tok]
		count++
		buf = append(buf, byte(tok))
	}
	value := 0.0
	if count > 0 {
		value = acc / float64(count)
	}
	for _, nd := range path {
		nd.visits++
		nd.valueSum += value
	}
}

// searchMCTS returns n bytes. logitsOf is the next-byte logit vector at a prefix.
func searchMCTS(prompt []byte, n int, cfg MCTSConfig, rng *rand.Rand, logitsOf func([]byte) []float32) []byte {
	ensure(cfg.Sims >= 1, "mcts sims %d", cfg.Sims)
	ensure(cfg.Width >= 1, "mcts width %d", cfg.Width)
	ensure(cfg.Horizon >= 1, "mcts horizon %d", cfg.Horizon)
	ensure(cfg.PUCT >= 0, "mcts puct %v", cfg.PUCT)
	if len(prompt) == 0 {
		prompt = []byte{'\n'}
	}
	buf := append([]byte{}, prompt...)
	out := make([]byte, 0, n)
	for len(out) < n {
		root := &mctsNode{}
		for s := 0; s < cfg.Sims; s++ {
			mctsSimulate(root, buf, cfg, rng, logitsOf)
		}
		ch := root.chosen()
		var tok int
		if ch == nil {
			tok = sampleDist(byteDist(logitsOf(buf), cfg.Temp), rng)
		} else {
			tok = ch.tok
		}
		out = append(out, byte(tok))
		buf = append(buf, byte(tok))
	}
	return out
}

// GenerateMCTS continues prompt by n bytes. Each byte is the most-visited
// child of a fresh tree rooted at the text so far.
func (m *OracleTransformer) GenerateMCTS(prompt []byte, n int, cfg MCTSConfig, rng *rand.Rand) []byte {
	was := autogradEnabled
	autogradEnabled = false
	defer func() { autogradEnabled = was }()
	return searchMCTS(prompt, n, cfg, rng, m.nextLogits)
}
