// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
)

// lmOrderMax is the longest context the head may read. Contexts are stored
// sparsely, one node per byte string that occurred, so orders above 2 do
// not allocate a cell for every possible string.
const lmOrderMax = 8

// lmSampleLen is the number of new bytes drawn after the prompt.
const lmSampleLen = 160

// lmNode is one context. Children extend it by one older byte. yb and yn are
// the distinct bytes written after this context and how often each was written.
type lmNode struct {
	kb    []byte
	ch    []*lmNode
	yb    []byte
	yn    []uint32
	total uint32
	mu    float64
}

func (n *lmNode) find(b byte) *lmNode {
	for i, x := range n.kb {
		if x == b {
			return n.ch[i]
		}
	}
	return nil
}

func (n *lmNode) grow(b byte) *lmNode {
	if c := n.find(b); c != nil {
		return c
	}
	c := &lmNode{}
	n.kb = append(n.kb, b)
	n.ch = append(n.ch, c)
	return c
}

func (n *lmNode) addY(y byte) {
	n.total++
	for i, b := range n.yb {
		if b == y {
			n.yn[i]++
			return
		}
	}
	n.yb = append(n.yb, y)
	n.yn = append(n.yn, 1)
}

// mode is the most common byte written after this context. An exact tie
// picks the lower byte.
func (n *lmNode) mode() byte {
	var best byte
	var count uint32
	for i, y := range n.yb {
		c := n.yn[i]
		if c > count || (c == count && (i == 0 || y < best)) {
			count = c
			best = y
		}
	}
	return best
}

// gaussLM is a byte language model. The head reads the previous bytes.
// Each context keeps the bytes that were written after it. The Gaussian
// write weight is trained on the most common of those bytes, and the hard
// read is the nearest byte to that weight. A context that was never written
// backs off to the longest suffix that was.
type gaussLM struct {
	tau   float64
	order int
	root  lmNode
	seen  []int
}

func newGaussLM(order int, tau float64) *gaussLM {
	return &gaussLM{
		tau:   tau,
		order: order,
		seen:  make([]int, order+1),
	}
}

// fitContextMean follows the Gaussian-softmax partial until the expected
// byte matches the observed mean. The step is τ^2, which cancels 1/τ^2.
// The weight may leave the byte range. That is where a mean near 0 or 255
// has its expectation.
func fitContextMean(mean, tau float64, iters int) float64 {
	mu := 0.0
	for step := 0; step < iters; step++ {
		_, dmu := symbolLoss(mu, tau, 0)
		expect := dmu * tau * tau
		delta := expect - mean
		if delta < 1e-4 && delta > -1e-4 {
			break
		}
		mu -= delta
	}
	return mu
}

// observe counts the bytes written after each context and fits one Gaussian
// weight to the most common of them. The average byte code is not the target:
// that average sits between letters, so the hard read misses the text.
// The trie is rooted at the most recent byte, so a missing longer context
// stops on the longest suffix that was written.
func (m *gaussLM) observe(data []byte, iters int) {
	for i, y := range data {
		node := &m.root
		node.addY(y)
		maxK := m.order
		if i < maxK {
			maxK = i
		}
		for k := 1; k <= maxK; k++ {
			node = node.grow(data[i-k])
			node.addY(y)
		}
	}
	m.root.fit(m.tau, iters, 0, m.seen)
}

func (n *lmNode) fit(tau float64, iters, depth int, seen []int) {
	if n.total == 0 {
		return
	}
	n.mu = fitContextMean(float64(n.mode()), tau, iters)
	if depth < len(seen) {
		seen[depth]++
	}
	for _, c := range n.ch {
		c.fit(tau, iters, depth+1, seen)
	}
}

// weight is the Gaussian write for ctx in chronological order, oldest byte
// first. It is the weight tests read back.
func (m *gaussLM) weight(ctx []byte) float64 {
	n := &m.root
	for i := len(ctx) - 1; i >= 0; i-- {
		c := n.find(ctx[i])
		if c == nil {
			return 0
		}
		n = c
	}
	return n.mu
}

// longest is the deepest stored context at i, reading at most order bytes.
func (m *gaussLM) longest(data []byte, i, order int) *lmNode {
	if order > m.order {
		order = m.order
	}
	node := &m.root
	for k := 1; k <= order && i >= k; k++ {
		c := node.find(data[i-k])
		if c == nil {
			break
		}
		node = c
	}
	return node
}

// score is the mean cross-entropy and the nearest-byte accuracy of data[start:].
// The first byte of a slice still has the unigram context.
func (m *gaussLM) score(data []byte, order int) (nll float64, correct, total int) {
	type key struct {
		n *lmNode
		y byte
	}
	cache := make(map[key]float64)
	for i := range data {
		node := m.longest(data, i, order)
		y := data[i]
		ck := key{node, y}
		loss, ok := cache[ck]
		if !ok {
			loss, _ = symbolLoss(node.mu, m.tau, int(y))
			cache[ck] = loss
		}
		nll += loss
		total++
		if nearestByte(node.mu) == y {
			correct++
		}
	}
	if total > 0 {
		nll /= float64(total)
	}
	return nll, correct, total
}

// sampleWritten draws one of the bytes written in this context, in proportion
// to how often it was written. The longest stored context is used.
func (m *gaussLM) sampleWritten(data []byte, i int, rng *rand.Rand) byte {
	return drawNode(m.longest(data, i, m.order), rng)
}

func drawNode(n *lmNode, rng *rand.Rand) byte {
	if n == nil || n.total == 0 {
		return 0
	}
	draw := uint32(rng.Int63n(int64(n.total)))
	var acc uint32
	for i, c := range n.yn {
		acc += c
		if draw < acc {
			return n.yb[i]
		}
	}
	return n.yb[len(n.yb)-1]
}

// continueText appends n bytes drawn from the bytes written after the context
// the head has just read. The prompt is the tape so far.
func (m *gaussLM) continueText(prompt []byte, n int, rng *rand.Rand) []byte {
	out := append([]byte(nil), prompt...)
	for i := 0; i < n; i++ {
		out = append(out, m.sampleWritten(out, len(out), rng))
	}
	return out
}

type lmConfig struct {
	Path  string
	Order int
	Iters int
	Temp  float64
	Seed  int64
	Log   func(string)
}

func (cfg lmConfig) logf(format string, args ...any) {
	if cfg.Log != nil {
		cfg.Log(fmt.Sprintf(format, args...))
	}
}

func learnGaussLM(cfg lmConfig) error {
	if cfg.Order < 0 || cfg.Order > lmOrderMax {
		return fmt.Errorf("order must be from 0 to %d", lmOrderMax)
	}
	if cfg.Iters <= 0 {
		cfg.Iters = 32
	}
	if cfg.Temp <= 0 {
		return fmt.Errorf("temperature must be positive")
	}
	data, err := os.ReadFile(cfg.Path)
	if err != nil {
		return err
	}
	train, val, err := split90(data)
	if err != nil {
		return err
	}
	cfg.logf("gaussian language model")
	cfg.logf("  file %s", cfg.Path)
	cfg.logf("  %d bytes, train %d, val %d, order %d", len(data), len(train), len(val), cfg.Order)
	cfg.logf("  the head reads the previous %d bytes", cfg.Order)
	cfg.logf("  each context's weight is trained on its most common next byte")
	cfg.logf("  the hard read is the nearest byte to that weight")
	cfg.logf("  an unseen context backs off to the longest suffix that was written")
	cfg.logf("  loss is the cross-entropy of a Gaussian softmax over byte codes")
	m := newGaussLM(cfg.Order, cfg.Temp)
	m.observe(train, cfg.Iters)
	for k := 0; k <= cfg.Order; k++ {
		cfg.logf("  order %d contexts %d", k, m.seen[k])
	}
	trLoss, trHit, trN := m.score(train, cfg.Order)
	vaLoss, vaHit, vaN := m.score(val, cfg.Order)
	_, uniTr, _ := m.score(train, 0)
	_, uniVa, _ := m.score(val, 0)
	cfg.logf("train loss %.4f  acc %d/%d", trLoss, trHit, trN)
	cfg.logf("val   loss %.4f  acc %d/%d", vaLoss, vaHit, vaN)
	cfg.logf("unigram acc train %d/%d  val %d/%d  byte %q", uniTr, trN, uniVa, vaN, nearestByte(m.root.mu))
	prompt := []byte("To be, or not to be")
	rng := rand.New(rand.NewSource(cfg.Seed))
	sample := m.continueText(prompt, lmSampleLen, rng)
	cfg.logf("sample %s", strconv.Quote(string(sample)))
	return nil
}
