// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
)

// MCTSConfig samples Sims continuations and keeps the one with the highest score.
// Each byte of a rollout is drawn from softmax(logits / Temp).
// The score is the sum of the log probabilities of the bytes in that rollout.
type MCTSConfig struct {
	Sims int
	Temp float64
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

// sampleRollout draws n bytes and returns them with the sum of their log probabilities.
func sampleRollout(prompt []byte, n int, temp float64, rng *rand.Rand, logitsOf func([]byte) []float32) ([]byte, float64) {
	buf := append([]byte{}, prompt...)
	out := make([]byte, 0, n)
	var sum float64
	for i := 0; i < n; i++ {
		probs := byteDist(logitsOf(buf), temp)
		tok := sampleDist(probs, rng)
		p := 0.0
		if tok >= 0 && tok < len(probs) {
			p = probs[tok]
		}
		if p > 0 {
			sum += math.Log(p)
		} else {
			sum = math.Inf(-1)
		}
		out = append(out, byte(tok))
		buf = append(buf, byte(tok))
	}
	return out, sum
}

// searchMCTS returns the sampled rollout whose log-probability sum is highest.
func searchMCTS(prompt []byte, n int, cfg MCTSConfig, rng *rand.Rand, logitsOf func([]byte) []float32) []byte {
	ensure(cfg.Sims >= 1, "mcts sims %d", cfg.Sims)
	ensure(n >= 1, "mcts n %d", n)
	if len(prompt) == 0 {
		prompt = []byte{'\n'}
	}
	best, bestSum := sampleRollout(prompt, n, cfg.Temp, rng, logitsOf)
	for s := 1; s < cfg.Sims; s++ {
		roll, sum := sampleRollout(prompt, n, cfg.Temp, rng, logitsOf)
		if sum > bestSum {
			best = roll
			bestSum = sum
		}
	}
	return best
}

// GenerateMCTS continues prompt by n bytes.
// It samples Sims rollouts from the temperature softmax and returns the one
// with the highest sum of log probabilities.
func (m *OracleTransformer) GenerateMCTS(prompt []byte, n int, cfg MCTSConfig, rng *rand.Rand) []byte {
	was := autogradEnabled
	autogradEnabled = false
	defer func() { autogradEnabled = was }()
	return searchMCTS(prompt, n, cfg, rng, m.nextLogits)
}
