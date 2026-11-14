// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"math/rand"
	"testing"
)

// A one-step greedy choice prefers token 0. Two-step search prefers token 1,
// because the model is confident about the byte that follows it.
func TestMCTSPrefersLookahead(t *testing.T) {
	prompt := []byte{'X'}
	logitsOf := func(buf []byte) []float32 {
		if bytes.Equal(buf, prompt) {
			return []float32{0, 0}
		}
		if buf[len(buf)-1] == 0 {
			return []float32{0, 0}
		}
		return []float32{6, -6}
	}
	cfg := MCTSConfig{Sims: 24, Width: 2, Horizon: 2, PUCT: 1.5, Temp: 1}
	got := searchMCTS(prompt, 1, cfg, rand.New(rand.NewSource(1)), logitsOf)
	if len(got) != 1 || got[0] != 1 {
		t.Fatalf("got %v", got)
	}
}

func TestMCTSDeterministic(t *testing.T) {
	prompt := []byte("abc")
	logitsOf := func(buf []byte) []float32 {
		out := []float32{0.1, 0.4, -0.2, 1.2, 0.3}
		out[len(buf)%len(out)] += 0.7
		return out
	}
	cfg := MCTSConfig{Sims: 6, Width: 3, Horizon: 3, PUCT: 1.5, Temp: 0.8}
	a := searchMCTS(prompt, 5, cfg, rand.New(rand.NewSource(3)), logitsOf)
	b := searchMCTS(prompt, 5, cfg, rand.New(rand.NewSource(3)), logitsOf)
	c := searchMCTS(prompt, 5, cfg, rand.New(rand.NewSource(4)), logitsOf)
	if !bytes.Equal(a, b) {
		t.Fatalf("same seed differed: %v %v", a, b)
	}
	if len(a) != 5 {
		t.Fatalf("len %d", len(a))
	}
	if bytes.Equal(a, c) {
		t.Fatal("different seeds matched")
	}
}

func TestGenerateMCTSLength(t *testing.T) {
	autogradEnabled = true
	rng := rand.New(rand.NewSource(6))
	cfg := defaultConfig()
	cfg.Ctx = 16
	cfg.DModel = 32
	cfg.Layers = 1
	cfg.Heads = 4
	cfg.OracleInput = 4
	cfg.MaxSteps = 8
	m, err := NewModel(cfg, rng)
	if err != nil {
		t.Fatal(err)
	}
	mc := MCTSConfig{Sims: 2, Width: 2, Horizon: 2, PUCT: 1.5, Temp: 1}
	sample := m.GenerateMCTS([]byte("To be"), 4, mc, rand.New(rand.NewSource(1)))
	again := m.GenerateMCTS([]byte("To be"), 4, mc, rand.New(rand.NewSource(1)))
	if len(sample) != 4 || !bytes.Equal(sample, again) {
		t.Fatalf("sample %q again %q", sample, again)
	}
	if !autogradEnabled {
		t.Fatal("mcts left autograd off")
	}
}
