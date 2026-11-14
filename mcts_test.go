// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

// Token 1 followed by token 0 has a higher log-probability sum than any
// rollout that starts with token 0. The search must return that rollout.
func TestMCTSPicksHighestSum(t *testing.T) {
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
	cfg := MCTSConfig{Sims: 32, Temp: 1}
	got := searchMCTS(prompt, 2, cfg, rand.New(rand.NewSource(1)), logitsOf)
	if !bytes.Equal(got, []byte{1, 0}) {
		t.Fatalf("got %v", got)
	}
}

func TestRolloutSumIsLogProb(t *testing.T) {
	prompt := []byte{'X'}
	logitsOf := func(buf []byte) []float32 {
		return []float32{0, 0}
	}
	// temp 0 is greedy on the first class, whose probability is 1, so the sum is 0.
	_, sum := sampleRollout(prompt, 3, 0, rand.New(rand.NewSource(1)), logitsOf)
	if sum != 0 {
		t.Fatalf("sum %v", sum)
	}
	roll, sum := sampleRollout(prompt, 4, 1, rand.New(rand.NewSource(2)), logitsOf)
	want := float64(len(roll)) * math.Log(0.5)
	if math.Abs(sum-want) > 1e-9 {
		t.Fatalf("sum %v want %v", sum, want)
	}
}

func TestMCTSDeterministic(t *testing.T) {
	prompt := []byte("abc")
	logitsOf := func(buf []byte) []float32 {
		out := []float32{0.1, 0.4, -0.2, 1.2, 0.3}
		out[len(buf)%len(out)] += 0.7
		return out
	}
	cfg := MCTSConfig{Sims: 6, Temp: 0.8}
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
	mc := MCTSConfig{Sims: 2, Temp: 1}
	sample := m.GenerateMCTS([]byte("To be"), 4, mc, rand.New(rand.NewSource(1)))
	again := m.GenerateMCTS([]byte("To be"), 4, mc, rand.New(rand.NewSource(1)))
	if len(sample) != 4 || !bytes.Equal(sample, again) {
		t.Fatalf("sample %q again %q", sample, again)
	}
	if !autogradEnabled {
		t.Fatal("mcts left autograd off")
	}
}
