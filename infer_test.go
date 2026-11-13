// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"math"
	"math/rand"
	"path/filepath"
	"testing"
)

func TestSampleLogitsTemperature(t *testing.T) {
	logits := []float32{0, 2}
	const n = 4000
	count := func(temp float64, seed int64) int {
		rng := rand.New(rand.NewSource(seed))
		k := 0
		for i := 0; i < n; i++ {
			if sampleLogits(logits, temp, rng) == 1 {
				k++
			}
		}
		return k
	}
	// softmax([0, 2]) puts about 0.881 on the second class.
	hot := count(1, 1)
	if hot < 3300 || hot > 3800 {
		t.Fatalf("temp 1 picked class 1 %d/%d times", hot, n)
	}
	// T = 0.3 sharpens that to about 0.999.
	cold := count(0.3, 1)
	if cold <= hot {
		t.Fatalf("lower temperature was less peaked: T=1 %d, T=0.3 %d", hot, cold)
	}
	if sampleLogits(logits, 0, rand.New(rand.NewSource(1))) != 1 {
		t.Fatal("greedy did not pick the mode")
	}

	rng := rand.New(rand.NewSource(2))
	var counts [4]int
	for i := 0; i < n; i++ {
		counts[sampleLogits([]float32{0, 0, 0, 0}, 1, rng)]++
	}
	for i, c := range counts {
		if c < 800 || c > 1200 {
			t.Fatalf("uniform class %d count %d", i, c)
		}
	}
}

func TestCheckpointRoundTripAndSample(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	cfg := defaultConfig()
	cfg.Ctx = 8
	cfg.DModel = 16
	cfg.Layers = 1
	cfg.Heads = 4
	cfg.OracleInput = 4
	cfg.MaxSteps = 8
	model, err := NewModel(cfg, rng)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "model.gob")
	if err := saveCheckpoint(path, model.Cfg, model.Params(), 7, 1.5); err != nil {
		t.Fatal(err)
	}
	loaded, step, val, err := loadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	if step != 7 || math.Abs(val-1.5) > 1e-6 {
		t.Fatalf("meta step %d val %v", step, val)
	}
	got := loaded.Params()
	for i, p := range model.Params() {
		if !bytes.Equal(floatBytes(p.Data), floatBytes(got[i].Data)) {
			t.Fatalf("param %d mismatch", i)
		}
	}
	prompt := []byte("Hello")
	a := loaded.Generate(prompt, 24, 1, rand.New(rand.NewSource(1)))
	again := loaded.Generate(prompt, 24, 1, rand.New(rand.NewSource(1)))
	b := loaded.Generate(prompt, 24, 1, rand.New(rand.NewSource(2)))
	if len(a) != 24 || !bytes.Equal(a, again) {
		t.Fatalf("same seed len %d equal %v", len(a), bytes.Equal(a, again))
	}
	if bytes.Equal(a, b) {
		t.Fatal("different seeds produced the same sample")
	}
}

func floatBytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		u := math.Float32bits(x)
		b[4*i] = byte(u)
		b[4*i+1] = byte(u >> 8)
		b[4*i+2] = byte(u >> 16)
		b[4*i+3] = byte(u >> 24)
	}
	return b
}
