// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"math/rand"
	"os"
	"strings"
	"testing"
)

func TestHuffmanRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	samples := [][]byte{
		[]byte("a"),
		[]byte("ab"),
		[]byte("aaaa"),
		bytes.Repeat([]byte("abracadabra"), 20),
	}
	raw := make([]byte, 4000)
	for i := range raw {
		raw[i] = byte(rng.Intn(50))
	}
	samples = append(samples, raw)
	if f, err := os.ReadFile("100.txt.utf-8"); err == nil {
		if len(f) > 200000 {
			f = f[:200000]
		}
		samples = append(samples, f)
	}
	for _, data := range samples {
		code, err := buildHuffman(freqOf(data))
		if err != nil {
			t.Fatal(err)
		}
		packed, nbits, err := encodeHuffman(code, data)
		if err != nil {
			t.Fatal(err)
		}
		got, err := decodeHuffman(code, packed, nbits, len(data))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, data) {
			t.Fatalf("round trip mismatch len %d vs %d", len(got), len(data))
		}
		var codes []string
		for _, c := range code.code {
			if c != "" {
				codes = append(codes, c)
			}
		}
		for i, a := range codes {
			for j, b := range codes {
				if i != j && strings.HasPrefix(b, a) {
					t.Fatalf("code %q is a prefix of %q", a, b)
				}
			}
		}
	}
}

func TestZeroGenomeAlwaysHalts(t *testing.T) {
	spec := recurSpec{Hidden: 4, States: 1, Tape: 1, MaxSteps: 8}
	g := make([]float64, spec.nGenes())
	sim := &UniversalBoundedSimulator{MaxSteps: spec.MaxSteps}
	table := haltTable(g, spec, sim)
	for h := 0; h < 2; h++ {
		for x := 0; x < 2; x++ {
			if table[h][x] != 1 {
				t.Fatalf("input %d %d -> %d", h, x, table[h][x])
			}
		}
	}
}

func TestHaltTableMatchesStepwise(t *testing.T) {
	spec := recurSpec{Hidden: 4, States: 1, Tape: 2, MaxSteps: 8}
	rng := rand.New(rand.NewSource(3))
	g := make([]float64, spec.nGenes())
	for i := range g {
		g[i] = rng.NormFloat64()
	}
	sim := &UniversalBoundedSimulator{MaxSteps: spec.MaxSteps}
	table := haltTable(g, spec, sim)
	bits := []byte{0, 1, 1, 0, 1, 0, 0, 1, 0}
	h := 0
	for step := 0; step < len(bits)-1; step++ {
		m, w := queryFromLogits(forwardLogits(g, spec, h, int(bits[step])), spec.States, spec.Tape)
		pred := 0
		if sim.Halts(m, w) {
			pred = 1
		}
		if pred != table[h][int(bits[step])] {
			t.Fatalf("step %d pred %d table %d", step, pred, table[h][bits[step]])
		}
		h = pred
	}
}

func TestFeedbackChangesPrediction(t *testing.T) {
	bits := []byte{0, 1, 0, 1, 0, 1, 0}
	ignore := [2][2]int{{0, 1}, {0, 1}}
	useState := [2][2]int{{1, 1}, {0, 0}}
	if sameRoll(bits, ignore, useState) {
		t.Fatal("feedback did not change the rollout")
	}
}

func TestEvolveRaisesAccuracy(t *testing.T) {
	data := bytes.Repeat([]byte("ab"), 100)
	res, err := Evolve(EvolveConfig{
		Data:     data,
		Pop:      24,
		Gens:     20,
		Hidden:   4,
		States:   1,
		Tape:     1,
		MaxSteps: 8,
		MutRate:  0.15,
		MutSigma: 0.5,
		Seed:     1,
		Log:      func(string) {},
	})
	if err != nil {
		t.Fatal(err)
	}
	again, err := Evolve(EvolveConfig{
		Data:     data,
		Pop:      24,
		Gens:     20,
		Hidden:   4,
		States:   1,
		Tape:     1,
		MaxSteps: 8,
		MutRate:  0.15,
		MutSigma: 0.5,
		Seed:     1,
		Log:      func(string) {},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.TrainAcc != again.TrainAcc || len(res.Genome) != len(again.Genome) {
		t.Fatal("same seed did not repeat")
	}
	for i := range res.Genome {
		if res.Genome[i] != again.Genome[i] {
			t.Fatal("genome mismatch")
		}
	}
	t.Logf("train %.3f val %.3f majority %.3f rules %v", res.TrainAcc, res.ValAcc, res.Majority, res.Rules)
	if res.TrainAcc < 0.9 {
		t.Fatalf("train acc %.3f", res.TrainAcc)
	}
	if res.Symbols != 2 {
		t.Fatalf("symbols %d", res.Symbols)
	}
}

func sameRoll(bits []byte, a, b [2][2]int) bool {
	ha, hb := 0, 0
	for t := 0; t < len(bits)-1; t++ {
		pa := a[ha][int(bits[t])]
		pb := b[hb][int(bits[t])]
		if pa != pb {
			return false
		}
		ha, hb = pa, pb
	}
	return true
}
