// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"testing"
)

func TestHaltAtMatchesHalts(t *testing.T) {
	sim := &UniversalBoundedSimulator{MaxSteps: 8}
	cases := []struct {
		m, w string
	}{
		{"", ""},
		{haltNowMachine(1), "00"},
		{loopMachine(1), "11"},
		{copyHeadMachine(), "100"},
		{copyHeadMachine(), "000"},
	}
	for _, c := range cases {
		at := sim.HaltAt(c.m, c.w)
		if sim.Halts(c.m, c.w) != (at >= 0) {
			t.Fatalf("Halts and HaltAt disagree on %q %q (%d)", c.m, c.w, at)
		}
	}
	if sim.HaltAt(haltNowMachine(1), "00") != 1 {
		t.Fatalf("immediate halt at %d", sim.HaltAt(haltNowMachine(1), "00"))
	}
	if sim.HaltAt(loopMachine(1), "11") != -1 {
		t.Fatal("loop reported a halt")
	}
}

func TestEarliestSymbolFollowsTheHighBit(t *testing.T) {
	// Byte 0 wins when the previous byte's high bit is 1.
	// Byte 128 wins when that bit is 0.
	spec := symbolSpec{Alphabet: []byte{0, 128}, States: 1, MaxSteps: 8}
	g := []int{
		0, 1, 1, // byte 0, read 0: keep going
		1, 1, 0, // byte 0, read 1: halt
		0, 1, 0, // byte 128, read 0: halt
		1, 1, 1, // byte 128, read 1: keep going
	}
	machines := decodeSymbols(g, spec)
	sim := &UniversalBoundedSimulator{MaxSteps: spec.MaxSteps}
	if sym, ok := earliestSymbol(sim, machines, spec.Alphabet, 0); !ok || sym != 128 {
		t.Fatalf("after 0 got %d ok %v", sym, ok)
	}
	if sym, ok := earliestSymbol(sim, machines, spec.Alphabet, 128); !ok || sym != 0 {
		t.Fatalf("after 128 got %d ok %v", sym, ok)
	}
	data := bytes.Repeat([]byte{0, 128}, 16)
	acc, _, _, missed := symbolDetail(sim, machines, spec.Alphabet, data)
	if acc != 1 || missed != 0 {
		t.Fatalf("acc %.3f missed %d", acc, missed)
	}
	got := continueSymbols(sim, machines, spec.Alphabet, []byte{0}, 4, 0)
	if !bytes.Equal(got, []byte{128, 0, 128, 0}) {
		t.Fatalf("sample %v", got)
	}
}

func TestSymbolSearchLearnsHighBit(t *testing.T) {
	data := bytes.Repeat([]byte{0, 128}, 30)
	res, err := NetLM(NetLMConfig{
		Data:     data,
		Pop:      24,
		Gens:     20,
		States:   1,
		MaxSteps: 8,
		Bytes:    len(data),
		ValBytes: len(data),
		MutRate:  0.2,
		GenBytes: 0,
		Seed:     1,
		OutDir:   "",
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.FitUnigram > 0.55 {
		t.Fatalf("unigram %.3f", res.FitUnigram)
	}
	if res.FitAcc < 0.9 {
		t.Fatalf("fit %.3f unigram %.3f bigram %.3f", res.FitAcc, res.FitUnigram, res.FitBigram)
	}
}

func TestSymbolOracleRejects(t *testing.T) {
	if err := (symbolSpec{States: 1, MaxSteps: 8}).validate(); err == nil {
		t.Fatal("expected an empty alphabet to be rejected")
	}
	_, err := NetLM(NetLMConfig{Data: []byte{1, 2, 3}, Pop: 1, Bytes: 3})
	if err == nil {
		t.Fatal("expected a population of 1 to be rejected")
	}
}

func TestNetLMHelp(t *testing.T) {
	if err := netLMFromArgs([]string{"-h"}); err != nil {
		t.Fatal(err)
	}
}
