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

func TestSurvivingSymbolFollowsTheHighBit(t *testing.T) {
	// Byte 255 is still running after a zero byte. Byte 0 is still running
	// after 255. The other machine halts, so the survivor is the next byte.
	spec := symbolSpec{Alphabet: []byte{0, 255}, States: 1, MaxSteps: 8}
	g := []int{
		0, 1, 0, // byte 0, read 0: halt
		1, 1, 1, // byte 0, read 1: walk right across the remaining ones
		0, 1, 1, // byte 255, read 0: walk right across zeros
		1, 1, 0, // byte 255, read 1: halt
	}
	machines := decodeSymbols(g, spec)
	sim := &UniversalBoundedSimulator{MaxSteps: spec.MaxSteps}
	if sym, ok := survivingSymbol(sim, machines, spec.Alphabet, 0); !ok || sym != 255 {
		t.Fatalf("after 0 got %d ok %v", sym, ok)
	}
	if sym, ok := survivingSymbol(sim, machines, spec.Alphabet, 255); !ok || sym != 0 {
		t.Fatalf("after 255 got %d ok %v", sym, ok)
	}
	data := bytes.Repeat([]byte{0, 255}, 16)
	acc, _, _, missed := symbolDetail(sim, machines, spec.Alphabet, data)
	if acc != 1 || missed != 0 {
		t.Fatalf("acc %.3f missed %d", acc, missed)
	}
	got := continueSymbols(sim, machines, spec.Alphabet, []byte{0}, 4, 0)
	if !bytes.Equal(got, []byte{255, 0, 255, 0}) {
		t.Fatalf("sample %v", got)
	}
}

func TestSymbolSearchLearnsHighBit(t *testing.T) {
	data := bytes.Repeat([]byte{0, 255}, 30)
	res, err := NetLM(NetLMConfig{
		Data:     data,
		States:   1,
		MaxSteps: 8,
		Bytes:    len(data),
		ValBytes: len(data),
		GenBytes: 0,
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
	spec := symbolSpec{Alphabet: collectAlphabet(data), States: res.States, MaxSteps: 8}
	machines := decodeSymbols(res.Genome, spec)
	if len(machines) != 2 || machines[0] == machines[1] {
		t.Fatal("symbols need two different machines")
	}
	sim := &UniversalBoundedSimulator{MaxSteps: 8}
	acc, _, _, _ := symbolDetail(sim, machines, spec.Alphabet, data)
	if acc != res.FitAcc {
		t.Fatalf("reported %.3f simulator %.3f", res.FitAcc, acc)
	}
}

func TestSymbolOracleRejects(t *testing.T) {
	if err := (symbolSpec{States: 1, MaxSteps: 8}).validate(); err == nil {
		t.Fatal("expected an empty alphabet to be rejected")
	}
	_, err := NetLM(NetLMConfig{Data: []byte{1, 2, 3}, States: 3, Bytes: 3})
	if err == nil {
		t.Fatal("expected 3 states to be rejected")
	}
}

func TestFastSimMatchesHaltAt(t *testing.T) {
	if machineCount(1) != 64 || machineCount(2) != 20736 {
		t.Fatalf("program counts %d %d", machineCount(1), machineCount(2))
	}
	check := func(states, maxSteps int, ids []int, tapes []byte) {
		t.Helper()
		sim := &UniversalBoundedSimulator{MaxSteps: maxSteps}
		for _, id := range ids {
			m := machineFromGenes(genesFromIndex(id, states), states)
			for _, b := range tapes {
				tape := byteTape(b)
				got := survivesID(states, id, maxSteps, tape)
				want := sim.HaltAt(m, tape) < 0
				if got != want {
					t.Fatalf("states %d steps %d id %d tape %d sim %v haltat %v", states, maxSteps, id, b, got, want)
				}
			}
		}
	}
	tapes := []byte{0, 1, 0x20, 255}
	ids1 := make([]int, machineCount(1))
	for i := range ids1 {
		ids1[i] = i
	}
	check(1, 8, ids1, tapes)
	check(1, 16, ids1, tapes)
	n2 := machineCount(2)
	ids2 := []int{0, 1, n2 - 1}
	for id := 0; id < n2; id += 64 {
		ids2 = append(ids2, id)
	}
	check(2, 16, ids2, tapes)
}

func TestNetLMHelp(t *testing.T) {
	if err := netLMFromArgs([]string{"-h"}); err != nil {
		t.Fatal(err)
	}
}
