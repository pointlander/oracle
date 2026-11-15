// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
	"strconv"
	"testing"
)

func TestTapeOfPacksSourceHaltBits(t *testing.T) {
	got := tapeOf([]int{2, 0, 1}, []bool{true, false, true})
	if got != "110" {
		t.Fatalf("tape %q", got)
	}
	if tapeFromMask(0b101, 3) != "101" {
		t.Fatalf("mask tape %q", tapeFromMask(0b101, 3))
	}
}

func TestCopyHeadReadsTheFirstCell(t *testing.T) {
	sim := &UniversalBoundedSimulator{MaxSteps: 8}
	m := copyHeadMachine()
	for mask := 0; mask < 8; mask++ {
		h := sim.Halts(m, tapeFromMask(mask, 3))
		if h != ((mask & 1) == 1) {
			t.Fatalf("mask %03b halt %v", mask, h)
		}
	}
	if sim.Halts(haltNowMachine(1), "00") != true {
		t.Fatal("halt-now")
	}
	if sim.Halts(loopMachine(2), "11") != false {
		t.Fatal("loop")
	}
}

func TestStepFeedsHaltBits(t *testing.T) {
	sim := &UniversalBoundedSimulator{MaxSteps: 8}
	// Node 0 halts exactly when the previous halt bit of node 1 is true.
	halting := &oracleNet{
		machines: []string{copyHeadMachine(), haltNowMachine(1)},
		sources:  [][]int{{1}, {0}},
	}
	first := halting.step(sim, []bool{false, false})
	if first[0] || !first[1] {
		t.Fatalf("halt-now first step %v", first)
	}
	second := halting.step(sim, first)
	if !second[0] || !second[1] {
		t.Fatalf("halt-now second step %v", second)
	}

	looping := &oracleNet{
		machines: []string{copyHeadMachine(), loopMachine(1)},
		sources:  [][]int{{1}, {0}},
	}
	// The loop never halts, so the copy follows whatever bit was already there.
	if got := looping.step(sim, []bool{false, true}); !got[0] || got[1] {
		t.Fatalf("copy of a true loop bit %v", got)
	}
	if got := looping.step(sim, []bool{true, false}); got[0] || got[1] {
		t.Fatalf("copy of a false loop bit %v", got)
	}
}

func TestAllLoopNetworkIsQuietFixedPoint(t *testing.T) {
	sim := &UniversalBoundedSimulator{MaxSteps: 8}
	n := &oracleNet{
		machines: []string{loopMachine(2), loopMachine(2), loopMachine(2), loopMachine(2)},
		sources:  [][]int{{1, 2}, {0, 3}, {3, 3}, {1, 0}},
	}
	zero := []bool{false, false, false, false}
	out := n.step(sim, zero)
	if meanHalt(out) != 0 || hammingBool(out, zero) != 0 {
		t.Fatalf("step %v", out)
	}
	cyc := findCycle(n, sim, zero, 4)
	if !cyc.Closed || !cyc.Fixed || cyc.Length != 1 || cyc.Halt != 0 {
		t.Fatalf("cycle %+v", cyc)
	}
	flipped := n.step(sim, []bool{true, false, false, false})
	if hammingBool(flipped, out) != 0 {
		t.Fatalf("echo %v", flipped)
	}
}

func TestAnalyzeNetsBounds(t *testing.T) {
	cfg := netConfig{
		Nodes: 4, Degree: 2, States: 1, MaxSteps: 8,
		Depth: 3, Trials: 2, Iters: 8, Survey: 16, Seed: 1,
	}
	rep, err := analyzeNets(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if rep.SurveyN != 16 || len(rep.LayerHalt) != cfg.Depth || len(rep.LayerEcho) != cfg.Depth {
		t.Fatalf("shape survey %d layers %d echo %d", rep.SurveyN, len(rep.LayerHalt), len(rep.LayerEcho))
	}
	if len(rep.Survey.CellInfluence) != cfg.Degree || len(rep.Survey.CellSigned) != cfg.Degree {
		t.Fatalf("cells %v signed %v", rep.Survey.CellInfluence, rep.Survey.CellSigned)
	}
	check01 := func(name string, v float64) {
		if math.IsNaN(v) || v < 0 || v > 1 {
			t.Fatalf("%s %v", name, v)
		}
	}
	check01("immediate", rep.Survey.Immediate)
	check01("zero", rep.Survey.ZeroHalt)
	if rep.Survey.ZeroHalt+1e-9 < rep.Survey.Immediate {
		t.Fatalf("zero tape %.3f below immediate %.3f", rep.Survey.ZeroHalt, rep.Survey.Immediate)
	}
	check01("rand", rep.Survey.RandHalt)
	check01("sens", rep.Survey.Sensitive)
	for i, v := range rep.Survey.CellInfluence {
		check01("cell", v)
		if math.IsNaN(rep.Survey.CellSigned[i]) || rep.Survey.CellSigned[i] < -1 || rep.Survey.CellSigned[i] > 1 {
			t.Fatalf("signed %v", rep.Survey.CellSigned[i])
		}
	}
	for i := range rep.LayerHalt {
		check01("halt", rep.LayerHalt[i])
		check01("echo", rep.LayerEcho[i])
	}
	check01("zc", rep.ZeroClosed)
	check01("zf", rep.ZeroFixed)
	check01("rc", rep.RandClosed)
	check01("rf", rep.RandFixed)
	if rep.ZeroFixed > rep.ZeroClosed || rep.RandFixed > rep.RandClosed {
		t.Fatalf("fixed exceeds closed: zero %v/%v rand %v/%v", rep.ZeroFixed, rep.ZeroClosed, rep.RandFixed, rep.RandClosed)
	}
	if rep.ZeroClosed > 0 && (rep.ZeroCycle < 1 || rep.ZeroHalt < 0 || rep.ZeroHalt > 1) {
		t.Fatalf("zero cycle %+v", rep)
	}
	again, err := analyzeNets(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if again.LayerHalt[0] != rep.LayerHalt[0] || again.Survey.Sensitive != rep.Survey.Sensitive {
		t.Fatalf("seed changed the report")
	}
}

func TestNetConfigRejects(t *testing.T) {
	bad := netConfig{Nodes: 1, Degree: 9, States: 1, MaxSteps: 8, Depth: 1, Trials: 1, Iters: 1, Survey: 1}
	if err := bad.validate(); err == nil {
		t.Fatal("expected degree to be rejected")
	}
	bad.Degree = 1
	bad.MaxSteps = 1
	if err := bad.validate(); err == nil {
		t.Fatal("expected a one-step bound to be rejected")
	}
}

func TestZeroTapeCountsImmediateHalts(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	sim := &UniversalBoundedSimulator{MaxSteps: 64}
	for _, states := range []int{1, 2} {
		for range 100 {
			m := randomTotalMachine(rng, states)
			next, err := strconv.ParseInt(m[2:8], 2, 32)
			if err != nil {
				t.Fatal(err)
			}
			if next == 0 && !sim.Halts(m, "0000") {
				t.Fatalf("%d-state machine with a halting 0-transition was reported as running", states)
			}
		}
	}
}

func TestNetHelp(t *testing.T) {
	if err := netFromArgs([]string{"-h"}); err != nil {
		t.Fatal(err)
	}
}
