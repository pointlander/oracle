// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func tinyLM() Config {
	cfg := defaultConfig()
	cfg.Ctx = 8
	cfg.DModel = 16
	cfg.Layers = 2
	cfg.Heads = 4
	cfg.OracleInput = 4
	cfg.MaxSteps = 8
	cfg.Vocab = 32
	return cfg
}

func TestNoOracleMatchesBackbone(t *testing.T) {
	onCfg := tinyLM()
	offCfg := onCfg
	offCfg.NoOracle = true
	on, err := NewModel(onCfg, rand.New(rand.NewSource(7)))
	if err != nil {
		t.Fatal(err)
	}
	off, err := NewModel(offCfg, rand.New(rand.NewSource(7)))
	if err != nil {
		t.Fatal(err)
	}
	if len(off.Params()) >= len(on.Params()) {
		t.Fatalf("param tensors on %d off %d", len(on.Params()), len(off.Params()))
	}
	if !sameData(on.Tok, off.Tok) || !sameData(on.Pos, off.Pos) {
		t.Fatal("token or position embeddings diverged")
	}
	for i := range on.Blocks {
		if off.Blocks[i].Ora != nil || off.Blocks[i].LN3 != nil {
			t.Fatalf("layer %d kept the oracle", i)
		}
		if on.Blocks[i].Ora == nil {
			t.Fatalf("layer %d dropped the oracle", i)
		}
		if !sameData(on.Blocks[i].Attn.WQ, off.Blocks[i].Attn.WQ) {
			t.Fatalf("layer %d WQ diverged", i)
		}
	}

	autogradEnabled = true
	B, T := 2, on.Cfg.Ctx
	toks := make([]int, B*T)
	tgt := make([]int, B*T)
	for i := range toks {
		toks[i] = i % on.Cfg.Vocab
		tgt[i] = (i*3 + 1) % on.Cfg.Vocab
	}
	_, _, _, halt := off.Loss(toks, tgt, B, T)
	if halt != 0 {
		t.Fatalf("no-oracle halt rate %v", halt)
	}
	total, _, _, _ := off.Loss(toks, tgt, B, T)
	Backward(total)
	if gradL2(off.Blocks[0].Attn.WQ) == 0 {
		t.Fatal("attention received no gradient with the oracle off")
	}
	_, _, _, _ = on.Loss(toks, tgt, B, T)
	if on.Blocks[0].Ora.Calls != B || on.Blocks[1].Ora.Calls != B {
		t.Fatalf("oracle calls %d %d", on.Blocks[0].Ora.Calls, on.Blocks[1].Ora.Calls)
	}
}

func TestClassifyHeadReceivesGrad(t *testing.T) {
	autogradEnabled = true
	cfg := tinyLM()
	cfg.Classify = true
	cfg.Vocab = 256
	cfg.Ctx = 24
	cfg.OracleStates = 1
	cfg.OracleInput = 4
	m, err := NewModel(cfg, rand.New(rand.NewSource(9)))
	if err != nil {
		t.Fatal(err)
	}
	ex, err := generateHaltExamples(2, cfg.OracleStates, cfg.OracleInput, cfg.MaxSteps, rand.New(rand.NewSource(9)))
	if err != nil {
		t.Fatal(err)
	}
	T := len(ex[0].text)
	if T > cfg.Ctx {
		t.Fatalf("width %d", T)
	}
	x := make([]int, T)
	for i := 0; i < T; i++ {
		x[i] = int(ex[0].text[i])
	}
	loss, _ := m.ClassifyLoss(x, []int{ex[0].label()}, 1, T)
	if m.Blocks[0].Ora.Calls != 1 {
		t.Fatalf("calls %d", m.Blocks[0].Ora.Calls)
	}
	Backward(loss)
	if gradL2(m.ClassW) == 0 {
		t.Fatal("halt head received no gradient")
	}
	if gradL2(m.Blocks[0].Attn.WQ) == 0 {
		t.Fatal("attention received no gradient")
	}
}

func TestHaltLabelsMatchSimulator(t *testing.T) {
	const n, states, tape, steps = 64, 2, 8, 32
	rng := rand.New(rand.NewSource(1))
	ex, err := generateHaltExamples(n, states, tape, steps, rng)
	if err != nil {
		t.Fatal(err)
	}
	if len(ex) != n {
		t.Fatalf("got %d examples", len(ex))
	}
	sim := &UniversalBoundedSimulator{MaxSteps: steps}
	haltN := 0
	for _, e := range ex {
		m, w, ok := haltParts(e.text)
		if !ok || len(m) != states*16 || len(w) != tape {
			t.Fatalf("bad example %q", e.text)
		}
		if sim.Halts(m, w) != e.halt {
			t.Fatalf("label mismatch for %s", e.text)
		}
		if e.halt {
			haltN++
		}
	}
	if haltN != n/2 {
		t.Fatalf("halting examples %d want %d", haltN, n/2)
	}
}

func TestHaltHeadOverfitsOne(t *testing.T) {
	autogradEnabled = true
	cfg := defaultConfig()
	cfg.Ctx = 24
	cfg.DModel = 32
	cfg.Layers = 1
	cfg.Heads = 4
	cfg.OracleStates = 1
	cfg.OracleInput = 4
	cfg.MaxSteps = 8
	cfg.Classify = true
	m, err := NewModel(cfg, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	ex, err := generateHaltExamples(2, cfg.OracleStates, cfg.OracleInput, cfg.MaxSteps, rand.New(rand.NewSource(2)))
	if err != nil {
		t.Fatal(err)
	}
	T := len(ex[0].text)
	x := make([]int, T)
	labels := []int{ex[0].label()}
	for i := 0; i < T; i++ {
		x[i] = int(ex[0].text[i])
	}
	ps := m.Params()
	opt := newAdamW(ps, 0)
	var first, last float64
	for step := 0; step < 40; step++ {
		loss, _ := m.ClassifyLoss(x, labels, 1, T)
		last = float64(loss.Data[0])
		if step == 0 {
			first = last
		}
		Backward(loss)
		opt.Step(ps, 0.05)
	}
	t.Logf("one-example bce %.4f -> %.4f label %d", first, last, labels[0])
	if last >= first*0.5 {
		t.Fatalf("did not overfit: %.4f -> %.4f", first, last)
	}
}

func TestTrainHaltFinite(t *testing.T) {
	cfg := defaultConfig()
	cfg.Ctx = 24
	cfg.DModel = 32
	cfg.Layers = 1
	cfg.Heads = 4
	cfg.OracleStates = 1
	cfg.OracleInput = 4
	cfg.MaxSteps = 8
	res, err := TrainHalt(TrainConfig{
		Steps:     0,
		Batch:     8,
		Ctx:       cfg.Ctx,
		LR:        1e-3,
		Warmup:    0,
		EvalEvery: 100,
		Seed:      1,
		HaltN:     32,
		Model:     cfg,
		Log:       func(string) {},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Task != "halt" || !res.Oracle {
		t.Fatalf("task %q oracle %v", res.Task, res.Oracle)
	}
	if res.Steps != 4 {
		t.Fatalf("steps %d want one epoch of 28 examples", res.Steps)
	}
	if res.Tokens != res.TrainBytes || res.TrainBytes+res.ValBytes != res.TotalBytes {
		t.Fatalf("bytes train %d val %d total %d tokens %d", res.TrainBytes, res.ValBytes, res.TotalBytes, res.Tokens)
	}
	if math.IsNaN(res.ValCE) || math.IsInf(res.ValCE, 0) || math.IsNaN(res.FirstTrainCE) || math.IsNaN(res.LastTrainCE) {
		t.Fatalf("non-finite %+v", res)
	}
	if res.ValAcc < 0 || res.ValAcc > 1 {
		t.Fatalf("acc %v", res.ValAcc)
	}
}

func TestCompareReportsBoth(t *testing.T) {
	raw := bytesRepeat("abcdefg", 20)
	cfg := tinyLM()
	cfg.Layers = 1
	cfg.Vocab = 256
	dir := t.TempDir()
	on, off, err := runCompare(TrainConfig{
		Data:         raw,
		Steps:        2,
		Batch:        4,
		Ctx:          cfg.Ctx,
		LR:           1e-3,
		Warmup:       0,
		EvalEvery:    2,
		SelBatches:   1,
		FinalBatches: 1,
		Seed:         3,
		OutDir:       dir,
		SampleN:      0,
		Model:        cfg,
		Log:          func(string) {},
	})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(on.ValCE) || math.IsInf(on.ValCE, 0) || math.IsNaN(off.ValCE) || math.IsInf(off.ValCE, 0) {
		t.Fatalf("val on %v off %v", on.ValCE, off.ValCE)
	}
	if !on.Oracle || off.Oracle {
		t.Fatalf("oracle flags on %v off %v", on.Oracle, off.Oracle)
	}
	if on.Params <= off.Params {
		t.Fatalf("params on %d off %d", on.Params, off.Params)
	}
	buf, err := os.ReadFile(filepath.Join(dir, "compare.json"))
	if err != nil {
		t.Fatal(err)
	}
	var rep compareReport
	if err := json.Unmarshal(buf, &rep); err != nil {
		t.Fatal(err)
	}
	if !rep.Oracle.Oracle || rep.NoOracle.Oracle {
		t.Fatalf("report %+v", rep)
	}
	if rep.Oracle.ValCE != on.ValCE || rep.NoOracle.ValCE != off.ValCE {
		t.Fatalf("report ce %+v", rep)
	}
}

func TestHaltFlagsReject(t *testing.T) {
	if err := trainFromArgs([]string{"-task", "halt", "-more", "-out", ""}); err == nil {
		t.Fatal("expected -more to be rejected")
	}
	if err := trainFromArgs([]string{"-task", "halt", "-compare", "-out", ""}); err == nil {
		t.Fatal("expected -compare to be rejected")
	}
	if err := trainFromArgs([]string{"-task", "other", "-out", ""}); err == nil {
		t.Fatal("expected unknown task to be rejected")
	}
}

func TestOldCheckpointKeepsOracle(t *testing.T) {
	if _, err := os.Stat("checkpoints/best.gob"); err != nil {
		t.Skip(err)
	}
	m, _, _, err := loadCheckpoint("checkpoints/best.gob")
	if err != nil {
		t.Fatal(err)
	}
	if m.Cfg.NoOracle || m.Cfg.Classify {
		t.Fatal("old checkpoint set a new config field")
	}
	if m.Blocks[0].Ora == nil {
		t.Fatal("old checkpoint dropped the oracle")
	}
}

func sameData(a, b *Tensor) bool {
	if len(a.Data) != len(b.Data) {
		return false
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			return false
		}
	}
	return true
}

func bytesRepeat(s string, n int) []byte {
	out := make([]byte, 0, len(s)*n)
	for i := 0; i < n; i++ {
		out = append(out, s...)
	}
	return out
}
