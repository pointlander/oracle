// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// haltExample is one M#W string labeled by Halts(M, W).
type haltExample struct {
	text string
	halt bool
}

func randomBits(rng *rand.Rand, n int) string {
	b := make([]byte, n)
	for i := range b {
		if rng.Intn(2) == 0 {
			b[i] = '0'
		} else {
			b[i] = '1'
		}
	}
	return string(b)
}

// randomTotalMachine draws a total binary machine whose next states lie in
// 0..states, the same categorical the oracle layer emits. 0 means halt.
func randomTotalMachine(rng *rand.Rand, states int) string {
	mp := make(map[int][2]Transition, states)
	for s := 1; s <= states; s++ {
		var pair [2]Transition
		for sym := range pair {
			move := -1
			if rng.Intn(2) == 1 {
				move = 1
			}
			pair[sym] = Transition{
				WriteSymbol: byte('0' + rng.Intn(2)),
				MoveDir:     move,
				NextState:   rng.Intn(states + 1),
			}
		}
		mp[s] = pair
	}
	return EncodeMachine(mp)
}

// generateHaltExamples builds a balanced set of M#W strings.
// Half are machines the bounded oracle says halt, and half are machines it says do not.
func generateHaltExamples(n, states, tape, maxSteps int, rng *rand.Rand) ([]haltExample, error) {
	if n < 2 {
		return nil, fmt.Errorf("need at least 2 halt examples")
	}
	if states < 1 || tape < 1 || maxSteps < 1 {
		return nil, fmt.Errorf("bad halt example shape")
	}
	sim := &UniversalBoundedSimulator{MaxSteps: maxSteps}
	wantHalt := n / 2
	wantLoop := n - wantHalt
	pos := make([]haltExample, 0, wantHalt)
	neg := make([]haltExample, 0, wantLoop)
	attempts := 0
	limit := n * 200
	for (len(pos) < wantHalt || len(neg) < wantLoop) && attempts < limit {
		attempts++
		m := randomTotalMachine(rng, states)
		w := randomBits(rng, tape)
		h := sim.Halts(m, w)
		ex := haltExample{text: m + "#" + w, halt: h}
		if h && len(pos) < wantHalt {
			pos = append(pos, ex)
		} else if !h && len(neg) < wantLoop {
			neg = append(neg, ex)
		}
	}
	if len(pos) < wantHalt || len(neg) < wantLoop {
		return nil, fmt.Errorf("found %d halting and %d non-halting strings after %d draws", len(pos), len(neg), attempts)
	}
	out := append(pos, neg...)
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out, nil
}

func (ex haltExample) label() int {
	if ex.halt {
		return 1
	}
	return 0
}

func fillHaltBatch(ex []haltExample, ids []int, x []int, labels []int, T int) {
	for b, id := range ids {
		s := ex[id].text
		base := b * T
		for t := 0; t < T; t++ {
			x[base+t] = int(s[t])
		}
		labels[b] = ex[id].label()
	}
}

// TrainHalt fits the halt head on M#W strings labeled by the bounded oracle.
// The first 90% of the generated strings are training examples. The last 10% are validation.
func TrainHalt(cfg TrainConfig) (*TrainResult, error) {
	if cfg.MorePath != "" {
		return nil, fmt.Errorf("-more applies to the language-model task")
	}
	if cfg.Batch < 1 {
		return nil, fmt.Errorf("batch must be positive")
	}
	if cfg.Steps < 0 {
		return nil, fmt.Errorf("steps must be non-negative")
	}
	n := cfg.HaltN
	if n < 2 {
		n = 2048
	}
	if cfg.Model.Ctx == 0 {
		cfg.Model.Ctx = cfg.Ctx
	}
	if cfg.Ctx == 0 {
		cfg.Ctx = cfg.Model.Ctx
	}
	cfg.Model.Classify = true
	if err := cfg.Model.validate(); err != nil {
		return nil, err
	}
	if cfg.LR == 0 {
		cfg.LR = 3e-4
	}
	width := cfg.Model.OracleStates*16 + 1 + cfg.Model.OracleInput
	if width > cfg.Model.Ctx {
		return nil, fmt.Errorf("M#W length %d exceeds context %d", width, cfg.Model.Ctx)
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	examples, err := generateHaltExamples(n, cfg.Model.OracleStates, cfg.Model.OracleInput, cfg.Model.MaxSteps, rng)
	if err != nil {
		return nil, err
	}
	cut := (len(examples) * 9) / 10
	if cut < 1 {
		cut = 1
	}
	if cut >= len(examples) {
		cut = len(examples) - 1
	}
	trainEx := examples[:cut]
	valEx := examples[cut:]
	full := cfg.Steps == 0
	if full {
		cfg.Steps = (len(trainEx) + cfg.Batch - 1) / cfg.Batch
	}
	if cfg.Steps < 1 {
		return nil, fmt.Errorf("steps must be positive")
	}
	if cfg.EvalEvery <= 0 {
		cfg.EvalEvery = max(1, cfg.Steps/10)
	}

	initRng := rand.New(rand.NewSource(cfg.Seed))
	model, err := NewModel(cfg.Model, initRng)
	if err != nil {
		return nil, err
	}
	ps := model.Params()
	opt := newAdamW(ps, cfg.WD)
	T := width

	cfg.logf("halt task")
	cfg.logf("  M#W strings, %d-state machines, %d-bit tapes, %d-step bound, width %d",
		cfg.Model.OracleStates, cfg.Model.OracleInput, cfg.Model.MaxSteps, T)
	cfg.logf("  examples: %d train, %d val, steps %d, batch %d, oracle sublayer %v",
		len(trainEx), len(valEx), cfg.Steps, cfg.Batch, !cfg.Model.NoOracle)

	evalHalt := func(ex []haltExample) (float64, float64, error) {
		was := autogradEnabled
		autogradEnabled = false
		defer func() { autogradEnabled = was }()
		var lossSum float64
		var correct float64
		var seen int
		for i := 0; i < len(ex); i += cfg.Batch {
			j := i + cfg.Batch
			if j > len(ex) {
				j = len(ex)
			}
			ids := make([]int, j-i)
			for k := range ids {
				ids[k] = i + k
			}
			b := len(ids)
			x := make([]int, b*T)
			labels := make([]int, b)
			fillHaltBatch(ex, ids, x, labels, T)
			loss, acc := model.ClassifyLoss(x, labels, b, T)
			v := float64(loss.Data[0])
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return 0, 0, fmt.Errorf("non-finite halt loss")
			}
			lossSum += v * float64(b)
			correct += acc * float64(b)
			seen += b
		}
		return lossSum / float64(seen), correct / float64(seen), nil
	}

	x := make([]int, cfg.Batch*T)
	labels := make([]int, cfg.Batch)
	order := rand.New(rand.NewSource(cfg.Seed + 17))
	ids := make([]int, cfg.Batch)

	take := func(step int) int {
		b := cfg.Batch
		if full {
			b0 := step * cfg.Batch
			b1 := b0 + cfg.Batch
			if b1 > len(trainEx) {
				b1 = len(trainEx)
			}
			b = b1 - b0
			for i := 0; i < b; i++ {
				ids[i] = b0 + i
			}
		} else {
			for i := 0; i < b; i++ {
				ids[i] = order.Intn(len(trainEx))
			}
		}
		fillHaltBatch(trainEx, ids[:b], x[:b*T], labels[:b], T)
		return b
	}

	bestVal := math.Inf(1)
	var bestSnap [][]float32
	bestStep := 0

	logEval := func(step int, trainBCE float64) error {
		bce, acc, err := evalHalt(valEx)
		if err != nil {
			return err
		}
		cfg.logf("step %d/%d  train_bce %.4f  val_bce %.4f  val_acc %.3f",
			step, cfg.Steps, trainBCE, bce, acc)
		if bce < bestVal {
			bestVal = bce
			bestStep = step
			bestSnap = snapshotParams(ps)
		}
		return nil
	}

	b0 := take(0)
	autogradEnabled = false
	loss0, _ := model.ClassifyLoss(x[:b0*T], labels[:b0], b0, T)
	autogradEnabled = true
	first := float64(loss0.Data[0])
	if err := logEval(0, first); err != nil {
		return nil, err
	}

	var runLoss float64
	var runN int
	var lastAvg float64
	for step := 0; step < cfg.Steps; step++ {
		b := take(step)
		autogradEnabled = true
		loss, _ := model.ClassifyLoss(x[:b*T], labels[:b], b, T)
		v := float64(loss.Data[0])
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("non-finite halt loss at step %d", step+1)
		}
		for _, p := range ps {
			if cap(p.Grad) < len(p.Data) {
				p.Grad = make([]float32, len(p.Data))
			} else {
				p.Grad = p.Grad[:len(p.Data)]
				clear(p.Grad)
			}
		}
		Backward(loss)
		gn := clipGradNorm(ps, 1)
		if math.IsNaN(gn) || math.IsInf(gn, 0) {
			return nil, fmt.Errorf("non-finite gradient at step %d", step+1)
		}
		lr := scheduledLR(step, cfg.Warmup, cfg.Steps, cfg.LR)
		opt.Step(ps, lr)
		runLoss += v
		runN++
		lastAvg = runLoss / float64(runN)
		if (step+1)%cfg.EvalEvery == 0 || step+1 == cfg.Steps {
			if err := logEval(step+1, lastAvg); err != nil {
				return nil, err
			}
			runLoss, runN = 0, 0
		}
	}
	if bestSnap == nil {
		return nil, fmt.Errorf("no validation checkpoint")
	}
	restoreParams(ps, bestSnap)
	valBCE, valAcc, err := evalHalt(valEx)
	if err != nil {
		return nil, err
	}
	nParam := 0
	for _, p := range ps {
		nParam += len(p.Data)
	}
	tokens := cfg.Steps * cfg.Batch * T
	if full {
		tokens = len(trainEx) * T
	}
	res := &TrainResult{
		Params:       nParam,
		TotalBytes:   len(examples) * T,
		TrainBytes:   len(trainEx) * T,
		ValBytes:     len(valEx) * T,
		Steps:        cfg.Steps,
		Tokens:       tokens,
		FirstTrainCE: first,
		LastTrainCE:  lastAvg,
		BestSelCE:    bestVal,
		BestSelPPL:   math.Exp(bestVal),
		BestStep:     bestStep,
		ValCE:        valBCE,
		ValPPL:       math.Exp(valBCE),
		ValAcc:       valAcc,
		Task:         "halt",
		Oracle:       !cfg.Model.NoOracle,
	}
	cfg.logf("best selection step %d  val_bce %.4f  val_acc %.3f", bestStep, res.ValCE, res.ValAcc)
	if cfg.OutDir != "" {
		if err := saveCheckpoint(outJoin(cfg.OutDir, "best.gob"), cfg.Model, ps, bestStep, res.ValCE); err != nil {
			return nil, err
		}
		if err := writeResult(outJoin(cfg.OutDir, "result.json"), res); err != nil {
			return nil, err
		}
		cfg.logf("wrote %s", outJoin(cfg.OutDir, "best.gob"))
	}
	cfg.logf("HALT_DONE val_bce=%.4f val_acc=%.3f", res.ValCE, res.ValAcc)
	return res, nil
}

func haltParts(s string) (string, string, bool) {
	m, w, ok := strings.Cut(s, "#")
	if !ok || strings.Contains(w, "#") {
		return "", "", false
	}
	return m, w, true
}
