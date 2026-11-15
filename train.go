// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/gob"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"time"
	"unicode/utf8"
)

// split90 cuts a byte string into a training prefix and a validation suffix.
// The cut is floor(9/10 * len). The two slices do not overlap.
func split90(data []byte) (train, val []byte, err error) {
	if len(data) < 2 {
		return nil, nil, fmt.Errorf("need at least 2 bytes, got %d", len(data))
	}
	cut := (len(data) * 9) / 10
	if cut < 1 {
		cut = 1
	}
	if cut >= len(data) {
		cut = len(data) - 1
	}
	return data[:cut], data[cut:], nil
}

// TrainConfig is one optimization run.
type TrainConfig struct {
	Data         []byte
	DataPath     string
	Steps        int
	Batch        int
	Ctx          int
	LR           float64
	WD           float64
	Warmup       int
	EvalEvery    int
	SelBatches   int
	FinalBatches int
	Seed         int64
	OutDir       string
	SampleN      int
	SampleTemp   float64
	Prompt       string
	MorePath     string
	MoreN        int
	HaltN        int // M#W examples for the halt task; 0 uses 2048
	Model        Config
	Log          func(string)
}

// TrainResult is the held-out summary. Validation numbers are measured on the
// suffix slice after restoring the checkpoint with the best selection loss.
type TrainResult struct {
	Params         int     `json:"params"`
	TotalBytes     int     `json:"total_bytes"`
	TrainBytes     int     `json:"train_bytes"`
	ValBytes       int     `json:"val_bytes"`
	Steps          int     `json:"steps"`
	Tokens         int     `json:"tokens"`
	FirstTrainCE   float64 `json:"first_train_ce"`
	LastTrainCE    float64 `json:"last_train_ce"`
	BestSelCE      float64 `json:"best_selection_ce"`
	BestSelPPL     float64 `json:"best_selection_ppl"`
	BestStep       int     `json:"best_step"`
	ValCE          float64 `json:"val_ce"`
	ValPPL         float64 `json:"val_ppl"`
	HaltRate       float64 `json:"halt_rate"`
	OracleAnswerL2 float64 `json:"oracle_answer_l2"`
	OracleBitL2    float64 `json:"oracle_bit_l2"`
	ExampleMachine string  `json:"example_machine"`
	ExampleTape    string  `json:"example_tape"`
	ExampleHalts   bool    `json:"example_halts"`
	Sample         string  `json:"sample"`
	Prompt         string  `json:"prompt"`
	// Task is "lm" or "halt". ValAcc is set for the halt task.
	// Oracle is false when the run is the no-oracle control.
	Task   string  `json:"task,omitempty"`
	ValAcc float64 `json:"val_acc,omitempty"`
	Oracle bool    `json:"oracle"`
}

type adamW struct {
	Beta1, Beta2, Eps, WD float64
	T                     int
	M, V                  [][]float64
}

func newAdamW(ps []*Tensor, wd float64) *adamW {
	a := &adamW{Beta1: 0.9, Beta2: 0.95, Eps: 1e-8, WD: wd}
	a.M = make([][]float64, len(ps))
	a.V = make([][]float64, len(ps))
	for i, p := range ps {
		a.M[i] = make([]float64, len(p.Data))
		a.V[i] = make([]float64, len(p.Data))
	}
	return a
}

func (a *adamW) Step(ps []*Tensor, lr float64) {
	a.T++
	b1c := 1 - math.Pow(a.Beta1, float64(a.T))
	b2c := 1 - math.Pow(a.Beta2, float64(a.T))
	for i, p := range ps {
		if len(p.Grad) != len(p.Data) {
			continue
		}
		for j := range p.Data {
			g := float64(p.Grad[j])
			a.M[i][j] = a.Beta1*a.M[i][j] + (1-a.Beta1)*g
			a.V[i][j] = a.Beta2*a.V[i][j] + (1-a.Beta2)*g*g
			mhat := a.M[i][j] / b1c
			vhat := a.V[i][j] / b2c
			upd := mhat / (math.Sqrt(vhat) + a.Eps)
			if p.decay {
				upd += a.WD * float64(p.Data[j])
			}
			p.Data[j] -= float32(lr * upd)
		}
	}
}

func clipGradNorm(ps []*Tensor, max float64) float64 {
	var ss float64
	for _, p := range ps {
		for _, g := range p.Grad {
			ss += float64(g) * float64(g)
		}
	}
	norm := math.Sqrt(ss)
	if math.IsNaN(norm) || math.IsInf(norm, 0) {
		return norm
	}
	if norm > max && norm > 0 {
		s := float32(max / norm)
		for _, p := range ps {
			for i := range p.Grad {
				p.Grad[i] *= s
			}
		}
	}
	return norm
}

func scheduledLR(step, warmup, total int, base float64) float64 {
	if warmup > 0 && step < warmup {
		return base * float64(step+1) / float64(warmup)
	}
	span := total - warmup
	if span < 1 {
		return base
	}
	p := float64(step-warmup) / float64(span)
	if p < 0 {
		p = 0
	}
	if p > 1 {
		p = 1
	}
	cos := 0.5 * (1 + math.Cos(math.Pi*p))
	return base * (0.1 + 0.9*cos)
}

func fillBatch(data []byte, x, y []int, B, T int, rng *rand.Rand) {
	maxStart := len(data) - T - 1
	for b := 0; b < B; b++ {
		start := 0
		if maxStart > 0 {
			start = rng.Intn(maxStart + 1)
		}
		base := b * T
		for t := 0; t < T; t++ {
			x[base+t] = int(data[start+t])
			y[base+t] = int(data[start+t+1])
		}
	}
}

// windowStarts lists context windows that together predict every byte after
// the first. Windows begin every T bytes, and the last window is aligned to
// the end of the slice when the length is not a multiple of T.
func windowStarts(n, T int) []int {
	if T < 1 || n < T+1 {
		return nil
	}
	last := n - T - 1
	starts := make([]int, 0, last/T+2)
	for s := 0; s <= last; s += T {
		starts = append(starts, s)
	}
	if starts[len(starts)-1] != last {
		starts = append(starts, last)
	}
	return starts
}

func fillWindows(data []byte, x, y []int, starts []int, T int) {
	for b, start := range starts {
		base := b * T
		for t := 0; t < T; t++ {
			x[base+t] = int(data[start+t])
			y[base+t] = int(data[start+t+1])
		}
	}
}

func (cfg TrainConfig) logf(format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	if cfg.Log != nil {
		cfg.Log(msg)
		return
	}
	fmt.Fprintln(os.Stdout, msg)
	_ = os.Stdout.Sync()
}

type checkpoint struct {
	Config Config
	Step   int
	ValCE  float64
	Params [][]float32
}

func snapshotParams(ps []*Tensor) [][]float32 {
	out := make([][]float32, len(ps))
	for i, p := range ps {
		out[i] = append([]float32(nil), p.Data...)
	}
	return out
}

func restoreParams(ps []*Tensor, snap [][]float32) {
	for i, p := range ps {
		copy(p.Data, snap[i])
	}
}

func saveCheckpoint(path string, cfg Config, ps []*Tensor, step int, val float64) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	ck := checkpoint{Config: cfg, Step: step, ValCE: val, Params: snapshotParams(ps)}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(&ck)
}

func outJoin(dir, name string) string {
	return filepath.Join(dir, name)
}

func writeResult(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	buf, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	buf = append(buf, '\n')
	return os.WriteFile(path, buf, 0o644)
}

func evalCE(m *OracleTransformer, data []byte, batch, ctx, batches int, seed int64) (float64, float64, error) {
	if batches < 1 {
		return 0, 0, fmt.Errorf("eval batches")
	}
	rng := rand.New(rand.NewSource(seed))
	x := make([]int, batch*ctx)
	y := make([]int, batch*ctx)
	was := autogradEnabled
	autogradEnabled = false
	defer func() { autogradEnabled = was }()
	var sum, halt float64
	for i := 0; i < batches; i++ {
		fillBatch(data, x, y, batch, ctx, rng)
		_, ce, _, h := m.Loss(x, y, batch, ctx)
		v := float64(ce.Data[0])
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, 0, fmt.Errorf("non-finite eval loss")
		}
		sum += v
		halt += h
	}
	return sum / float64(batches), halt / float64(batches), nil
}

// evalAll is the mean next-byte loss on every window in data.
func evalAll(m *OracleTransformer, data []byte, batch, ctx int) (float64, float64, error) {
	starts := windowStarts(len(data), ctx)
	if len(starts) == 0 {
		return 0, 0, fmt.Errorf("no validation windows")
	}
	if batch < 1 {
		return 0, 0, fmt.Errorf("eval batches")
	}
	x := make([]int, batch*ctx)
	y := make([]int, batch*ctx)
	was := autogradEnabled
	autogradEnabled = false
	defer func() { autogradEnabled = was }()
	var sum float64
	var n int
	var haltSum float64
	var haltN int
	for i := 0; i < len(starts); i += batch {
		j := i + batch
		if j > len(starts) {
			j = len(starts)
		}
		b := j - i
		fillWindows(data, x[:b*ctx], y[:b*ctx], starts[i:j], ctx)
		_, ce, _, h := m.Loss(x[:b*ctx], y[:b*ctx], b, ctx)
		v := float64(ce.Data[0])
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, 0, fmt.Errorf("non-finite eval loss")
		}
		sum += v * float64(b)
		n += b
		haltSum += h
		haltN++
	}
	return sum / float64(n), haltSum / float64(haltN), nil
}

func oracleWeightNorms(m *OracleTransformer) (answer, bit float64) {
	var as, bs float64
	n := 0
	for _, b := range m.Blocks {
		if b.Ora == nil {
			continue
		}
		as += l2Data(b.Ora.Answer)
		bs += l2Data(b.Ora.BitW)
		n++
	}
	if n == 0 {
		return 0, 0
	}
	return as / float64(n), bs / float64(n)
}

// Train fits an OracleTransformer. The file is split once: the first 90% of
// bytes are training windows and the last 10% are validation windows.
func Train(cfg TrainConfig) (*TrainResult, error) {
	if cfg.Steps < 0 {
		return nil, fmt.Errorf("steps must be non-negative")
	}
	if cfg.Batch < 1 {
		return nil, fmt.Errorf("batch must be positive")
	}
	if cfg.Model.Ctx == 0 {
		cfg.Model.Ctx = cfg.Ctx
	}
	if cfg.Ctx == 0 {
		cfg.Ctx = cfg.Model.Ctx
	}
	if cfg.Ctx != cfg.Model.Ctx {
		return nil, fmt.Errorf("ctx %d != model ctx %d", cfg.Ctx, cfg.Model.Ctx)
	}
	if err := cfg.Model.validate(); err != nil {
		return nil, err
	}
	if cfg.LR == 0 {
		cfg.LR = 3e-4
	}
	if cfg.SelBatches <= 0 {
		cfg.SelBatches = 20
	}
	if cfg.FinalBatches <= 0 {
		cfg.FinalBatches = 80
	}
	if cfg.SampleTemp == 0 {
		cfg.SampleTemp = 0.8
	}
	if cfg.Prompt == "" {
		cfg.Prompt = "The quality of mercy "
	}
	data := cfg.Data
	if data == nil {
		path := cfg.DataPath
		if path == "" {
			path = "100.txt.utf-8"
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		data = b
		cfg.DataPath = path
	}
	train, val, err := split90(data)
	if err != nil {
		return nil, err
	}
	baseTrain := len(train)
	var moreNames []string
	var moreBytes int
	if cfg.MorePath != "" {
		n := cfg.MoreN
		if n < 1 {
			n = moreTextCount
		}
		extra, names, err := readFirstTexts(cfg.MorePath, n)
		if err != nil {
			return nil, err
		}
		moreNames = names
		moreBytes = len(extra)
		joined := make([]byte, len(train), len(train)+1+len(extra))
		copy(joined, train)
		if len(extra) > 0 {
			if len(joined) == 0 || joined[len(joined)-1] != '\n' {
				joined = append(joined, '\n')
			}
			joined = append(joined, extra...)
		}
		train = joined
	}
	if len(train) < cfg.Ctx+1 || len(val) < cfg.Ctx+1 {
		return nil, fmt.Errorf("split too small for context %d (train %d, val %d)", cfg.Ctx, len(train), len(val))
	}
	// Steps == 0 is one pass over every training window. Validation then
	// scores every window in the holdout instead of a random subset.
	trainStarts := windowStarts(len(train), cfg.Ctx)
	full := cfg.Steps == 0
	if full {
		if len(trainStarts) == 0 {
			return nil, fmt.Errorf("no training windows")
		}
		cfg.Steps = (len(trainStarts) + cfg.Batch - 1) / cfg.Batch
	}
	if cfg.EvalEvery <= 0 {
		cfg.EvalEvery = max(1, cfg.Steps/10)
	}
	if cfg.Model.Vocab < 256 {
		for _, b := range train {
			if int(b) >= cfg.Model.Vocab {
				return nil, fmt.Errorf("byte %d does not fit vocab %d", b, cfg.Model.Vocab)
			}
		}
	}

	initRng := rand.New(rand.NewSource(cfg.Seed))
	dataRng := rand.New(rand.NewSource(cfg.Seed + 17))
	model, err := NewModel(cfg.Model, initRng)
	if err != nil {
		return nil, err
	}
	ps := model.Params()
	nParam := 0
	for _, p := range ps {
		nParam += len(p.Data)
	}
	opt := newAdamW(ps, cfg.WD)
	oracleOn := model.Blocks[0].Ora != nil

	cfg.logf("oracle transformer")
	cfg.logf("  byte-level decoder, %d layers, d=%d, heads=%d, ctx=%d, vocab=%d",
		cfg.Model.Layers, cfg.Model.DModel, cfg.Model.Heads, cfg.Model.Ctx, cfg.Model.Vocab)
	if oracleOn {
		cfg.logf("  halting oracle: %d-state total binary machines, %d-bit tapes, %d-step bound, %d query features",
			cfg.Model.OracleStates, cfg.Model.OracleInput, cfg.Model.MaxSteps, model.Blocks[0].Ora.Feat)
	} else {
		cfg.logf("  halting oracle: off")
	}
	cfg.logf("  params: %d", nParam)
	if cfg.DataPath != "" {
		cfg.logf("  data: %s", cfg.DataPath)
	}
	cfg.logf("  split: train %d bytes (%.4f%%), val %d bytes (%.4f%%), contiguous suffix holdout",
		baseTrain, 100*float64(baseTrain)/float64(len(data)),
		len(val), 100*float64(len(val))/float64(len(data)))
	cut := baseTrain
	cfg.logf("  boundary around byte %d: %s", cut, boundaryPreview(data, cut))
	if len(moreNames) > 0 {
		cfg.logf("  more: %d bytes from %d texts in %s, training bytes now %d",
			moreBytes, len(moreNames), cfg.MorePath, len(train))
		for _, name := range moreNames {
			cfg.logf("    %s", name)
		}
	}
	if full {
		cfg.logf("  pass: %d training windows, stride %d, %d steps; validation uses every holdout window",
			len(trainStarts), cfg.Ctx, cfg.Steps)
	}

	x := make([]int, cfg.Batch*cfg.Ctx)
	y := make([]int, cfg.Batch*cfg.Ctx)

	tokens := cfg.Steps * cfg.Batch * cfg.Ctx
	if full {
		tokens = len(trainStarts) * cfg.Ctx
	}
	res := &TrainResult{
		Params:     nParam,
		TotalBytes: len(train) + len(val),
		TrainBytes: len(train),
		ValBytes:   len(val),
		Steps:      cfg.Steps,
		Tokens:     tokens,
		Prompt:     cfg.Prompt,
		Task:       "lm",
		Oracle:     oracleOn,
	}

	bestVal := math.Inf(1)
	var bestSnap [][]float32
	bestStep := 0

	logEval := func(step int, trainCE float64) error {
		var ce, halt float64
		var err error
		if full {
			ce, halt, err = evalAll(model, val, cfg.Batch, cfg.Ctx)
		} else {
			ce, halt, err = evalCE(model, val, cfg.Batch, cfg.Ctx, cfg.SelBatches, 1234567)
		}
		if err != nil {
			return err
		}
		cfg.logf("step %d/%d  train_ce %.4f  val_ce %.4f  val_ppl %.2f  halt %.3f",
			step, cfg.Steps, trainCE, ce, math.Exp(ce), halt)
		if ce < bestVal {
			bestVal = ce
			bestStep = step
			bestSnap = snapshotParams(ps)
			res.HaltRate = halt
		}
		return nil
	}

	fillBatch(train, x, y, cfg.Batch, cfg.Ctx, dataRng)
	autogradEnabled = false
	_, ce0, _, _ := model.Loss(x, y, cfg.Batch, cfg.Ctx)
	autogradEnabled = true
	res.FirstTrainCE = float64(ce0.Data[0])
	if err := logEval(0, res.FirstTrainCE); err != nil {
		return nil, err
	}

	var runLoss float64
	var runN int
	var lastAvg float64
	stepTime := time.Now()
	for step := 0; step < cfg.Steps; step++ {
		b := cfg.Batch
		if full {
			b0 := step * cfg.Batch
			b1 := b0 + cfg.Batch
			if b1 > len(trainStarts) {
				b1 = len(trainStarts)
			}
			b = b1 - b0
			fillWindows(train, x[:b*cfg.Ctx], y[:b*cfg.Ctx], trainStarts[b0:b1], cfg.Ctx)
		} else {
			fillBatch(train, x, y, b, cfg.Ctx, dataRng)
		}
		autogradEnabled = true
		total, ce, aux, halt := model.Loss(x[:b*cfg.Ctx], y[:b*cfg.Ctx], b, cfg.Ctx)
		cev := float64(ce.Data[0])
		if math.IsNaN(cev) || math.IsInf(cev, 0) || math.IsNaN(float64(total.Data[0])) {
			return nil, fmt.Errorf("non-finite loss at step %d", step+1)
		}
		for _, p := range ps {
			if cap(p.Grad) < len(p.Data) {
				p.Grad = make([]float32, len(p.Data))
			} else {
				p.Grad = p.Grad[:len(p.Data)]
				clear(p.Grad)
			}
		}
		Backward(total)
		gn := clipGradNorm(ps, 1)
		if math.IsNaN(gn) || math.IsInf(gn, 0) {
			return nil, fmt.Errorf("non-finite gradient at step %d", step+1)
		}
		lr := scheduledLR(step, cfg.Warmup, cfg.Steps, cfg.LR)
		opt.Step(ps, lr)
		runLoss += cev
		runN++
		lastAvg = runLoss / float64(runN)
		if (step+1)%10 == 0 || step == 0 {
			elapsed := time.Since(stepTime)
			cfg.logf("step %d/%d  lr %.3e  train_ce %.4f  aux %.4f  halt %.3f  grad %.3f  %.0fms",
				step+1, cfg.Steps, lr, lastAvg, aux.Data[0], halt, gn, elapsed.Seconds()*1000/float64(max(1, runN)))
		}
		if (step+1)%cfg.EvalEvery == 0 || step+1 == cfg.Steps {
			if err := logEval(step+1, lastAvg); err != nil {
				return nil, err
			}
			runLoss, runN = 0, 0
			stepTime = time.Now()
		}
	}
	res.LastTrainCE = lastAvg
	if bestSnap == nil {
		return nil, fmt.Errorf("no validation checkpoint")
	}
	restoreParams(ps, bestSnap)
	var valCE, valHalt float64
	if full {
		valCE, valHalt, err = evalAll(model, val, cfg.Batch, cfg.Ctx)
	} else {
		valCE, valHalt, err = evalCE(model, val, cfg.Batch, cfg.Ctx, cfg.FinalBatches, 7654321)
	}
	if err != nil {
		return nil, err
	}
	ans, bit := oracleWeightNorms(model)
	res.BestSelCE = bestVal
	res.BestSelPPL = math.Exp(bestVal)
	res.BestStep = bestStep
	res.ValCE = valCE
	res.ValPPL = math.Exp(valCE)
	res.HaltRate = valHalt
	res.OracleAnswerL2 = ans
	res.OracleBitL2 = bit
	if ora := model.Blocks[0].Ora; ora != nil {
		res.ExampleMachine = ora.LastM
		res.ExampleTape = ora.LastW
		res.ExampleHalts = ora.LastHalt
	}

	if cfg.SampleN > 0 {
		genRng := rand.New(rand.NewSource(cfg.Seed + 99))
		sample := model.Generate([]byte(cfg.Prompt), cfg.SampleN, cfg.SampleTemp, genRng)
		if utf8.Valid(sample) {
			res.Sample = string(sample)
		} else {
			res.Sample = strconv.Quote(string(sample))
		}
	}

	if full {
		cfg.logf("best selection step %d  val_ce %.4f  val_ppl %.2f (full validation split)",
			bestStep, res.ValCE, res.ValPPL)
	} else {
		cfg.logf("best selection step %d  val_ce %.4f  val_ppl %.2f (larger holdout, seed 7654321)",
			bestStep, res.ValCE, res.ValPPL)
	}
	cfg.logf("oracle answer L2 %.4f  bit-projection L2 %.4f  halt rate %.3f", ans, bit, valHalt)
	if res.ExampleMachine != "" {
		cfg.logf("example query layer0 machine=%s tape=%s halts=%v", res.ExampleMachine, res.ExampleTape, res.ExampleHalts)
	}
	if cfg.SampleN > 0 {
		cfg.logf("sample prompt %q", cfg.Prompt)
		cfg.logf("sample continuation:\n%s", res.Sample)
	}

	if cfg.OutDir != "" {
		if err := os.MkdirAll(cfg.OutDir, 0o755); err != nil {
			return nil, err
		}
		bestPath := filepath.Join(cfg.OutDir, "best.gob")
		if err := saveCheckpoint(bestPath, cfg.Model, ps, bestStep, res.ValCE); err != nil {
			return nil, err
		}
		jp := filepath.Join(cfg.OutDir, "result.json")
		if err := writeResult(jp, res); err != nil {
			return nil, err
		}
		cfg.logf("wrote %s", bestPath)
		cfg.logf("wrote %s", jp)
	}
	cfg.logf("TRAIN_DONE val_ce=%.4f val_ppl=%.2f", res.ValCE, res.ValPPL)
	return res, nil
}

type compareSide struct {
	Oracle   bool    `json:"oracle"`
	Params   int     `json:"params"`
	ValCE    float64 `json:"val_ce"`
	ValPPL   float64 `json:"val_ppl"`
	HaltRate float64 `json:"halt_rate"`
	BestStep int     `json:"best_step"`
}

type compareReport struct {
	Oracle   compareSide `json:"oracle"`
	NoOracle compareSide `json:"no_oracle"`
}

func sideOf(res *TrainResult) compareSide {
	return compareSide{
		Oracle:   res.Oracle,
		Params:   res.Params,
		ValCE:    res.ValCE,
		ValPPL:   res.ValPPL,
		HaltRate: res.HaltRate,
		BestStep: res.BestStep,
	}
}

// runCompare trains the language model twice from the same seed, data, and
// schedule: once with the oracle sublayer and once without it.
func runCompare(cfg TrainConfig) (onRes, offRes *TrainResult, err error) {
	on := cfg
	on.Model.NoOracle = false
	off := cfg
	off.Model.NoOracle = true
	if cfg.OutDir != "" {
		on.OutDir = filepath.Join(cfg.OutDir, "oracle")
		off.OutDir = filepath.Join(cfg.OutDir, "no-oracle")
	}
	onRes, err = Train(on)
	if err != nil {
		return nil, nil, err
	}
	offRes, err = Train(off)
	if err != nil {
		return onRes, nil, err
	}
	cfg.logf("compare  oracle val_ce %.4f (ppl %.2f)  no-oracle val_ce %.4f (ppl %.2f)",
		onRes.ValCE, onRes.ValPPL, offRes.ValCE, offRes.ValPPL)
	if cfg.OutDir != "" {
		rep := compareReport{Oracle: sideOf(onRes), NoOracle: sideOf(offRes)}
		path := filepath.Join(cfg.OutDir, "compare.json")
		if err := writeResult(path, rep); err != nil {
			return onRes, offRes, err
		}
		cfg.logf("wrote %s", path)
	}
	return onRes, offRes, nil
}

func boundaryPreview(data []byte, cut int) string {
	s := cut - 24
	if s < 0 {
		s = 0
	}
	e := cut + 24
	if e > len(data) {
		e = len(data)
	}
	return strconv.Quote(string(data[s:e]))
}

func trainFromArgs(args []string) error {
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: oracle train [flags]\n\nTrain a byte-level transformer with a halting-oracle sublayer.\nThe text file is split 90%% train / 10%% validation by byte offset.\nThe default run walks every training window once and scores every validation window.\n-oracle=false drops the oracle sublayer. -compare trains both and reports both validation losses.\n-task halt trains a head on M#W strings labeled by the oracle.\n\n")
		fs.PrintDefaults()
	}
	data := fs.String("data", "100.txt.utf-8", "UTF-8 text file")
	more := fs.Bool("more", false, "append the first 10 texts in txt-files.tar.zip to the training split")
	task := fs.String("task", "lm", "lm is next-byte loss; halt labels M#W strings with the oracle")
	oracleOn := fs.Bool("oracle", true, "query the halting oracle in each block")
	compare := fs.Bool("compare", false, "train the language model with the oracle on and off")
	haltN := fs.Int("halt-n", 2048, "M#W examples for -task halt")
	steps := fs.Int("steps", 0, "optimizer steps; 0 is one pass over the training split")
	batch := fs.Int("batch", 16, "batch size")
	ctx := fs.Int("ctx", 64, "context length")
	dmodel := fs.Int("d", 64, "model dimension")
	layers := fs.Int("layers", 3, "transformer layers")
	heads := fs.Int("heads", 4, "attention heads")
	lr := fs.Float64("lr", 3e-4, "peak learning rate")
	wd := fs.Float64("wd", 0.01, "AdamW weight decay on matrices")
	warmup := fs.Int("warmup", 100, "linear warmup steps")
	evalEvery := fs.Int("eval-every", 0, "validation interval in steps; 0 is about 10 times per run")
	oracleSteps := fs.Int("oracle-steps", defaultOracleSteps, "bounded oracle step limit")
	oracleStates := fs.Int("oracle-states", defaultOracleStates, "states in each queried machine")
	oracleInput := fs.Int("oracle-input", defaultOracleInput, "tape length of each query")
	seed := fs.Int64("seed", 42, "seed")
	out := fs.String("out", "checkpoints", "directory for the checkpoint and result.json")
	sampleN := fs.Int("sample", 180, "bytes to generate after training")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	switch *task {
	case "lm", "halt":
	default:
		return fmt.Errorf("unknown task %q", *task)
	}
	if *compare && *task != "lm" {
		return fmt.Errorf("-compare applies to the language-model task")
	}
	if *more && *task != "lm" {
		return fmt.Errorf("-more applies to the language-model task")
	}
	if *task == "halt" && *haltN < 2 {
		return fmt.Errorf("-halt-n must be at least 2")
	}
	cfg := defaultConfig()
	cfg.Ctx = *ctx
	cfg.DModel = *dmodel
	cfg.Layers = *layers
	cfg.Heads = *heads
	cfg.MaxSteps = *oracleSteps
	cfg.OracleStates = *oracleStates
	cfg.OracleInput = *oracleInput
	cfg.NoOracle = !*oracleOn
	var logFile *os.File
	logFn := func(msg string) {
		fmt.Fprintln(os.Stdout, msg)
		_ = os.Stdout.Sync()
		if logFile != nil {
			fmt.Fprintln(logFile, msg)
		}
	}
	if *out != "" {
		if err := os.MkdirAll(*out, 0o755); err != nil {
			return err
		}
		f, err := os.Create(filepath.Join(*out, "train.log"))
		if err != nil {
			return err
		}
		defer f.Close()
		logFile = f
	}
	morePath := ""
	moreN := 0
	if *more {
		morePath = "txt-files.tar.zip"
		moreN = moreTextCount
	}
	tc := TrainConfig{
		DataPath:  *data,
		Steps:     *steps,
		Batch:     *batch,
		Ctx:       *ctx,
		LR:        *lr,
		WD:        *wd,
		Warmup:    *warmup,
		EvalEvery: *evalEvery,
		Seed:      *seed,
		OutDir:    *out,
		SampleN:   *sampleN,
		MorePath:  morePath,
		MoreN:     moreN,
		HaltN:     *haltN,
		Model:     cfg,
		Log:       logFn,
	}
	var err error
	switch {
	case *compare:
		_, _, err = runCompare(tc)
	case *task == "halt":
		_, err = TrainHalt(tc)
	default:
		_, err = Train(tc)
	}
	return err
}
