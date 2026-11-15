// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
)

// OracleTransformer is a decoder-only Transformer with a halting-oracle
// sublayer in every block. Config.NoOracle drops that sublayer.
//
// Each block pools its residual stream and maps that vector to a total
// binary Turing machine plus a tape. It then calls HaltingOracle. The
// Boolean answer is embedded and added back into the residual stream.
//
// The oracle used here is UniversalBoundedSimulator: the same bounded
// approximation of the classical halting oracle as the rest of this
// program. The simulator is not differentiated. A straight-through
// estimator on the machine bits, and a small loss that predicts the
// oracle bit, give the query a training signal. Bit and answer
// projections start at zero, so step 0 matches an ordinary Transformer
// and the oracle path opens as those weights move.

const (
	defaultOracleStates = 2
	defaultOracleInput  = 8
	defaultOracleSteps  = 64
)

// Config is the architecture, not the optimization schedule.
type Config struct {
	Vocab        int
	Ctx          int
	DModel       int
	Layers       int
	Heads        int
	OracleStates int
	OracleInput  int
	MaxSteps     int
	AuxWeight    float32
	// NoOracle drops the oracle sublayer. The backbone is initialized with the
	// same RNG draws as the oracle model, so a paired run is a control.
	NoOracle bool
	// Classify adds a halt/non-halt head on the last token.
	Classify bool
}

func (c Config) validate() error {
	if c.Vocab < 2 || c.Ctx < 2 || c.DModel < 2 || c.Layers < 1 || c.Heads < 1 {
		return fmt.Errorf("invalid model config %+v", c)
	}
	if c.DModel%c.Heads != 0 {
		return fmt.Errorf("d_model %d is not divisible by heads %d", c.DModel, c.Heads)
	}
	if c.OracleStates < 1 || c.OracleStates > 63 {
		return fmt.Errorf("oracle states must be in 1..63, got %d", c.OracleStates)
	}
	if c.OracleInput < 1 || c.OracleInput > 64 {
		return fmt.Errorf("oracle input length must be in 1..64, got %d", c.OracleInput)
	}
	if c.MaxSteps < 1 {
		return fmt.Errorf("oracle max steps must be positive, got %d", c.MaxSteps)
	}
	return nil
}

func defaultConfig() Config {
	return Config{
		Vocab:        256,
		Ctx:          64,
		DModel:       64,
		Layers:       3,
		Heads:        4,
		OracleStates: defaultOracleStates,
		OracleInput:  defaultOracleInput,
		MaxSteps:     defaultOracleSteps,
		AuxWeight:    0.1,
	}
}

type lnParams struct {
	G, B *Tensor
}

func newLN(d int) *lnParams {
	g := paramTensor([]int{d}, nil, 0, false)
	for i := range g.Data {
		g.Data[i] = 1
	}
	b := paramTensor([]int{d}, nil, 0, false)
	return &lnParams{G: g, B: b}
}

type multiHead struct {
	WQ, WK, WV, WO *Tensor
	BQ, BK, BV, BO *Tensor
	Heads          int
}

func newMultiHead(d, heads int, rng *rand.Rand, outScale float64) *multiHead {
	return &multiHead{
		WQ:    paramTensor([]int{d, d}, rng, 0.02, true),
		WK:    paramTensor([]int{d, d}, rng, 0.02, true),
		WV:    paramTensor([]int{d, d}, rng, 0.02, true),
		WO:    paramTensor([]int{d, d}, rng, outScale, true),
		BQ:    paramTensor([]int{d}, nil, 0, false),
		BK:    paramTensor([]int{d}, nil, 0, false),
		BV:    paramTensor([]int{d}, nil, 0, false),
		BO:    paramTensor([]int{d}, nil, 0, false),
		Heads: heads,
	}
}

func (a *multiHead) Forward(x *Tensor, B, T int) *Tensor {
	D := x.Dims[1]
	H := a.Heads
	Hd := D / H
	q := AddBias(MatMul(x, a.WQ), a.BQ)
	k := AddBias(MatMul(x, a.WK), a.BK)
	v := AddBias(MatMul(x, a.WV), a.BV)
	qh := ArrangeHeads(q, B, T, H, Hd)
	kh := ArrangeHeads(k, B, T, H, Hd)
	vh := ArrangeHeads(v, B, T, H, Hd)
	scores := BatchedMatMul(qh, TransposeLast2(kh))
	scores = Scale(scores, float32(1/math.Sqrt(float64(Hd))))
	weights := SoftmaxLast(CausalMask(scores))
	ctx := BatchedMatMul(weights, vh)
	merged := UnarrangeHeads(ctx, B, T, H, Hd)
	return AddBias(MatMul(merged, a.WO), a.BO)
}

type mlp struct {
	W1, W2 *Tensor
	B1, B2 *Tensor
}

func newMLP(d int, rng *rand.Rand, outScale float64) *mlp {
	ff := 4 * d
	return &mlp{
		W1: paramTensor([]int{d, ff}, rng, 0.02, true),
		B1: paramTensor([]int{ff}, nil, 0, false),
		W2: paramTensor([]int{ff, d}, rng, outScale, true),
		B2: paramTensor([]int{d}, nil, 0, false),
	}
}

func (m *mlp) Forward(x *Tensor) *Tensor {
	h := ReLU(AddBias(MatMul(x, m.W1), m.B1))
	return AddBias(MatMul(h, m.W2), m.B2)
}

// oracleLayer queries a HaltingOracle and writes the answer into the residual stream.
type oracleLayer struct {
	QueryW, QueryB *Tensor
	BitW, BitB     *Tensor
	AuxW, AuxB     *Tensor
	Answer         *Tensor
	Groups         []stGroup
	Feat           int
	States         int
	InputLen       int
	Sim            HaltingOracle
	Calls, Halts   int
	LastM, LastW   string
	LastHalt       bool
}

func buildOracleGroups(states, inputLen int) ([]stGroup, int) {
	groups := make([]stGroup, 0, states*2*3+1)
	off := 0
	nNext := states + 1
	for range states * 2 {
		groups = append(groups, stGroup{Offset: off, N: 1, Cat: false})
		off++
		groups = append(groups, stGroup{Offset: off, N: 1, Cat: false})
		off++
		groups = append(groups, stGroup{Offset: off, N: nNext, Cat: true})
		off += nNext
	}
	groups = append(groups, stGroup{Offset: off, N: inputLen, Cat: false})
	off += inputLen
	return groups, off
}

func newOracleLayer(d, states, inputLen int, rng *rand.Rand, sim HaltingOracle) *oracleLayer {
	groups, feat := buildOracleGroups(states, inputLen)
	// Query is random so the first machines are diverse. The projections that
	// enter the residual start at 0, so the block is initially an ordinary one.
	return &oracleLayer{
		QueryW:   paramTensor([]int{d, feat}, rng, 0.02, true),
		QueryB:   paramTensor([]int{feat}, nil, 0, false),
		BitW:     paramTensor([]int{feat, d}, nil, 0, true),
		BitB:     paramTensor([]int{d}, nil, 0, false),
		AuxW:     paramTensor([]int{feat, 1}, nil, 0, true),
		AuxB:     paramTensor([]int{1}, nil, 0, false),
		Answer:   paramTensor([]int{2, d}, nil, 0, true),
		Groups:   groups,
		Feat:     feat,
		States:   states,
		InputLen: inputLen,
		Sim:      sim,
	}
}

func hardBit(v float32) int {
	if v >= 0.5 {
		return 1
	}
	return 0
}

func oneHotIndex(row []float32) int {
	for i, v := range row {
		if v >= 0.5 {
			return i
		}
	}
	return 0
}

// machineFromHard decodes a straight-through feature row into the simulator's
// binary machine encoding and tape. Next-state choices are restricted to
// {0..states} so every query is a total transition function: 0 is halt.
func machineFromHard(hard []float32, states, inputLen int, groups []stGroup) (string, string) {
	ensure(len(groups) == states*2*3+1, "oracle groups")
	decs := make([]Transition, states*2)
	gi := 0
	for i := 0; i < states*2; i++ {
		w := hardBit(hard[groups[gi].Offset])
		gi++
		mv := hardBit(hard[groups[gi].Offset])
		gi++
		nx := oneHotIndex(hard[groups[gi].Offset : groups[gi].Offset+groups[gi].N])
		gi++
		ws := byte('0')
		if w == 1 {
			ws = '1'
		}
		dir := -1
		if mv == 1 {
			dir = 1
		}
		decs[i] = Transition{WriteSymbol: ws, MoveDir: dir, NextState: nx}
	}
	inGroup := groups[gi]
	ensure(inGroup.N == inputLen, "input group")
	input := make([]byte, inputLen)
	for i := 0; i < inputLen; i++ {
		if hard[inGroup.Offset+i] >= 0.5 {
			input[i] = '1'
		} else {
			input[i] = '0'
		}
	}
	mp := make(map[int][2]Transition, states)
	for s := 1; s <= states; s++ {
		mp[s] = [2]Transition{decs[(s-1)*2], decs[(s-1)*2+1]}
	}
	return EncodeMachine(mp), string(input)
}

// haltBits runs one oracle query per batch row.
// Halts only reads MaxSteps, so the calls share one simulator.
// Counters and the recorded example query are updated after the calls join.
func (o *oracleLayer) haltBits(hard []float32, B int) []int {
	F := o.Feat
	machines := make([]string, B)
	tapes := make([]string, B)
	for b := 0; b < B; b++ {
		machines[b], tapes[b] = machineFromHard(hard[b*F:(b+1)*F], o.States, o.InputLen, o.Groups)
	}
	halt := make([]bool, B)
	parallelN(B, func(i, j int) {
		for b := i; b < j; b++ {
			halt[b] = o.Sim.Halts(machines[b], tapes[b])
		}
	})
	haltIdx := make([]int, B)
	nHalt := 0
	for b := 0; b < B; b++ {
		if halt[b] {
			haltIdx[b] = 1
			nHalt++
		}
	}
	o.Calls += B
	o.Halts += nHalt
	if B > 0 {
		o.LastM = machines[0]
		o.LastW = tapes[0]
		o.LastHalt = halt[0]
	}
	return haltIdx
}

// Forward returns the residual delta [B*T, D] and a scalar BCE against the oracle bit.
func (o *oracleLayer) Forward(h *Tensor, B, T int) (*Tensor, *Tensor) {
	pooled := MeanSeq(h, B, T)
	logits := AddBias(MatMul(pooled, o.QueryW), o.QueryB)
	st := MixedST(logits, o.Groups)
	haltIdx := o.haltBits(st.Data, B)
	bit := AddBias(MatMul(st, o.BitW), o.BitB)
	ans := Embedding(o.Answer, haltIdx)
	delta := RepeatSeq(Add(bit, ans), T)
	auxLogit := AddBias(MatMul(st, o.AuxW), o.AuxB)
	aux := BCEWithLogits(auxLogit, haltIdx)
	return delta, aux
}

type block struct {
	LN1, LN2, LN3 *lnParams
	Attn          *multiHead
	MLP           *mlp
	Ora           *oracleLayer
}

func (b *block) Forward(x *Tensor, B, T int) (*Tensor, *Tensor) {
	h := LayerNorm(x, b.LN1.G, b.LN1.B)
	x = Add(x, b.Attn.Forward(h, B, T))
	h = LayerNorm(x, b.LN2.G, b.LN2.B)
	x = Add(x, b.MLP.Forward(h))
	if b.Ora == nil {
		return x, scalarTensor(0)
	}
	h = LayerNorm(x, b.LN3.G, b.LN3.B)
	delta, aux := b.Ora.Forward(h, B, T)
	return Add(x, delta), aux
}

// OracleTransformer is the trainable language model.
type OracleTransformer struct {
	Cfg    Config
	Tok    *Tensor
	Pos    *Tensor
	Blocks []*block
	Final  *lnParams
	Sim    HaltingOracle
	// ClassW and ClassB map the last position to a halt logit.
	ClassW, ClassB *Tensor
}

func NewModel(cfg Config, rng *rand.Rand) (*OracleTransformer, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if rng == nil {
		return nil, fmt.Errorf("nil rng")
	}
	sim := &UniversalBoundedSimulator{MaxSteps: cfg.MaxSteps}
	m := &OracleTransformer{Cfg: cfg, Sim: sim}
	m.Tok = paramTensor([]int{cfg.Vocab, cfg.DModel}, rng, 0.02, true)
	m.Pos = paramTensor([]int{cfg.Ctx, cfg.DModel}, rng, 0.02, true)
	outScale := 0.02 / math.Sqrt(2*float64(cfg.Layers))
	for layer := 0; layer < cfg.Layers; layer++ {
		// Oracle parameters are always drawn, then dropped for a NoOracle model,
		// so both models share the backbone initialization at the same seed.
		blk := &block{
			LN1:  newLN(cfg.DModel),
			Attn: newMultiHead(cfg.DModel, cfg.Heads, rng, outScale),
			LN2:  newLN(cfg.DModel),
			MLP:  newMLP(cfg.DModel, rng, outScale),
		}
		ln3 := newLN(cfg.DModel)
		ora := newOracleLayer(cfg.DModel, cfg.OracleStates, cfg.OracleInput, rng, sim)
		if !cfg.NoOracle {
			blk.LN3 = ln3
			blk.Ora = ora
		}
		m.Blocks = append(m.Blocks, blk)
	}
	m.Final = newLN(cfg.DModel)
	if cfg.Classify {
		m.ClassW = paramTensor([]int{cfg.DModel, 1}, rng, 0.02, true)
		m.ClassB = paramTensor([]int{1}, nil, 0, false)
	}
	return m, nil
}

func (m *OracleTransformer) Params() []*Tensor {
	ps := []*Tensor{m.Tok, m.Pos}
	for _, b := range m.Blocks {
		ps = append(ps,
			b.LN1.G, b.LN1.B,
			b.Attn.WQ, b.Attn.BQ, b.Attn.WK, b.Attn.BK, b.Attn.WV, b.Attn.BV, b.Attn.WO, b.Attn.BO,
			b.LN2.G, b.LN2.B,
			b.MLP.W1, b.MLP.B1, b.MLP.W2, b.MLP.B2,
		)
		if b.Ora != nil {
			ps = append(ps,
				b.LN3.G, b.LN3.B,
				b.Ora.QueryW, b.Ora.QueryB, b.Ora.BitW, b.Ora.BitB, b.Ora.AuxW, b.Ora.AuxB, b.Ora.Answer,
			)
		}
	}
	ps = append(ps, m.Final.G, m.Final.B)
	if m.ClassW != nil {
		ps = append(ps, m.ClassW, m.ClassB)
	}
	return ps
}

func (m *OracleTransformer) haltRate() float64 {
	c, h := 0, 0
	for _, b := range m.Blocks {
		if b.Ora == nil {
			continue
		}
		c += b.Ora.Calls
		h += b.Ora.Halts
	}
	if c == 0 {
		return 0
	}
	return float64(h) / float64(c)
}

func (m *OracleTransformer) resetOracleStats() {
	for _, b := range m.Blocks {
		if b.Ora == nil {
			continue
		}
		b.Ora.Calls = 0
		b.Ora.Halts = 0
	}
}

// gatherLast returns the hidden state at the last position of each sequence.
func gatherLast(x *Tensor, B, T int) *Tensor {
	D := x.Dims[1]
	ensure(x.Dims[0] == B*T, "gatherLast rows %d", x.Dims[0])
	out := newTensor([]int{B, D}, nil)
	for b := 0; b < B; b++ {
		copy(out.Data[b*D:(b+1)*D], x.Data[((b+1)*T-1)*D:(b+1)*T*D])
	}
	return track(out, []*Tensor{x}, func() {
		if !x.gradAcc {
			return
		}
		for b := 0; b < B; b++ {
			dst := x.Grad[((b+1)*T-1)*D : (b+1)*T*D]
			src := out.Grad[b*D : (b+1)*D]
			for d := 0; d < D; d++ {
				dst[d] += src[d]
			}
		}
	})
}

// features is the final hidden state [B*T, D], the mean oracle BCE, and the
// fraction of oracle queries that halted.
func (m *OracleTransformer) features(toks []int, B, T int) (*Tensor, *Tensor, float64) {
	ensure(T >= 1 && T <= m.Cfg.Ctx, "T %d ctx %d", T, m.Cfg.Ctx)
	ensure(len(toks) == B*T, "tokens %d != %d", len(toks), B*T)
	m.resetOracleStats()
	x := AddPosition(Embedding(m.Tok, toks), m.Pos, B, T)
	var auxSum *Tensor
	for _, blk := range m.Blocks {
		var aux *Tensor
		x, aux = blk.Forward(x, B, T)
		if auxSum == nil {
			auxSum = aux
		} else {
			auxSum = Add(auxSum, aux)
		}
	}
	auxMean := Scale(auxSum, 1/float32(len(m.Blocks)))
	x = LayerNorm(x, m.Final.G, m.Final.B)
	return x, auxMean, m.haltRate()
}

// Logits returns next-byte logits [B*T, V], the mean oracle BCE, and the
// fraction of oracle queries that halted.
func (m *OracleTransformer) Logits(toks []int, B, T int) (*Tensor, *Tensor, float64) {
	h, aux, halt := m.features(toks, B, T)
	return MatMul(h, Transpose(m.Tok)), aux, halt
}

// ClassifyLoss is the mean BCE of the halt head against labels in {0, 1}.
// Accuracy is the fraction of logits whose sign matches the label.
func (m *OracleTransformer) ClassifyLoss(toks []int, labels []int, B, T int) (loss *Tensor, acc float64) {
	ensure(m.ClassW != nil, "model has no halt head")
	ensure(len(labels) == B, "labels %d != %d", len(labels), B)
	h, _, _ := m.features(toks, B, T)
	logit := AddBias(MatMul(gatherLast(h, B, T), m.ClassW), m.ClassB)
	loss = BCEWithLogits(logit, labels)
	correct := 0
	for i, y := range labels {
		pred := logit.Data[i] >= 0
		if pred == (y == 1) {
			correct++
		}
	}
	return loss, float64(correct) / float64(B)
}

// Loss is next-byte cross-entropy plus AuxWeight times the mean oracle BCE.
func (m *OracleTransformer) Loss(toks, tgt []int, B, T int) (total, ce, aux *Tensor, halt float64) {
	logits, aux, halt := m.Logits(toks, B, T)
	ce = CrossEntropy(logits, tgt)
	total = Add(ce, Scale(aux, m.Cfg.AuxWeight))
	return total, ce, aux, halt
}

// nextLogits is the next-byte distribution at the end of buf.
// The returned slice is a copy. An empty buffer is scored as a single newline.
func (m *OracleTransformer) nextLogits(buf []byte) []float32 {
	if len(buf) == 0 {
		buf = []byte{'\n'}
	}
	T := len(buf)
	if T > m.Cfg.Ctx {
		T = m.Cfg.Ctx
	}
	window := buf[len(buf)-T:]
	toks := make([]int, T)
	for j := 0; j < T; j++ {
		id := int(window[j])
		if id >= m.Cfg.Vocab {
			id = id % m.Cfg.Vocab
		}
		toks[j] = id
	}
	logits, _, _ := m.Logits(toks, 1, T)
	last := logits.Data[(T-1)*m.Cfg.Vocab : T*m.Cfg.Vocab]
	out := make([]float32, len(last))
	copy(out, last)
	return out
}

// Generate continues prompt by n bytes.
// Each byte is one draw from softmax(logits / temperature).
// Temperature <= 0 picks the mode instead of sampling.
func (m *OracleTransformer) Generate(prompt []byte, n int, temp float64, rng *rand.Rand) []byte {
	if len(prompt) == 0 {
		prompt = []byte{'\n'}
	}
	buf := append([]byte{}, prompt...)
	was := autogradEnabled
	autogradEnabled = false
	defer func() { autogradEnabled = was }()
	for i := 0; i < n; i++ {
		next := sampleLogits(m.nextLogits(buf), temp, rng)
		buf = append(buf, byte(next))
	}
	return buf[len(prompt):]
}

func sampleLogits(logits []float32, temp float64, rng *rand.Rand) int {
	if temp <= 0 {
		best := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return best
	}
	maxv := logits[0]
	for _, v := range logits[1:] {
		if v > maxv {
			maxv = v
		}
	}
	probs := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		e := math.Exp((float64(v) - float64(maxv)) / temp)
		probs[i] = e
		sum += e
	}
	r := rng.Float64() * sum
	var c float64
	for i, p := range probs {
		c += p
		if r <= c {
			return i
		}
	}
	return len(logits) - 1
}

func l2Data(t *Tensor) float64 {
	var s float64
	for _, v := range t.Data {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s)
}

func gradL2(t *Tensor) float64 {
	if t == nil || t.Grad == nil {
		return 0
	}
	var s float64
	for _, v := range t.Grad {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s)
}
