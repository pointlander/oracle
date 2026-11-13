// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"math"
	"math/rand"
	"os"
	"testing"
)

func near(got, want float32) bool {
	diff := math.Abs(float64(got - want))
	base := math.Abs(float64(want))
	if base < 1e-3 {
		return diff < 2e-3
	}
	return diff/base < 0.08 || diff < 2e-3
}

func finiteDiff(f func() float32, x *Tensor, idx int) float32 {
	orig := x.Data[idx]
	const eps float32 = 1e-3
	x.Data[idx] = orig + eps
	lp := float64(f())
	x.Data[idx] = orig - eps
	lm := float64(f())
	x.Data[idx] = orig
	return float32((lp - lm) / (2 * float64(eps)))
}

func TestMatMulGrad(t *testing.T) {
	autogradEnabled = true
	a := newTensor([]int{2, 2}, []float32{1, 2, 3, 4})
	b := newTensor([]int{2, 2}, []float32{5, 6, 7, 8})
	a.gradAcc, b.gradAcc = true, true
	c := MatMul(a, b)
	wantC := []float32{19, 22, 43, 50}
	for i, v := range wantC {
		if c.Data[i] != v {
			t.Fatalf("C[%d]=%v want %v", i, c.Data[i], v)
		}
	}
	Backward(Sum(c))
	wantA := []float32{11, 15, 11, 15}
	wantB := []float32{4, 4, 6, 6}
	for i := range wantA {
		if a.Grad[i] != wantA[i] || b.Grad[i] != wantB[i] {
			t.Fatalf("grads A %v B %v", a.Grad, b.Grad)
		}
	}
}

func TestCrossEntropyValueAndGrad(t *testing.T) {
	autogradEnabled = true
	logits := newTensor([]int{1, 4}, []float32{0, 0, 0, 0})
	logits.gradAcc = true
	loss := CrossEntropy(logits, []int{0})
	want := float32(math.Log(4))
	if math.Abs(float64(loss.Data[0]-want)) > 1e-5 {
		t.Fatalf("loss %v want %v", loss.Data[0], want)
	}
	Backward(loss)
	// softmax is uniform 0.25; target grad = 0.25-1, others 0.25
	if !near(logits.Grad[0], -0.75) {
		t.Fatalf("target grad %v", logits.Grad[0])
	}
	for j := 1; j < 4; j++ {
		if !near(logits.Grad[j], 0.25) {
			t.Fatalf("grad[%d]=%v", j, logits.Grad[j])
		}
	}
}

func TestBCE(t *testing.T) {
	autogradEnabled = true
	z := newTensor([]int{1, 1}, []float32{0})
	z.gradAcc = true
	loss := BCEWithLogits(z, []int{1})
	if math.Abs(float64(loss.Data[0])-math.Log(2)) > 1e-5 {
		t.Fatalf("loss %v", loss.Data[0])
	}
	Backward(loss)
	if !near(z.Grad[0], -0.5) {
		t.Fatalf("grad %v", z.Grad[0])
	}
}

func TestLayerNormGrad(t *testing.T) {
	autogradEnabled = true
	rng := rand.New(rand.NewSource(3))
	x := paramTensor([]int{4, 5}, rng, 0.8, false)
	g := paramTensor([]int{5}, nil, 0, false)
	b := paramTensor([]int{5}, nil, 0, false)
	for i := range g.Data {
		g.Data[i] = 1
	}
	w := newTensor([]int{5, 1}, []float32{0.2, -0.4, 0.7, 0.1, -0.3})
	forward := func() float32 {
		y := LayerNorm(x, g, b)
		return Sum(MatMul(y, w)).Data[0]
	}
	loss := Sum(MatMul(LayerNorm(x, g, b), w))
	Backward(loss)
	for _, idx := range []int{0, 6, len(x.Data) - 1} {
		got := x.Grad[idx]
		want := finiteDiff(forward, x, idx)
		if !near(got, want) {
			t.Fatalf("x[%d] analytic %g numeric %g", idx, got, want)
		}
	}
	got := g.Grad[2]
	want := finiteDiff(forward, g, 2)
	if !near(got, want) {
		t.Fatalf("gamma analytic %g numeric %g", got, want)
	}
}

func TestAttentionGrad(t *testing.T) {
	autogradEnabled = true
	rng := rand.New(rand.NewSource(4))
	const B, T, D, H = 2, 4, 16, 4
	attn := newMultiHead(D, H, rng, 0.05)
	x := paramTensor([]int{B * T, D}, rng, 0.4, false)
	forward := func() float32 {
		return Sum(attn.Forward(x, B, T)).Data[0]
	}
	Backward(Sum(attn.Forward(x, B, T)))
	for _, idx := range []int{0, 5, len(x.Data) - 1} {
		got := x.Grad[idx]
		want := finiteDiff(forward, x, idx)
		if !near(got, want) {
			t.Fatalf("x[%d] analytic %g numeric %g", idx, got, want)
		}
	}
	got := attn.WQ.Grad[3]
	want := finiteDiff(forward, attn.WQ, 3)
	if !near(got, want) {
		t.Fatalf("WQ analytic %g numeric %g", got, want)
	}
}

func TestCausalMask(t *testing.T) {
	autogradEnabled = true
	x := newTensor([]int{1, 2, 2}, []float32{1, 2, 3, 4})
	x.gradAcc = true
	y := CausalMask(x)
	if y.Data[1] != causalNeg || y.Data[0] != 1 || y.Data[2] != 3 || y.Data[3] != 4 {
		t.Fatalf("masked %v", y.Data)
	}
	Backward(Sum(y))
	if x.Grad[0] != 1 || x.Grad[1] != 0 || x.Grad[2] != 1 || x.Grad[3] != 1 {
		t.Fatalf("mask grad %v", x.Grad)
	}
}

func TestArrangeRoundTrip(t *testing.T) {
	autogradEnabled = true
	const B, T, H, Hd = 2, 3, 2, 2
	in := newTensor([]int{B * T, H * Hd}, nil)
	for i := range in.Data {
		in.Data[i] = float32(i + 1)
	}
	in.gradAcc = true
	arr := ArrangeHeads(in, B, T, H, Hd)
	back := UnarrangeHeads(arr, B, T, H, Hd)
	for i := range in.Data {
		if back.Data[i] != in.Data[i] {
			t.Fatalf("roundtrip %d: %v != %v", i, back.Data[i], in.Data[i])
		}
	}
	// One head-major spot: token tt=1, head h=0, d=1 lives at arranged (h=0,tt=1,d=1).
	b, tt, h, d := 1, 1, 0, 1
	src := ((b*T+tt)*H+h)*Hd + d
	dst := ((b*H+h)*T+tt)*Hd + d
	if arr.Data[dst] != in.Data[src] {
		t.Fatalf("arrange %v != %v", arr.Data[dst], in.Data[src])
	}
	Backward(Sum(back))
	for i := range in.Grad {
		if in.Grad[i] != 1 {
			t.Fatalf("grad %d = %v", i, in.Grad[i])
		}
	}
}

func TestMeanAndRepeat(t *testing.T) {
	autogradEnabled = true
	x := newTensor([]int{4, 1}, []float32{1, 3, 5, 7})
	x.gradAcc = true
	m := MeanSeq(x, 2, 2)
	if m.Data[0] != 2 || m.Data[1] != 6 {
		t.Fatalf("mean %v", m.Data)
	}
	Backward(Sum(m))
	for i, g := range x.Grad {
		if g != 0.5 {
			t.Fatalf("mean grad[%d]=%v", i, g)
		}
	}
	v := newTensor([]int{2, 1}, []float32{10, 20})
	v.gradAcc = true
	r := RepeatSeq(v, 3)
	want := []float32{10, 10, 10, 20, 20, 20}
	for i := range want {
		if r.Data[i] != want[i] {
			t.Fatalf("repeat %v", r.Data)
		}
	}
	Backward(Sum(r))
	if v.Grad[0] != 3 || v.Grad[1] != 3 {
		t.Fatalf("repeat grad %v", v.Grad)
	}
}

func TestMixedST(t *testing.T) {
	autogradEnabled = true
	z := paramTensor([]int{1, 4}, nil, 0, false)
	z.Data[0], z.Data[1], z.Data[2], z.Data[3] = 2, 0, 0, 3
	groups := []stGroup{{Offset: 0, N: 1, Cat: false}, {Offset: 1, N: 3, Cat: true}}
	out := MixedST(z, groups)
	if out.Data[0] != 1 || out.Data[1] != 0 || out.Data[2] != 0 || out.Data[3] != 1 {
		t.Fatalf("hard %v", out.Data)
	}
	c := newTensor([]int{4, 1}, []float32{1.5, -0.2, 0.4, 0.7})
	Backward(Sum(MatMul(out, c)))
	p := sigmoid(2)
	wantBit := 1.5 * p * (1 - p)
	if !near(z.Grad[0], wantBit) {
		t.Fatalf("bit grad %v want %v", z.Grad[0], wantBit)
	}
	// Independent softmax backward on logits [0,0,3] with upstream [-0.2,0.4,0.7].
	zz := []float32{0, 0, 3}
	maxv := float32(3)
	soft := make([]float32, 3)
	var sum float32
	for i, v := range zz {
		soft[i] = float32(math.Exp(float64(v - maxv)))
		sum += soft[i]
	}
	up := []float32{-0.2, 0.4, 0.7}
	var dot float32
	for i := range soft {
		soft[i] /= sum
		dot += soft[i] * up[i]
	}
	for i := range soft {
		want := soft[i] * (up[i] - dot)
		if !near(z.Grad[1+i], want) {
			t.Fatalf("cat grad[%d]=%v want %v", i, z.Grad[1+i], want)
		}
	}
}

func TestBoundedOracleHaltAndLoop(t *testing.T) {
	sim := &UniversalBoundedSimulator{MaxSteps: 32}
	halt := EncodeMachine(map[int][2]Transition{
		1: {
			{WriteSymbol: '0', MoveDir: 1, NextState: 0},
			{WriteSymbol: '0', MoveDir: 1, NextState: 0},
		},
		2: {
			{WriteSymbol: '0', MoveDir: 1, NextState: 1},
			{WriteSymbol: '0', MoveDir: 1, NextState: 1},
		},
	})
	if !sim.Halts(halt, "0000") {
		t.Fatal("expected halt")
	}
	loop := EncodeMachine(map[int][2]Transition{
		1: {
			{WriteSymbol: '0', MoveDir: 1, NextState: 1},
			{WriteSymbol: '0', MoveDir: 1, NextState: 1},
		},
		2: {
			{WriteSymbol: '0', MoveDir: 1, NextState: 1},
			{WriteSymbol: '0', MoveDir: 1, NextState: 1},
		},
	})
	if sim.Halts(loop, "0000") {
		t.Fatal("expected non-halt within the step bound")
	}
}

func TestMachineEncodingMatchesFormat(t *testing.T) {
	states, inLen := 2, 8
	groups, feat := buildOracleGroups(states, inLen)
	if feat != 28 {
		t.Fatalf("feat %d", feat)
	}
	logits := make([]float32, feat)
	// state1/sym0: write 1, move left, next 0
	logits[0] = 1
	logits[1] = -1
	logits[2], logits[3], logits[4] = 5, 0, 0
	// state1/sym1: write 0, move right, next 1
	logits[5] = -2
	logits[6] = 2
	logits[7], logits[8], logits[9] = 0, 4, 0
	// state2/sym0: write 0, move right, next 1
	logits[10] = -1
	logits[11] = 1
	logits[12], logits[13], logits[14] = 0, 3, 1
	// state2/sym1: write 1, move left, next 2
	logits[15] = 1
	logits[16] = -3
	logits[17], logits[18], logits[19] = 0, 0, 6
	// tape 10100000
	logits[20] = 1
	logits[21] = -1
	logits[22] = 0.5
	for i := 23; i < 28; i++ {
		logits[i] = -1
	}
	hard := make([]float32, feat)
	writeHard(hard, logits, groups)
	gotM, gotW := machineFromHard(hard, states, inLen, groups)
	want := EncodeMachine(map[int][2]Transition{
		1: {
			{WriteSymbol: '1', MoveDir: -1, NextState: 0},
			{WriteSymbol: '0', MoveDir: 1, NextState: 1},
		},
		2: {
			{WriteSymbol: '0', MoveDir: 1, NextState: 1},
			{WriteSymbol: '1', MoveDir: -1, NextState: 2},
		},
	})
	if gotM != want || gotW != "10100000" {
		t.Fatalf("machine %s tape %s\nwant %s", gotM, gotW, want)
	}
	// sym0 of state 1 halts, but the tape starts with 1, so the sym1 rule runs.
	// That rule returns to state 1 and writes 0 while moving right, then the
	// next cell is 0 and state 1 halts. The bounded oracle must agree.
	sim := &UniversalBoundedSimulator{MaxSteps: 32}
	if !sim.Halts(gotM, gotW) {
		t.Fatal("expected this query to halt")
	}
}

func TestOraclePathReceivesGrad(t *testing.T) {
	autogradEnabled = true
	rng := rand.New(rand.NewSource(5))
	cfg := defaultConfig()
	cfg.Ctx = 8
	cfg.DModel = 16
	cfg.Layers = 1
	cfg.Heads = 4
	cfg.OracleInput = 4
	cfg.MaxSteps = 16
	cfg.Vocab = 32
	m, err := NewModel(cfg, rng)
	if err != nil {
		t.Fatal(err)
	}
	B, T := 2, 8
	toks := make([]int, B*T)
	tgt := make([]int, B*T)
	for i := range toks {
		toks[i] = i % cfg.Vocab
		tgt[i] = (i*3 + 1) % cfg.Vocab
	}
	total, _, _, _ := m.Loss(toks, tgt, B, T)
	Backward(total)
	if gradL2(m.Blocks[0].Attn.WQ) == 0 {
		t.Fatal("attention received no gradient")
	}
	if gradL2(m.Blocks[0].Ora.BitW) == 0 {
		t.Fatal("oracle bit projection received no gradient")
	}
	if gradL2(m.Blocks[0].Ora.Answer) == 0 {
		t.Fatal("oracle answer embedding received no gradient")
	}
	if m.Blocks[0].Ora.Calls != B {
		t.Fatalf("oracle calls %d", m.Blocks[0].Ora.Calls)
	}
}

func TestSplit90(t *testing.T) {
	data := make([]byte, 1000)
	for i := range data {
		data[i] = byte(i)
	}
	tr, va, err := split90(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(tr) != 900 || len(va) != 100 {
		t.Fatalf("lens %d %d", len(tr), len(va))
	}
	if tr[len(tr)-1] != data[899] || va[0] != data[900] {
		t.Fatal("boundary bytes")
	}
}

func TestRealFileSplit(t *testing.T) {
	data, err := os.ReadFile("100.txt.utf-8")
	if err != nil {
		t.Skip(err)
	}
	tr, va, err := split90(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(tr)+len(va) != len(data) {
		t.Fatal("coverage")
	}
	ratio := float64(len(tr)) / float64(len(data))
	if ratio < 0.899 || ratio > 0.901 {
		t.Fatalf("train ratio %v (%d/%d)", ratio, len(tr), len(data))
	}
	if !bytes.Equal(data[:len(tr)], tr) || !bytes.Equal(data[len(tr):], va) {
		t.Fatal("slices are not a partition of the file")
	}
}

func TestGenerateAndLearn(t *testing.T) {
	autogradEnabled = true
	rng := rand.New(rand.NewSource(6))
	cfg := defaultConfig()
	cfg.Ctx = 16
	cfg.DModel = 32
	cfg.Layers = 1
	cfg.Heads = 4
	cfg.OracleInput = 4
	cfg.MaxSteps = 16
	m, err := NewModel(cfg, rng)
	if err != nil {
		t.Fatal(err)
	}
	sample := m.Generate([]byte("abcdefghijklmnopEXTRA"), 5, 0, rng)
	if len(sample) != 5 {
		t.Fatalf("sample len %d", len(sample))
	}
	if !autogradEnabled {
		t.Fatal("generate left autograd off")
	}

	raw := bytes.Repeat([]byte("To be, or not to be, that is the question. "), 40)
	res, err := Train(TrainConfig{
		Data:         raw,
		Steps:        40,
		Batch:        8,
		Ctx:          cfg.Ctx,
		LR:           2e-2,
		WD:           0,
		Warmup:       0,
		EvalEvery:    20,
		SelBatches:   4,
		FinalBatches: 4,
		Seed:         6,
		SampleN:      0,
		Model:        cfg,
		Log:          func(string) {},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.TrainBytes*10 < res.TotalBytes*8 || res.ValBytes == 0 {
		t.Fatalf("split train %d val %d total %d", res.TrainBytes, res.ValBytes, res.TotalBytes)
	}
	if !(res.LastTrainCE < res.FirstTrainCE-0.5) {
		t.Fatalf("did not learn: first %.3f last %.3f", res.FirstTrainCE, res.LastTrainCE)
	}
	if math.IsNaN(res.ValCE) || res.ValPPL <= 1 {
		t.Fatalf("val %+v", res)
	}
}

func refNN(M, N, K int, A, B, C []float32) {
	for i := 0; i < M; i++ {
		for p := 0; p < K; p++ {
			av := A[i*K+p]
			for j := 0; j < N; j++ {
				C[i*N+j] += av * B[p*N+j]
			}
		}
	}
}

func refAD(M, N, K int, dC, B, dA []float32) {
	for i := 0; i < M; i++ {
		for p := 0; p < K; p++ {
			var s float32
			for j := 0; j < N; j++ {
				s += dC[i*N+j] * B[p*N+j]
			}
			dA[i*K+p] += s
		}
	}
}

func refBD(M, N, K int, A, dC, dB []float32) {
	for k := 0; k < K; k++ {
		for m := 0; m < M; m++ {
			av := A[m*K+k]
			for j := 0; j < N; j++ {
				dB[k*N+j] += av * dC[m*N+j]
			}
		}
	}
}

func gemmClose(got, want float32, exact bool) bool {
	if exact || got == want {
		return got == want
	}
	diff := math.Abs(float64(got - want))
	scale := math.Abs(float64(want))
	if scale < 1 {
		scale = 1
	}
	return diff <= 1e-4*scale
}

func TestGemmMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	shapes := [][3]int{
		{1, 1, 1},
		{2, 2, 2},
		{3, 3, 5},
		{4, 4, 4},
		{5, 7, 3},
		{8, 32, 1},
		{8, 32, 5},
		{6, 6, 6},
		{17, 13, 11},
		{32, 64, 15},
		{65, 50, 36},
	}
	fill := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.Float64()*2 - 1)
		}
		return s
	}
	for _, sh := range shapes {
		M, N, K := sh[0], sh[1], sh[2]
		exact := N < 4
		A := fill(M * K)
		B := fill(K * N)
		C0 := fill(M * N)
		dC := fill(M * N)
		dA0 := fill(M * K)
		dB0 := fill(K * N)

		wantC := append([]float32(nil), C0...)
		gotC := append([]float32(nil), C0...)
		refNN(M, N, K, A, B, wantC)
		gemmNN(M, N, K, A, B, gotC)

		wantA := append([]float32(nil), dA0...)
		gotA := append([]float32(nil), dA0...)
		refAD(M, N, K, dC, B, wantA)
		gemmAD(M, N, K, dC, B, gotA)

		wantB := append([]float32(nil), dB0...)
		gotB := append([]float32(nil), dB0...)
		refBD(M, N, K, A, dC, wantB)
		gemmBD(M, N, K, A, dC, gotB)

		check := func(name string, got, want []float32) {
			t.Helper()
			for i := range want {
				if !gemmClose(got[i], want[i], exact) {
					t.Fatalf("%s %v [%d] got %v want %v", name, sh, i, got[i], want[i])
				}
			}
		}
		check("NN", gotC, wantC)
		check("AD", gotA, wantA)
		check("BD", gotB, wantB)
	}
}

func BenchmarkGemm(b *testing.B) {
	M, N, K := 1024, 256, 64
	A := make([]float32, M*K)
	Bm := make([]float32, K*N)
	C := make([]float32, M*N)
	for i := range A {
		A[i] = 0.01
	}
	for i := range Bm {
		Bm[i] = 0.02
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range C {
			C[j] = 0
		}
		gemmNN(M, N, K, A, Bm, C)
	}
	b.ReportMetric(2*float64(M*N*K), "flops/op")
}
