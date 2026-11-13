// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
)

// autogradEnabled gates graph construction. Eval and sampling turn it off.
var autogradEnabled = true

// Tensor is a float32 n-dimensional array on a tape.
// Leaves that require gradients set gradAcc. Interior nodes get gradAcc from track.
type Tensor struct {
	Data    []float32
	Dims    []int
	Grad    []float32
	gradAcc bool
	isParam bool
	decay   bool
	prev    []*Tensor
	bw      func()
}

func vol(dims []int) int {
	n := 1
	for _, d := range dims {
		if d <= 0 {
			panic(fmt.Sprintf("bad dim %v", dims))
		}
		n *= d
	}
	return n
}

func newTensor(dims []int, data []float32) *Tensor {
	n := vol(dims)
	if data == nil {
		data = make([]float32, n)
	} else if len(data) != n {
		panic(fmt.Sprintf("data len %d != %d for %v", len(data), n, dims))
	}
	dcopy := make([]int, len(dims))
	copy(dcopy, dims)
	return &Tensor{Data: data, Dims: dcopy}
}

func paramTensor(dims []int, rng *rand.Rand, scale float64, decay bool) *Tensor {
	t := newTensor(dims, nil)
	t.gradAcc = true
	t.isParam = true
	t.decay = decay
	if rng != nil && scale != 0 {
		for i := range t.Data {
			t.Data[i] = float32(rng.NormFloat64() * scale)
		}
	}
	return t
}

func scalarTensor(v float32) *Tensor {
	return newTensor([]int{1}, []float32{v})
}

func ensure(cond bool, format string, args ...any) {
	if !cond {
		panic(fmt.Sprintf(format, args...))
	}
}

func track(out *Tensor, parents []*Tensor, bw func()) *Tensor {
	if autogradEnabled {
		out.gradAcc = true
		out.prev = parents
		out.bw = bw
	}
	return out
}

// Backward differentiates a scalar loss. The scalar's own gradient is 1.
func Backward(loss *Tensor) {
	ensure(len(loss.Data) == 1, "Backward expects a scalar, got %v", loss.Dims)
	nodes := make([]*Tensor, 0, 256)
	seen := make(map[*Tensor]bool, 256)
	var dfs func(*Tensor)
	dfs = func(t *Tensor) {
		if t == nil || seen[t] {
			return
		}
		seen[t] = true
		for _, p := range t.prev {
			dfs(p)
		}
		nodes = append(nodes, t)
	}
	dfs(loss)
	for _, n := range nodes {
		if !n.gradAcc {
			continue
		}
		if cap(n.Grad) < len(n.Data) {
			n.Grad = make([]float32, len(n.Data))
		} else {
			n.Grad = n.Grad[:len(n.Data)]
			clear(n.Grad)
		}
	}
	loss.Grad[0] = 1
	for i := len(nodes) - 1; i >= 0; i-- {
		if nodes[i].bw != nil {
			nodes[i].bw()
		}
	}
}

func MatMul(a, b *Tensor) *Tensor {
	ensure(len(a.Dims) == 2 && len(b.Dims) == 2, "MatMul ranks %v %v", a.Dims, b.Dims)
	M, K := a.Dims[0], a.Dims[1]
	K2, N := b.Dims[0], b.Dims[1]
	ensure(K == K2, "MatMul K %d != %d", K, K2)
	out := newTensor([]int{M, N}, nil)
	gemmNN(M, N, K, a.Data, b.Data, out.Data)
	return track(out, []*Tensor{a, b}, func() {
		if a.gradAcc {
			gemmAD(M, N, K, out.Grad, b.Data, a.Grad)
		}
		if b.gradAcc {
			gemmBD(M, N, K, a.Data, out.Grad, b.Grad)
		}
	})
}

// BatchedMatMul multiplies a [B,M,K] by b [B,K,N].
func BatchedMatMul(a, b *Tensor) *Tensor {
	ensure(len(a.Dims) == 3 && len(b.Dims) == 3, "BMM ranks %v %v", a.Dims, b.Dims)
	B, M, K := a.Dims[0], a.Dims[1], a.Dims[2]
	B2, K2, N := b.Dims[0], b.Dims[1], b.Dims[2]
	ensure(B == B2 && K == K2, "BMM shape %v @ %v", a.Dims, b.Dims)
	out := newTensor([]int{B, M, N}, nil)
	aStride := M * K
	bStride := K * N
	cStride := M * N
	parallelRange(B, B*M*N*K, func(i0, i1 int) {
		for i := i0; i < i1; i++ {
			gemmNNRange(0, M, N, K,
				a.Data[i*aStride:(i+1)*aStride],
				b.Data[i*bStride:(i+1)*bStride],
				out.Data[i*cStride:(i+1)*cStride])
		}
	})
	return track(out, []*Tensor{a, b}, func() {
		if a.gradAcc {
			parallelRange(B, B*M*N*K, func(i0, i1 int) {
				for i := i0; i < i1; i++ {
					gemmADRange(0, M, N, K,
						out.Grad[i*cStride:(i+1)*cStride],
						b.Data[i*bStride:(i+1)*bStride],
						a.Grad[i*aStride:(i+1)*aStride])
				}
			})
		}
		if b.gradAcc {
			parallelRange(B, B*M*N*K, func(i0, i1 int) {
				for i := i0; i < i1; i++ {
					gemmBDRange(0, K, M, N, K,
						a.Data[i*aStride:(i+1)*aStride],
						out.Grad[i*cStride:(i+1)*cStride],
						b.Grad[i*bStride:(i+1)*bStride])
				}
			})
		}
	})
}

func Add(a, b *Tensor) *Tensor {
	ensure(len(a.Data) == len(b.Data), "Add len %d != %d", len(a.Data), len(b.Data))
	ensure(len(a.Dims) == len(b.Dims), "Add rank")
	for i := range a.Dims {
		ensure(a.Dims[i] == b.Dims[i], "Add shape %v %v", a.Dims, b.Dims)
	}
	out := newTensor(a.Dims, nil)
	for i := range out.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return track(out, []*Tensor{a, b}, func() {
		if a.gradAcc {
			for i, g := range out.Grad {
				a.Grad[i] += g
			}
		}
		if b.gradAcc {
			for i, g := range out.Grad {
				b.Grad[i] += g
			}
		}
	})
}

// AddBias adds bias[D] onto x[N,D].
func AddBias(x, bias *Tensor) *Tensor {
	ensure(len(x.Dims) == 2 && len(bias.Dims) == 1, "AddBias ranks")
	N, D := x.Dims[0], x.Dims[1]
	ensure(bias.Dims[0] == D, "AddBias %d != %d", bias.Dims[0], D)
	out := newTensor(x.Dims, nil)
	for i := 0; i < N; i++ {
		dst := out.Data[i*D : (i+1)*D]
		src := x.Data[i*D : (i+1)*D]
		for j := 0; j < D; j++ {
			dst[j] = src[j] + bias.Data[j]
		}
	}
	return track(out, []*Tensor{x, bias}, func() {
		if x.gradAcc {
			for i, g := range out.Grad {
				x.Grad[i] += g
			}
		}
		if bias.gradAcc {
			for i := 0; i < N; i++ {
				g := out.Grad[i*D : (i+1)*D]
				for j := 0; j < D; j++ {
					bias.Grad[j] += g[j]
				}
			}
		}
	})
}

func Scale(x *Tensor, s float32) *Tensor {
	out := newTensor(x.Dims, nil)
	for i, v := range x.Data {
		out.Data[i] = v * s
	}
	return track(out, []*Tensor{x}, func() {
		if !x.gradAcc {
			return
		}
		for i, g := range out.Grad {
			x.Grad[i] += s * g
		}
	})
}

func Sum(x *Tensor) *Tensor {
	var s float32
	for _, v := range x.Data {
		s += v
	}
	out := scalarTensor(s)
	return track(out, []*Tensor{x}, func() {
		if !x.gradAcc {
			return
		}
		g := out.Grad[0]
		for i := range x.Grad {
			x.Grad[i] += g
		}
	})
}

func ReLU(x *Tensor) *Tensor {
	out := newTensor(x.Dims, nil)
	for i, v := range x.Data {
		if v > 0 {
			out.Data[i] = v
		}
	}
	return track(out, []*Tensor{x}, func() {
		if !x.gradAcc {
			return
		}
		for i, v := range x.Data {
			if v > 0 {
				x.Grad[i] += out.Grad[i]
			}
		}
	})
}

func Transpose(x *Tensor) *Tensor {
	ensure(len(x.Dims) == 2, "Transpose %v", x.Dims)
	M, N := x.Dims[0], x.Dims[1]
	out := newTensor([]int{N, M}, nil)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			out.Data[j*M+i] = x.Data[i*N+j]
		}
	}
	return track(out, []*Tensor{x}, func() {
		if !x.gradAcc {
			return
		}
		for i := 0; i < M; i++ {
			for j := 0; j < N; j++ {
				x.Grad[i*N+j] += out.Grad[j*M+i]
			}
		}
	})
}

// TransposeLast2 turns [B,M,N] into [B,N,M].
func TransposeLast2(x *Tensor) *Tensor {
	ensure(len(x.Dims) == 3, "TransposeLast2 %v", x.Dims)
	B, M, N := x.Dims[0], x.Dims[1], x.Dims[2]
	out := newTensor([]int{B, N, M}, nil)
	for b := 0; b < B; b++ {
		for i := 0; i < M; i++ {
			for j := 0; j < N; j++ {
				out.Data[(b*N+j)*M+i] = x.Data[(b*M+i)*N+j]
			}
		}
	}
	return track(out, []*Tensor{x}, func() {
		if !x.gradAcc {
			return
		}
		for b := 0; b < B; b++ {
			for i := 0; i < M; i++ {
				for j := 0; j < N; j++ {
					x.Grad[(b*M+i)*N+j] += out.Grad[(b*N+j)*M+i]
				}
			}
		}
	})
}

const lnEps float32 = 1e-5

func LayerNorm(x, gamma, beta *Tensor) *Tensor {
	ensure(len(x.Dims) == 2, "LN rank %v", x.Dims)
	N, D := x.Dims[0], x.Dims[1]
	ensure(len(gamma.Dims) == 1 && gamma.Dims[0] == D, "LN gamma")
	ensure(len(beta.Dims) == 1 && beta.Dims[0] == D, "LN beta")
	out := newTensor([]int{N, D}, nil)
	if !autogradEnabled {
		for i := 0; i < N; i++ {
			row := x.Data[i*D : (i+1)*D]
			var sum float32
			for _, v := range row {
				sum += v
			}
			mean := sum / float32(D)
			var variance float32
			for _, v := range row {
				d := v - mean
				variance += d * d
			}
			variance /= float32(D)
			is := float32(1 / math.Sqrt(float64(variance)+float64(lnEps)))
			orow := out.Data[i*D : (i+1)*D]
			for j := 0; j < D; j++ {
				orow[j] = gamma.Data[j]*(row[j]-mean)*is + beta.Data[j]
			}
		}
		return out
	}
	invstd := make([]float32, N)
	xhat := make([]float32, N*D)
	for i := 0; i < N; i++ {
		row := x.Data[i*D : (i+1)*D]
		var sum float32
		for _, v := range row {
			sum += v
		}
		mean := sum / float32(D)
		var variance float32
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float32(D)
		is := float32(1 / math.Sqrt(float64(variance)+float64(lnEps)))
		invstd[i] = is
		xh := xhat[i*D : (i+1)*D]
		orow := out.Data[i*D : (i+1)*D]
		for j := 0; j < D; j++ {
			v := (row[j] - mean) * is
			xh[j] = v
			orow[j] = gamma.Data[j]*v + beta.Data[j]
		}
	}
	return track(out, []*Tensor{x, gamma, beta}, func() {
		for i := 0; i < N; i++ {
			xh := xhat[i*D : (i+1)*D]
			dy := out.Grad[i*D : (i+1)*D]
			is := invstd[i]
			var sumDxh, sumDxhXhat float32
			for j := 0; j < D; j++ {
				dxh := dy[j] * gamma.Data[j]
				sumDxh += dxh
				sumDxhXhat += dxh * xh[j]
				if gamma.gradAcc {
					gamma.Grad[j] += dy[j] * xh[j]
				}
				if beta.gradAcc {
					beta.Grad[j] += dy[j]
				}
			}
			if !x.gradAcc {
				continue
			}
			sumDxh /= float32(D)
			sumDxhXhat /= float32(D)
			dx := x.Grad[i*D : (i+1)*D]
			for j := 0; j < D; j++ {
				dxh := dy[j] * gamma.Data[j]
				dx[j] += is * (dxh - sumDxh - xh[j]*sumDxhXhat)
			}
		}
	})
}

func SoftmaxLast(x *Tensor) *Tensor {
	ensure(len(x.Dims) >= 1, "softmax")
	last := x.Dims[len(x.Dims)-1]
	rows := len(x.Data) / last
	out := newTensor(x.Dims, nil)
	for r := 0; r < rows; r++ {
		src := x.Data[r*last : (r+1)*last]
		dst := out.Data[r*last : (r+1)*last]
		maxv := src[0]
		for _, v := range src[1:] {
			if v > maxv {
				maxv = v
			}
		}
		var sum float32
		for i, v := range src {
			e := float32(math.Exp(float64(v - maxv)))
			dst[i] = e
			sum += e
		}
		inv := 1 / sum
		for i := range dst {
			dst[i] *= inv
		}
	}
	return track(out, []*Tensor{x}, func() {
		if !x.gradAcc {
			return
		}
		for r := 0; r < rows; r++ {
			y := out.Data[r*last : (r+1)*last]
			dy := out.Grad[r*last : (r+1)*last]
			dx := x.Grad[r*last : (r+1)*last]
			var dot float32
			for i := 0; i < last; i++ {
				dot += y[i] * dy[i]
			}
			for i := 0; i < last; i++ {
				dx[i] += y[i] * (dy[i] - dot)
			}
		}
	})
}

const causalNeg float32 = -1e9

// CausalMask replaces strictly-upper entries of each [T,T] slice with a large negative.
func CausalMask(x *Tensor) *Tensor {
	ensure(len(x.Dims) == 3 && x.Dims[1] == x.Dims[2], "CausalMask %v", x.Dims)
	B, T := x.Dims[0], x.Dims[1]
	out := newTensor(x.Dims, nil)
	copy(out.Data, x.Data)
	for b := 0; b < B; b++ {
		base := b * T * T
		for i := 0; i < T; i++ {
			row := out.Data[base+i*T : base+(i+1)*T]
			for j := i + 1; j < T; j++ {
				row[j] = causalNeg
			}
		}
	}
	return track(out, []*Tensor{x}, func() {
		if !x.gradAcc {
			return
		}
		for b := 0; b < B; b++ {
			base := b * T * T
			for i := 0; i < T; i++ {
				for j := 0; j <= i; j++ {
					x.Grad[base+i*T+j] += out.Grad[base+i*T+j]
				}
			}
		}
	})
}

func Embedding(w *Tensor, idx []int) *Tensor {
	ensure(len(w.Dims) == 2, "Embedding %v", w.Dims)
	V, D := w.Dims[0], w.Dims[1]
	N := len(idx)
	out := newTensor([]int{N, D}, nil)
	for n, id := range idx {
		ensure(id >= 0 && id < V, "embedding id %d vocab %d", id, V)
		copy(out.Data[n*D:(n+1)*D], w.Data[id*D:(id+1)*D])
	}
	return track(out, []*Tensor{w}, func() {
		if !w.gradAcc {
			return
		}
		for n, id := range idx {
			g := out.Grad[n*D : (n+1)*D]
			dst := w.Grad[id*D : (id+1)*D]
			for d := 0; d < D; d++ {
				dst[d] += g[d]
			}
		}
	})
}

// AddPosition adds pos[t] to each batch row. pos may be longer than T.
func AddPosition(x, pos *Tensor, B, T int) *Tensor {
	ensure(len(x.Dims) == 2 && len(pos.Dims) == 2, "AddPosition ranks")
	D := x.Dims[1]
	ensure(x.Dims[0] == B*T, "AddPosition rows %d", x.Dims[0])
	ensure(pos.Dims[0] >= T && pos.Dims[1] == D, "AddPosition pos %v T %d D %d", pos.Dims, T, D)
	out := newTensor([]int{B * T, D}, nil)
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			dst := out.Data[(b*T+t)*D : (b*T+t+1)*D]
			src := x.Data[(b*T+t)*D : (b*T+t+1)*D]
			p := pos.Data[t*D : (t+1)*D]
			for d := 0; d < D; d++ {
				dst[d] = src[d] + p[d]
			}
		}
	}
	return track(out, []*Tensor{x, pos}, func() {
		if x.gradAcc {
			for i, g := range out.Grad {
				x.Grad[i] += g
			}
		}
		if pos.gradAcc {
			for b := 0; b < B; b++ {
				for t := 0; t < T; t++ {
					g := out.Grad[(b*T+t)*D : (b*T+t+1)*D]
					pg := pos.Grad[t*D : (t+1)*D]
					for d := 0; d < D; d++ {
						pg[d] += g[d]
					}
				}
			}
		}
	})
}

// ArrangeHeads packs [B*T, H*Hd] tokens into [B*H, T, Hd].
func ArrangeHeads(x *Tensor, B, T, H, Hd int) *Tensor {
	ensure(x.Dims[0] == B*T && x.Dims[1] == H*Hd, "Arrange %v", x.Dims)
	out := newTensor([]int{B * H, T, Hd}, nil)
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			src := x.Data[(b*T+t)*H*Hd : (b*T+t+1)*H*Hd]
			for h := 0; h < H; h++ {
				dst := ((b*H+h)*T + t) * Hd
				copy(out.Data[dst:dst+Hd], src[h*Hd:(h+1)*Hd])
			}
		}
	}
	return track(out, []*Tensor{x}, func() {
		if !x.gradAcc {
			return
		}
		for b := 0; b < B; b++ {
			for t := 0; t < T; t++ {
				dst := x.Grad[(b*T+t)*H*Hd : (b*T+t+1)*H*Hd]
				for h := 0; h < H; h++ {
					src := ((b*H+h)*T + t) * Hd
					for d := 0; d < Hd; d++ {
						dst[h*Hd+d] += out.Grad[src+d]
					}
				}
			}
		}
	})
}

// UnarrangeHeads inverts ArrangeHeads back to [B*T, H*Hd].
func UnarrangeHeads(x *Tensor, B, T, H, Hd int) *Tensor {
	ensure(len(x.Dims) == 3 && x.Dims[0] == B*H && x.Dims[1] == T && x.Dims[2] == Hd, "Unarrange %v", x.Dims)
	out := newTensor([]int{B * T, H * Hd}, nil)
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			dst := out.Data[(b*T+t)*H*Hd : (b*T+t+1)*H*Hd]
			for h := 0; h < H; h++ {
				src := ((b*H+h)*T + t) * Hd
				copy(dst[h*Hd:(h+1)*Hd], x.Data[src:src+Hd])
			}
		}
	}
	return track(out, []*Tensor{x}, func() {
		if !x.gradAcc {
			return
		}
		for b := 0; b < B; b++ {
			for t := 0; t < T; t++ {
				src := out.Grad[(b*T+t)*H*Hd : (b*T+t+1)*H*Hd]
				for h := 0; h < H; h++ {
					dst := ((b*H+h)*T + t) * Hd
					for d := 0; d < Hd; d++ {
						x.Grad[dst+d] += src[h*Hd+d]
					}
				}
			}
		}
	})
}

// MeanSeq averages groups of T rows. x is [B*T, D], result is [B, D].
func MeanSeq(x *Tensor, B, T int) *Tensor {
	ensure(len(x.Dims) == 2 && x.Dims[0] == B*T, "MeanSeq %v", x.Dims)
	D := x.Dims[1]
	out := newTensor([]int{B, D}, nil)
	inv := 1 / float32(T)
	for b := 0; b < B; b++ {
		acc := out.Data[b*D : (b+1)*D]
		for t := 0; t < T; t++ {
			src := x.Data[(b*T+t)*D : (b*T+t+1)*D]
			for d := 0; d < D; d++ {
				acc[d] += src[d]
			}
		}
		for d := 0; d < D; d++ {
			acc[d] *= inv
		}
	}
	return track(out, []*Tensor{x}, func() {
		if !x.gradAcc {
			return
		}
		for b := 0; b < B; b++ {
			g := out.Grad[b*D : (b+1)*D]
			for t := 0; t < T; t++ {
				dst := x.Grad[(b*T+t)*D : (b*T+t+1)*D]
				for d := 0; d < D; d++ {
					dst[d] += g[d] * inv
				}
			}
		}
	})
}

// RepeatSeq repeats each row of v[B,D] across T positions -> [B*T, D].
func RepeatSeq(v *Tensor, T int) *Tensor {
	ensure(len(v.Dims) == 2, "RepeatSeq")
	B, D := v.Dims[0], v.Dims[1]
	out := newTensor([]int{B * T, D}, nil)
	for b := 0; b < B; b++ {
		src := v.Data[b*D : (b+1)*D]
		for t := 0; t < T; t++ {
			copy(out.Data[(b*T+t)*D:(b*T+t+1)*D], src)
		}
	}
	return track(out, []*Tensor{v}, func() {
		if !v.gradAcc {
			return
		}
		for b := 0; b < B; b++ {
			acc := v.Grad[b*D : (b+1)*D]
			for t := 0; t < T; t++ {
				g := out.Grad[(b*T+t)*D : (b*T+t+1)*D]
				for d := 0; d < D; d++ {
					acc[d] += g[d]
				}
			}
		}
	})
}

func sigmoid(x float32) float32 {
	if x >= 0 {
		z := math.Exp(float64(-x))
		return float32(1 / (1 + z))
	}
	z := math.Exp(float64(x))
	return float32(z / (1 + z))
}

// stGroup is a slice of logits that is either independent Bernoulli bits
// or one categorical distribution.
type stGroup struct {
	Offset int
	N      int
	Cat    bool
}

func writeHard(dst, logits []float32, groups []stGroup) {
	for i := range dst {
		dst[i] = 0
	}
	for _, g := range groups {
		if !g.Cat {
			for i := 0; i < g.N; i++ {
				if logits[g.Offset+i] > 0 {
					dst[g.Offset+i] = 1
				}
			}
			continue
		}
		best := 0
		bestV := logits[g.Offset]
		for i := 1; i < g.N; i++ {
			if logits[g.Offset+i] > bestV {
				bestV = logits[g.Offset+i]
				best = i
			}
		}
		dst[g.Offset+best] = 1
	}
}

// MixedST emits hard Bernoulli / one-hot decisions and backpropagates
// through the sigmoid / softmax surrogate (straight-through estimator).
func MixedST(logits *Tensor, groups []stGroup) *Tensor {
	ensure(len(logits.Dims) == 2, "MixedST")
	B, F := logits.Dims[0], logits.Dims[1]
	out := newTensor([]int{B, F}, nil)
	for b := 0; b < B; b++ {
		writeHard(out.Data[b*F:(b+1)*F], logits.Data[b*F:(b+1)*F], groups)
	}
	return track(out, []*Tensor{logits}, func() {
		if !logits.gradAcc {
			return
		}
		for b := 0; b < B; b++ {
			z := logits.Data[b*F : (b+1)*F]
			gOut := out.Grad[b*F : (b+1)*F]
			gZ := logits.Grad[b*F : (b+1)*F]
			for _, g := range groups {
				if !g.Cat {
					for i := 0; i < g.N; i++ {
						p := sigmoid(z[g.Offset+i])
						gZ[g.Offset+i] += gOut[g.Offset+i] * p * (1 - p)
					}
					continue
				}
				soft := make([]float32, g.N)
				maxv := z[g.Offset]
				for i := 1; i < g.N; i++ {
					if z[g.Offset+i] > maxv {
						maxv = z[g.Offset+i]
					}
				}
				var sum float32
				for i := 0; i < g.N; i++ {
					e := float32(math.Exp(float64(z[g.Offset+i] - maxv)))
					soft[i] = e
					sum += e
				}
				inv := 1 / sum
				var dot float32
				for i := 0; i < g.N; i++ {
					soft[i] *= inv
					dot += soft[i] * gOut[g.Offset+i]
				}
				for i := 0; i < g.N; i++ {
					gZ[g.Offset+i] += soft[i] * (gOut[g.Offset+i] - dot)
				}
			}
		}
	})
}

func bceTerm(z, y float64) float64 {
	az := z
	if az < 0 {
		az = -az
	}
	maxz := z
	if maxz < 0 {
		maxz = 0
	}
	return maxz - z*y + math.Log1p(math.Exp(-az))
}

// BCEWithLogits is mean binary cross-entropy. z is [B] or [B,1].
// Targets are detached oracle bits.
func BCEWithLogits(z *Tensor, y []int) *Tensor {
	B := z.Dims[0]
	ensure(len(z.Data) == B && len(y) == B, "BCE shape %v vs %d", z.Dims, len(y))
	var loss float64
	sig := make([]float32, B)
	for i := 0; i < B; i++ {
		loss += bceTerm(float64(z.Data[i]), float64(y[i]))
		sig[i] = sigmoid(z.Data[i])
	}
	out := scalarTensor(float32(loss / float64(B)))
	if !autogradEnabled {
		return out
	}
	return track(out, []*Tensor{z}, func() {
		if !z.gradAcc {
			return
		}
		g := out.Grad[0] / float32(B)
		for i := 0; i < B; i++ {
			z.Grad[i] += g * (sig[i] - float32(y[i]))
		}
	})
}

// CrossEntropy is mean next-token loss in nats. logits is [N,V].
func CrossEntropy(logits *Tensor, target []int) *Tensor {
	ensure(len(logits.Dims) == 2, "CE rank")
	N, V := logits.Dims[0], logits.Dims[1]
	ensure(len(target) == N, "CE targets %d != %d", len(target), N)
	var loss float64
	var probs []float32
	if autogradEnabled {
		probs = make([]float32, N*V)
	}
	for i := 0; i < N; i++ {
		row := logits.Data[i*V : (i+1)*V]
		maxv := row[0]
		for _, v := range row[1:] {
			if v > maxv {
				maxv = v
			}
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v) - float64(maxv))
		}
		t := target[i]
		ensure(t >= 0 && t < V, "target %d vocab %d", t, V)
		logp := float64(row[t]) - float64(maxv) - math.Log(sum)
		loss -= logp
		if probs != nil {
			dst := probs[i*V : (i+1)*V]
			inv := 1 / sum
			for j, v := range row {
				dst[j] = float32(math.Exp(float64(v)-float64(maxv)) * inv)
			}
		}
	}
	out := scalarTensor(float32(loss / float64(N)))
	if !autogradEnabled {
		return out
	}
	return track(out, []*Tensor{logits}, func() {
		if !logits.gradAcc {
			return
		}
		invN := out.Grad[0] / float32(N)
		for i := 0; i < N; i++ {
			g := logits.Grad[i*V : (i+1)*V]
			p := probs[i*V : (i+1)*V]
			t := target[i]
			for j := 0; j < V; j++ {
				delta := p[j]
				if j == t {
					delta -= 1
				}
				g[j] += invN * delta
			}
		}
	})
}
