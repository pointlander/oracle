// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import "simd"

// f32Hsum adds the lanes of v. VectorBitSize is 128, 256, or 512,
// so a Float32s holds at most 16 lanes.
func f32Hsum(v simd.Float32s) float32 {
	n := v.Len()
	var buf [16]float32
	v.Store(buf[:n])
	s := buf[0]
	for i := 1; i < n; i++ {
		s += buf[i]
	}
	return s
}

// f32Axpy computes y += a*x over the shared prefix.
// Whole vectors use a fused multiply-add. The tail is scalar,
// which keeps shapes narrower than one vector bit-identical to the scalar kernel.
func f32Axpy(a float32, x, y []float32) {
	n := len(y)
	if len(x) < n {
		n = len(x)
	}
	var z simd.Float32s
	vl := z.Len()
	j := 0
	if vl > 0 && n >= vl {
		av := simd.BroadcastFloat32s(a)
		for ; j+vl <= n; j += vl {
			yv := simd.LoadFloat32s(y[j : j+vl])
			xv := simd.LoadFloat32s(x[j : j+vl])
			xv.MulAdd(av, yv).Store(y[j : j+vl])
		}
	}
	for ; j < n; j++ {
		y[j] += a * x[j]
	}
}

// f32Dot is the dot product of the shared prefix.
func f32Dot(x, y []float32) float32 {
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	var a0, a1, a2, a3 simd.Float32s
	vl := a0.Len()
	j := 0
	if vl > 0 {
		step := 4 * vl
		for ; j+step <= n; j += step {
			a0 = simd.LoadFloat32s(x[j:j+vl]).MulAdd(simd.LoadFloat32s(y[j:j+vl]), a0)
			a1 = simd.LoadFloat32s(x[j+vl:j+2*vl]).MulAdd(simd.LoadFloat32s(y[j+vl:j+2*vl]), a1)
			a2 = simd.LoadFloat32s(x[j+2*vl:j+3*vl]).MulAdd(simd.LoadFloat32s(y[j+2*vl:j+3*vl]), a2)
			a3 = simd.LoadFloat32s(x[j+3*vl:j+4*vl]).MulAdd(simd.LoadFloat32s(y[j+3*vl:j+4*vl]), a3)
		}
		for ; j+vl <= n; j += vl {
			a0 = simd.LoadFloat32s(x[j:j+vl]).MulAdd(simd.LoadFloat32s(y[j:j+vl]), a0)
		}
	}
	s := f32Hsum(a0.Add(a1).Add(a2.Add(a3)))
	for ; j < n; j++ {
		s += x[j] * y[j]
	}
	return s
}

// saxpy4 adds four scaled rows into y: y += a0*x0 + a1*x1 + a2*x2 + a3*x3.
// Four adjacent vectors stay in flight so the fused multiply-adds are independent.
// The tail is scalar.
func saxpy4(a0, a1, a2, a3 float32, x0, x1, x2, x3, y []float32) {
	n := len(y)
	if len(x0) < n {
		n = len(x0)
	}
	if len(x1) < n {
		n = len(x1)
	}
	if len(x2) < n {
		n = len(x2)
	}
	if len(x3) < n {
		n = len(x3)
	}
	y, x0, x1, x2, x3 = y[:n], x0[:n], x1[:n], x2[:n], x3[:n]

	var z simd.Float32s
	vl := z.Len()
	j := 0
	if vl > 0 && n >= vl {
		v0 := simd.BroadcastFloat32s(a0)
		v1 := simd.BroadcastFloat32s(a1)
		v2 := simd.BroadcastFloat32s(a2)
		v3 := simd.BroadcastFloat32s(a3)
		step := 4 * vl
		for ; j+step <= n; j += step {
			j1 := j + vl
			j2 := j1 + vl
			j3 := j2 + vl
			j4 := j3 + vl
			c0 := simd.LoadFloat32s(y[j:j1])
			c1 := simd.LoadFloat32s(y[j1:j2])
			c2 := simd.LoadFloat32s(y[j2:j3])
			c3 := simd.LoadFloat32s(y[j3:j4])
			c0 = simd.LoadFloat32s(x0[j:j1]).MulAdd(v0, c0)
			c1 = simd.LoadFloat32s(x0[j1:j2]).MulAdd(v0, c1)
			c2 = simd.LoadFloat32s(x0[j2:j3]).MulAdd(v0, c2)
			c3 = simd.LoadFloat32s(x0[j3:j4]).MulAdd(v0, c3)
			c0 = simd.LoadFloat32s(x1[j:j1]).MulAdd(v1, c0)
			c1 = simd.LoadFloat32s(x1[j1:j2]).MulAdd(v1, c1)
			c2 = simd.LoadFloat32s(x1[j2:j3]).MulAdd(v1, c2)
			c3 = simd.LoadFloat32s(x1[j3:j4]).MulAdd(v1, c3)
			c0 = simd.LoadFloat32s(x2[j:j1]).MulAdd(v2, c0)
			c1 = simd.LoadFloat32s(x2[j1:j2]).MulAdd(v2, c1)
			c2 = simd.LoadFloat32s(x2[j2:j3]).MulAdd(v2, c2)
			c3 = simd.LoadFloat32s(x2[j3:j4]).MulAdd(v2, c3)
			c0 = simd.LoadFloat32s(x3[j:j1]).MulAdd(v3, c0)
			c1 = simd.LoadFloat32s(x3[j1:j2]).MulAdd(v3, c1)
			c2 = simd.LoadFloat32s(x3[j2:j3]).MulAdd(v3, c2)
			c3 = simd.LoadFloat32s(x3[j3:j4]).MulAdd(v3, c3)
			c0.Store(y[j:j1])
			c1.Store(y[j1:j2])
			c2.Store(y[j2:j3])
			c3.Store(y[j3:j4])
		}
		for ; j+vl <= n; j += vl {
			yv := simd.LoadFloat32s(y[j : j+vl])
			yv = simd.LoadFloat32s(x0[j:j+vl]).MulAdd(v0, yv)
			yv = simd.LoadFloat32s(x1[j:j+vl]).MulAdd(v1, yv)
			yv = simd.LoadFloat32s(x2[j:j+vl]).MulAdd(v2, yv)
			yv = simd.LoadFloat32s(x3[j:j+vl]).MulAdd(v3, yv)
			yv.Store(y[j : j+vl])
		}
	}
	for ; j < n; j++ {
		y[j] += a0*x0[j] + a1*x1[j] + a2*x2[j] + a3*x3[j]
	}
}

// gemmNNRange computes C[r0:r1, :] += A[r0:r1, :] @ B.
// A is [M,K] (only rows [r0,r1) are read), B is [K,N], C is [M,N].
func gemmNNRange(r0, r1, N, K int, A, B, C []float32) {
	var z simd.Float32s
	vl := z.Len()
	for i := r0; i < r1; i++ {
		cRow := C[i*N : (i+1)*N]
		aRow := A[i*K : (i+1)*K]
		p := 0
		// Narrow rows stay on the scalar axpy so they match the non-SIMD kernel.
		if vl > 0 && N >= vl {
			for ; p+3 < K; p += 4 {
				saxpy4(
					aRow[p], aRow[p+1], aRow[p+2], aRow[p+3],
					B[p*N:(p+1)*N], B[(p+1)*N:(p+2)*N], B[(p+2)*N:(p+3)*N], B[(p+3)*N:(p+4)*N],
					cRow)
			}
		}
		for ; p < K; p++ {
			f32Axpy(aRow[p], B[p*N:(p+1)*N], cRow)
		}
	}
}

// gemmADRange accumulates dA[r0:r1,:] += dC[r0:r1,:] @ B.T.
// dC is [M,N], B is [K,N], dA is [M,K].
func gemmADRange(r0, r1, N, K int, dC, B, dA []float32) {
	for i := r0; i < r1; i++ {
		dc := dC[i*N : (i+1)*N]
		da := dA[i*K : (i+1)*K]
		for p := 0; p < K; p++ {
			da[p] += f32Dot(dc, B[p*N:(p+1)*N])
		}
	}
}

// gemmBDRange accumulates dB[k,:] += A[:,k]^T @ dC for k in [k0,k1).
// A is [M,K], dC is [M,N], dB is [K,N]. Each k owns one row of dB.
func gemmBDRange(k0, k1, M, N, K int, A, dC, dB []float32) {
	var z simd.Float32s
	vl := z.Len()
	for k := k0; k < k1; k++ {
		row := dB[k*N : (k+1)*N]
		m := 0
		if vl > 0 && N >= vl {
			for ; m+3 < M; m += 4 {
				saxpy4(
					A[(m+0)*K+k], A[(m+1)*K+k], A[(m+2)*K+k], A[(m+3)*K+k],
					dC[(m+0)*N:(m+1)*N], dC[(m+1)*N:(m+2)*N], dC[(m+2)*N:(m+3)*N], dC[(m+3)*N:(m+4)*N],
					row)
			}
		}
		for ; m < M; m++ {
			f32Axpy(A[m*K+k], dC[m*N:(m+1)*N], row)
		}
	}
}
