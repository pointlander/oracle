// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.simd

package main

// gemmNNRange computes C[r0:r1, :] += A[r0:r1, :] @ B
// A is [M,K] (only rows [r0,r1) are read), B is [K,N], C is [M,N].
func gemmNNRange(r0, r1, N, K int, A, B, C []float32) {
	for i := r0; i < r1; i++ {
		cRow := C[i*N : (i+1)*N]
		aRow := A[i*K : (i+1)*K]
		for p := 0; p < K; p++ {
			av := aRow[p]
			bRow := B[p*N : (p+1)*N]
			j := 0
			n4 := N &^ 3
			for ; j < n4; j += 4 {
				cRow[j] += av * bRow[j]
				cRow[j+1] += av * bRow[j+1]
				cRow[j+2] += av * bRow[j+2]
				cRow[j+3] += av * bRow[j+3]
			}
			for ; j < N; j++ {
				cRow[j] += av * bRow[j]
			}
		}
	}
}

// gemmADRange accumulates dA[r0:r1,:] += dC[r0:r1,:] @ B.T
// dC is [M,N], B is [K,N], dA is [M,K].
func gemmADRange(r0, r1, N, K int, dC, B, dA []float32) {
	for i := r0; i < r1; i++ {
		dc := dC[i*N : (i+1)*N]
		da := dA[i*K : (i+1)*K]
		for p := 0; p < K; p++ {
			bRow := B[p*N : (p+1)*N]
			var s float32
			j := 0
			n4 := N &^ 3
			for ; j < n4; j += 4 {
				s += dc[j]*bRow[j] + dc[j+1]*bRow[j+1] + dc[j+2]*bRow[j+2] + dc[j+3]*bRow[j+3]
			}
			for ; j < N; j++ {
				s += dc[j] * bRow[j]
			}
			da[p] += s
		}
	}
}

// gemmBDRange accumulates dB[k,:] += A[:,k]^T @ dC for k in [k0,k1).
// A is [M,K], dC is [M,N], dB is [K,N]. Parallelize over K: each k owns one row.
func gemmBDRange(k0, k1, M, N, K int, A, dC, dB []float32) {
	for k := k0; k < k1; k++ {
		row := dB[k*N : (k+1)*N]
		for m := 0; m < M; m++ {
			av := A[m*K+k]
			src := dC[m*N : (m+1)*N]
			j := 0
			n4 := N &^ 3
			for ; j < n4; j += 4 {
				row[j] += av * src[j]
				row[j+1] += av * src[j+1]
				row[j+2] += av * src[j+2]
				row[j+3] += av * src[j+3]
			}
			for ; j < N; j++ {
				row[j] += av * src[j]
			}
		}
	}
}
