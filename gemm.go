// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"sync"
)

// parallelRange splits work [0,n) across cores when cost is large enough
// that the launch overhead is worth it. fn ranges are contiguous and disjoint.
func parallelRange(n, cost int, fn func(start, end int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || n < 2 || cost < 80_000 {
		fn(0, n)
		return
	}
	if workers > n {
		workers = n
	}
	var wg sync.WaitGroup
	wg.Add(workers)
	base := n / workers
	rem := n % workers
	start := 0
	for w := 0; w < workers; w++ {
		sz := base
		if w < rem {
			sz++
		}
		s, e := start, start+sz
		start = e
		go func(s, e int) {
			defer wg.Done()
			fn(s, e)
		}(s, e)
	}
	wg.Wait()
}

// gemmNN, gemmAD, and gemmBD are C += A@B and the two gradient products.
// The row kernels are gemmNNRange, gemmADRange, and gemmBDRange.
// gemm_simd.go implements them with simd.Float32s when GOEXPERIMENT=simd.
// gemm_scalar.go is the same arithmetic without that experiment.

func gemmNN(M, N, K int, A, B, C []float32) {
	parallelRange(M, M*N*K, func(a, b int) {
		gemmNNRange(a, b, N, K, A, B, C)
	})
}

func gemmAD(M, N, K int, dC, B, dA []float32) {
	parallelRange(M, M*N*K, func(a, b int) {
		gemmADRange(a, b, N, K, dC, B, dA)
	})
}

func gemmBD(M, N, K int, A, dC, dB []float32) {
	parallelRange(K, M*N*K, func(a, b int) {
		gemmBDRange(a, b, M, N, K, A, dC, dB)
	})
}
