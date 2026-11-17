// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"math"
	"os"
	"testing"
)

func TestContextMeanMatchesTheExpectation(t *testing.T) {
	const mean = 80.0
	const tau = 8.0
	mu := fitContextMean(mean, tau, 16)
	_, dmu := symbolLoss(mu, tau, 0)
	expect := dmu * tau * tau
	if math.Abs(expect-mean) > 1e-2 {
		t.Fatalf("expectation %.4f mean %.4f at mu %.4f", expect, mean, mu)
	}
}

func TestGaussLMOrder2BeatsOrder1(t *testing.T) {
	// "xb" is always followed by c, and "yb" is always followed by d.
	// One previous byte cannot tell those apart.
	text := bytes.Repeat([]byte("xbcybd"), 50)
	longer := newGaussLM(2, 8)
	longer.observe(text, 8)
	_, hit2, n := longer.score(text, 2)
	_, hit1, n1 := longer.score(text, 1)
	// The first two bytes have no full context, so they may miss.
	if n != n1 || n == 0 || hit2 <= hit1 || hit2 < n-2 {
		t.Fatalf("order2 %d/%d order1 %d/%d", hit2, n, hit1, n1)
	}
}

func TestGaussLMPredictsTheMode(t *testing.T) {
	// 'q' is followed by 'a' three times and 'z' once. The average code is
	// not 'a', and the hard read has to be the common byte.
	text := []byte("qaqaqaqz")
	m := newGaussLM(1, 8)
	m.observe(text, 8)
	if nearestByte(m.weight([]byte{'q'})) != 'a' {
		t.Fatalf("got %q", nearestByte(m.weight([]byte{'q'})))
	}
}

func TestGaussLMAlternates(t *testing.T) {
	text := bytes.Repeat([]byte("ab"), 30)
	m := newGaussLM(1, 8)
	m.observe(text, 8)
	_, hit, n := m.score(text, 1)
	// The first byte has only the unigram context.
	if hit < n-1 {
		t.Fatalf("acc %d/%d", hit, n)
	}
	if nearestByte(m.weight([]byte{'a'})) != 'b' || nearestByte(m.weight([]byte{'b'})) != 'a' {
		t.Fatalf("a→%q b→%q", nearestByte(m.weight([]byte{'a'})), nearestByte(m.weight([]byte{'b'})))
	}
}

func TestGaussLMOrder3BeatsOrder2(t *testing.T) {
	// "pxa" is always followed by b, and "qxa" is always followed by c.
	// The two bytes "xa" are followed by both.
	text := bytes.Repeat([]byte("pxabqxac"), 40)
	m := newGaussLM(3, 8)
	m.observe(text, 8)
	_, hit3, n := m.score(text, 3)
	_, hit2, n2 := m.score(text, 2)
	if n != n2 || n == 0 || hit3 <= hit2 || hit3 < n-3 {
		t.Fatalf("order3 %d/%d order2 %d/%d", hit3, n, hit2, n2)
	}
}

func TestGaussLMBeatsUnigramOnShakespeare(t *testing.T) {
	data, err := os.ReadFile("100.txt.utf-8")
	if err != nil {
		t.Fatal(err)
	}
	if len(data) > 80000 {
		data = data[:80000]
	}
	train, val, err := split90(data)
	if err != nil {
		t.Fatal(err)
	}
	m := newGaussLM(2, 8)
	m.observe(train, 8)
	_, hit, n := m.score(val, 2)
	_, uni, n0 := m.score(val, 0)
	if n != n0 || n == 0 || hit <= uni {
		t.Fatalf("order2 %d/%d unigram %d/%d", hit, n, uni, n0)
	}
}
