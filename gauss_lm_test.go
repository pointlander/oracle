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

func TestContextWriteMatchesTheByte(t *testing.T) {
	const target = byte(80)
	const tau = 8.0
	w := fitContextWrite(target, tau, 16)
	_, dmu := symbolLoss(w, tau, 0)
	expect := dmu * tau * tau
	if math.Abs(expect-float64(target)) > 1e-2 {
		t.Fatalf("expectation %.4f target %d at %.4f", expect, target, w)
	}
	if nearestByte(w) != target {
		t.Fatalf("nearest %d", nearestByte(w))
	}
}

func contextNode(m *gaussLM, ctx []byte) *lmNode {
	n := &m.root
	for i := len(ctx) - 1; i >= 0; i-- {
		n = n.find(ctx[i])
		if n == nil {
			return nil
		}
	}
	return n
}

func TestZeroByteUsesTheOtherWrite(t *testing.T) {
	// Newest byte 0 reads w0. Newest byte 'b' reads w1, including the longer
	// context whose older byte is 0.
	text := []byte{0, 'b', 'Z', 0, 'b', 'Z'}
	m := newGaussLM(2, 8)
	m.observe(text, 8)
	zero := contextNode(m, []byte{0})
	bee := contextNode(m, []byte{'b'})
	longer := contextNode(m, []byte{0, 'b'})
	if zero == nil || bee == nil || longer == nil {
		t.Fatal("missing context")
	}
	if nearestByte(zero.w0) != 'b' || zero.w1 != 0 {
		t.Fatalf("zero w0 %.3f w1 %.3f", zero.w0, zero.w1)
	}
	if nearestByte(bee.w1) != 'Z' || bee.w0 != 0 {
		t.Fatalf("b w0 %.3f w1 %.3f", bee.w0, bee.w1)
	}
	if nearestByte(longer.w1) != 'Z' || longer.w0 != 0 {
		t.Fatalf("longer w0 %.3f w1 %.3f", longer.w0, longer.w1)
	}
	if nearestByte(m.weight([]byte{0, 'b'})) != 'Z' || nearestByte(m.weight([]byte{'Z', 0})) != 'b' {
		t.Fatalf("weight %q %q", nearestByte(m.weight([]byte{0, 'b'})), nearestByte(m.weight([]byte{'Z', 0})))
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
