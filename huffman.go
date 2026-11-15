// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"container/heap"
	"fmt"
)

// huffmanCode is a byte-level Huffman code. Bit 0 is the left branch.
// Codes are assigned from symbol frequencies. A symbol that does not occur has an empty code.
type huffmanCode struct {
	code [256]string
	root *huffNode
}

type huffNode struct {
	sym   int
	id    int
	left  *huffNode
	right *huffNode
}

type huffHeap []huffItem

type huffItem struct {
	node *huffNode
	freq int
	id   int
}

func (h huffHeap) Len() int { return len(h) }
func (h huffHeap) Less(i, j int) bool {
	if h[i].freq != h[j].freq {
		return h[i].freq < h[j].freq
	}
	return h[i].id < h[j].id
}
func (h huffHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *huffHeap) Push(x interface{}) { *h = append(*h, x.(huffItem)) }
func (h *huffHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// buildHuffman builds a deterministic code from byte counts.
// freq[b] == 0 omits that byte. At least one byte must occur.
func buildHuffman(freq [256]int) (huffmanCode, error) {
	var leaves []huffItem
	for sym, n := range freq {
		if n <= 0 {
			continue
		}
		leaves = append(leaves, huffItem{
			node: &huffNode{sym: sym, id: sym},
			freq: n,
			id:   sym,
		})
	}
	if len(leaves) == 0 {
		return huffmanCode{}, fmt.Errorf("huffman: no symbols")
	}
	if len(leaves) == 1 {
		var code huffmanCode
		code.code[leaves[0].node.sym] = "0"
		code.root = leaves[0].node
		return code, nil
	}
	h := &huffHeap{}
	for _, leaf := range leaves {
		heap.Push(h, leaf)
	}
	nextID := 256
	for h.Len() > 1 {
		a := heap.Pop(h).(huffItem)
		b := heap.Pop(h).(huffItem)
		if a.id > b.id {
			a, b = b, a
		}
		parent := &huffNode{sym: -1, id: nextID, left: a.node, right: b.node}
		heap.Push(h, huffItem{node: parent, freq: a.freq + b.freq, id: nextID})
		nextID++
	}
	root := heap.Pop(h).(huffItem).node
	var code huffmanCode
	code.root = root
	assignHuffman(&code, root, "")
	return code, nil
}

func assignHuffman(code *huffmanCode, n *huffNode, prefix string) {
	if n.left == nil && n.right == nil {
		code.code[n.sym] = prefix
		return
	}
	assignHuffman(code, n.left, prefix+"0")
	assignHuffman(code, n.right, prefix+"1")
}

func freqOf(data []byte) [256]int {
	var freq [256]int
	for _, b := range data {
		freq[b]++
	}
	return freq
}

// encodeHuffman packs data with code. The first emitted bit is the high bit of packed[0].
func encodeHuffman(code huffmanCode, data []byte) (packed []byte, nbits int, err error) {
	if len(data) == 0 {
		return nil, 0, fmt.Errorf("huffman: empty input")
	}
	var acc byte
	var fill int
	for _, b := range data {
		bits := code.code[b]
		if bits == "" {
			return nil, 0, fmt.Errorf("huffman: byte %d has no code", b)
		}
		for i := 0; i < len(bits); i++ {
			acc <<= 1
			if bits[i] == '1' {
				acc |= 1
			}
			fill++
			nbits++
			if fill == 8 {
				packed = append(packed, acc)
				acc = 0
				fill = 0
			}
		}
	}
	if fill > 0 {
		acc <<= 8 - fill
		packed = append(packed, acc)
	}
	return packed, nbits, nil
}

// decodeHuffman restores exactly n bytes from the first nbits of packed.
func decodeHuffman(code huffmanCode, packed []byte, nbits, n int) ([]byte, error) {
	if n < 0 || nbits < 0 {
		return nil, fmt.Errorf("huffman: bad length")
	}
	if code.root == nil {
		return nil, fmt.Errorf("huffman: empty code")
	}
	out := make([]byte, 0, n)
	if code.root.left == nil && code.root.right == nil {
		for i := 0; i < nbits && len(out) < n; i++ {
			if bitAt(packed, i) != 0 {
				return nil, fmt.Errorf("huffman: bad single-symbol bit")
			}
			out = append(out, byte(code.root.sym))
		}
		if len(out) != n {
			return nil, fmt.Errorf("huffman: decoded %d bytes, want %d", len(out), n)
		}
		return out, nil
	}
	node := code.root
	for i := 0; i < nbits && len(out) < n; i++ {
		if bitAt(packed, i) == 0 {
			node = node.left
		} else {
			node = node.right
		}
		if node == nil {
			return nil, fmt.Errorf("huffman: walked off the tree")
		}
		if node.left == nil && node.right == nil {
			out = append(out, byte(node.sym))
			node = code.root
		}
	}
	if len(out) != n {
		return nil, fmt.Errorf("huffman: decoded %d bytes, want %d", len(out), n)
	}
	return out, nil
}

func bitAt(packed []byte, i int) int {
	return int((packed[i>>3] >> (7 - (i & 7))) & 1)
}

func unpackBits(packed []byte, nbits int) []byte {
	out := make([]byte, nbits)
	for i := 0; i < nbits; i++ {
		out[i] = byte(bitAt(packed, i))
	}
	return out
}

func huffmanSymbols(code huffmanCode) int {
	n := 0
	for _, c := range code.code {
		if c != "" {
			n++
		}
	}
	return n
}
