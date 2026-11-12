// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"crypto/rand"
	"errors"
	"fmt"
	"math/big"
	"strings"
)

// ============================================================================
// 1. THE HYPERCOMPUTATIONAL ORACLE INTERFACE
// ============================================================================

// HaltingOracle represents the uncomputable black-box function H(m, w).
// In true hypercomputation, this resolves in O(1) time without step limits.
type HaltingOracle interface {
	Halts(machineEncoding string, input string) bool
}

// ============================================================================
// 2. APPROXIMATE ORACLE (BOUNDED TURING MACHINE SIMULATOR)
// ============================================================================

// BoundedSimulator approximates a Halting Oracle by running a toy Turing Machine
// up to a maximum number of steps. If it hasn't halted by maxSteps, it assumes
// the machine loops forever (returning false).
type BoundedSimulator struct {
	MaxSteps int
}

// Halts simulates a simple binary-encoded Turing Machine on an input tape.
// Toy Encoding Format (4-bit instructions per state):
// Bit 0: Write value on '0' (0 or 1)
// Bit 1: Move on '0' (0 = Left, 1 = Right)
// Bit 2: Next state on '0' (0 = State 0, 1 = HALT)
// Bit 3: Next state on '1' (0 = State 0, 1 = HALT)
func (sim *BoundedSimulator) Halts(m string, w string) bool {
	if len(m) < 4 {
		return true // Malformed machines halt immediately
	}

	// Initialize tape with input w
	tape := make(map[int]byte)
	for i := 0; i < len(w); i++ {
		tape[i] = w[i]
	}

	head := 0
	state := 0 // 0 = Running, 1 = Halted

	for step := 0; step < sim.MaxSteps; step++ {
		if state == 1 {
			return true // Machine halted!
		}

		currentVal, exists := tape[head]
		if !exists {
			currentVal = '0'
		}

		if currentVal == '0' {
			tape[head] = m[0]
			if m[1] == '0' {
				head--
			} else {
				head++
			}
			if m[2] == '1' {
				state = 1
			}
		} else {
			// On '1', keep value, move right, check halt bit
			head++
			if m[3] == '1' {
				state = 1
			}
		}
	}

	// Exceeded MaxSteps: approximated as non-halting (infinite loop)
	return false
}

// ============================================================================
// 3. THE ORACLE-CONDITIONED GRAMMAR (G^H)
// ============================================================================

type HyperGrammar struct {
	Oracle HaltingOracle
}

// Generate attempts to produce a valid string m#w in L(G^H).
// Rule: S -> M#W -> m#w iff NOT H(m, w)
func (g *HyperGrammar) Generate(maxAttempts int) (string, error) {
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		// 1. Expand non-terminals M and W into candidate binary strings
		m := randomBinaryString(4)
		w := randomBinaryString(3)

		// 2. Apply the Hypercomputational Guard: iff !H(m, w)
		if !g.Oracle.Halts(m, w) {
			return fmt.Sprintf("%s#%s", m, w), nil
		}
	}
	return "", errors.New("generation failed: oracle rejected all candidate derivations")
}

// Parse verifies whether a candidate string belongs to L(G^H).
// It checks both syntactic structure (M#W) and the semantic oracle guard (!H(m, w)).
func (g *HyperGrammar) Parse(candidate string) (bool, string) {
	// Step 1: Syntactic Check (Regular grammar check for {0,1}+ # {0,1}*)
	parts := strings.Split(candidate, "#")
	if len(parts) != 2 {
		return false, "Syntax Error: Expected exact format M#W"
	}

	m, w := parts[0], parts[1]
	if len(m) == 0 || !isBinary(m) || !isBinary(w) {
		return false, "Syntax Error: M must be non-empty binary and W must be binary"
	}

	// Step 2: Hypercomputational Semantic Guard (!H(m, w))
	if g.Oracle.Halts(m, w) {
		return false, fmt.Sprintf("Oracle Rejection: Machine '%s' HALTS on input '%s' (violates !H(m,w))", m, w)
	}

	return true, fmt.Sprintf("Accepted: Machine '%s' loops forever on '%s'", m, w)
}

// ============================================================================
// HELPER FUNCTIONS & EXECUTION
// ============================================================================

func isBinary(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] != '0' && s[i] != '1' {
			return false
		}
	}
	return true
}

func randomBinaryString(length int) string {
	var sb strings.Builder
	for i := 0; i < length; i++ {
		n, _ := rand.Int(rand.Reader, big.NewInt(2))
		if n.Int64() == 0 {
			sb.WriteByte('0')
		} else {
			sb.WriteByte('1')
		}
	}
	return sb.String()
}

func main() {
	// Instantiate the grammar with a 100-step bounded simulator acting as our Oracle
	oracle := &BoundedSimulator{MaxSteps: 100}
	grammar := &HyperGrammar{Oracle: oracle}

	fmt.Println("=== 1. HYPERCOMPUTATIONAL GENERATION ===")
	generated, err := grammar.Generate(50)
	if err != nil {
		fmt.Println("Error:", err)
	} else {
		fmt.Printf("Generated valid string in L(G^H): %s\n\n", generated)
	}

	fmt.Println("=== 2. HYPERCOMPUTATIONAL PARSING ===")
	testStrings := []string{
		generated, // Should pass (known non-halting from generator)
		"0111#00", // Halts immediately when reading '0' (Bit 2 is '1') -> Should fail
		"0100#00", // Writes '0', moves right, stays in state 0 forever -> Should pass
		"10101",   // Missing '#' separator -> Syntax failure
	}

	for _, s := range testStrings {
		valid, reason := grammar.Parse(s)
		status := "INVALID"
		if valid {
			status = "VALID"
		}
		fmt.Printf("Input: %-10s | %-7s | %s\n", s, status, reason)
	}
}
