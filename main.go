// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"crypto/rand"
	"errors"
	"fmt"
	"math/big"
	"os"
	"strconv"
	"strings"
)

// ============================================================================
// 1. THE HYPERCOMPUTATIONAL ORACLE INTERFACE
// ============================================================================

type HaltingOracle interface {
	Halts(machineEncoding string, input string) bool
}

// ============================================================================
// 2. UNIVERSAL TURING MACHINE (MULTI-STATE 5-TUPLE SIMULATOR)
// ============================================================================

// Transition defines a single Turing Machine rule:
// (CurrentState, ReadSymbol) -> (WriteSymbol, MoveDirection, NextState)
type Transition struct {
	WriteSymbol byte // '0' or '1'
	MoveDir     int  // -1 (Left), +1 (Right)
	NextState   int  // 0 = HALT, 1..N = Active States
}

// UniversalBoundedSimulator executes arbitrary multi-state binary Turing Machines.
//
// Binary Encoding Format for `m`:
// Every active state (starting from State 1) is encoded as a 16-bit block:
//   - Bits 0..7:  Transition when reading '0'
//   - Bits 8..15: Transition when reading '1'
//
// Each 8-bit transition instruction is structured as:
//   - Bit 0:    Write symbol ('0' or '1')
//   - Bit 1:    Move direction ('0' = Left, '1' = Right)
//   - Bits 2..7: 6-bit unsigned integer for NextState (0 = HALT, 1..63 = State)
type UniversalBoundedSimulator struct {
	MaxSteps int
}

func (utm *UniversalBoundedSimulator) Halts(m string, w string) bool {
	// Validate that `m` consists of complete 16-bit state blocks
	if len(m) == 0 || len(m)%16 != 0 {
		return true // Malformed machines halt immediately
	}

	numStates := len(m) / 16
	table := make(map[int][2]Transition)

	// Decode binary string `m` into the state transition table
	for s := 1; s <= numStates; s++ {
		offset := (s - 1) * 16
		t0, ok0 := decodeTransition(m[offset : offset+8])
		t1, ok1 := decodeTransition(m[offset+8 : offset+16])
		if !ok0 || !ok1 {
			return true
		}
		table[s] = [2]Transition{t0, t1}
	}

	// Initialize infinite tape with input `w`
	tape := make(map[int]byte)
	for i := 0; i < len(w); i++ {
		tape[i] = w[i]
	}

	head := 0
	state := 1 // Start in State 1 (State 0 is the universal HALT state)

	for step := 0; step < utm.MaxSteps; step++ {
		if state == 0 {
			return true // Machine halted!
		}

		// Jump to an undefined state triggers an immediate halt
		transitions, exists := table[state]
		if !exists {
			return true
		}

		currentVal, cellExists := tape[head]
		if !cellExists {
			currentVal = '0'
		}

		var rule Transition
		if currentVal == '0' {
			rule = transitions[0]
		} else {
			rule = transitions[1]
		}

		tape[head] = rule.WriteSymbol
		head += rule.MoveDir
		state = rule.NextState
	}

	// Exceeded MaxSteps: approximated as non-halting (infinite loop)
	return false
}

func decodeTransition(bits string) (Transition, bool) {
	if len(bits) != 8 {
		return Transition{}, false
	}

	writeSym := bits[0]
	moveDir := -1
	if bits[1] == '1' {
		moveDir = 1
	}

	nextStateVal, err := strconv.ParseInt(bits[2:8], 2, 32)
	if err != nil {
		return Transition{}, false
	}

	return Transition{
		WriteSymbol: writeSym,
		MoveDir:     moveDir,
		NextState:   int(nextStateVal),
	}, true
}

// ============================================================================
// 3. HELPER TO COMPILE HUMAN-READABLE MACHINES INTO UTM BINARY
// ============================================================================

// EncodeMachine compiles a map of state transitions into the binary string `m`.
func EncodeMachine(states map[int][2]Transition) string {
	var sb strings.Builder
	for s := 1; s <= len(states); s++ {
		pair := states[s]
		sb.WriteString(encodeTransition(pair[0]))
		sb.WriteString(encodeTransition(pair[1]))
	}
	return sb.String()
}

func encodeTransition(t Transition) string {
	dirBit := '0'
	if t.MoveDir == 1 {
		dirBit = '1'
	}
	return fmt.Sprintf("%c%c%06b", t.WriteSymbol, dirBit, t.NextState)
}

// ============================================================================
// 4. THE HYPERCOMPUTATIONAL GRAMMAR (G^H)
// ============================================================================

type HyperGrammar struct {
	Oracle HaltingOracle
}

func (g *HyperGrammar) Generate(numStates int, maxAttempts int) (string, error) {
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		m := randomBinaryString(numStates * 16)
		w := randomBinaryString(4)

		if !g.Oracle.Halts(m, w) {
			return fmt.Sprintf("%s#%s", m, w), nil
		}
	}
	return "", errors.New("generation failed: oracle rejected all candidate derivations")
}

func (g *HyperGrammar) Parse(candidate string) (bool, string) {
	parts := strings.Split(candidate, "#")
	if len(parts) != 2 {
		return false, "Syntax Error: Expected exact format M#W"
	}

	m, w := parts[0], parts[1]
	if len(m) == 0 || !isBinary(m) || !isBinary(w) {
		return false, "Syntax Error: M must be non-empty binary and W must be binary"
	}

	if g.Oracle.Halts(m, w) {
		return false, fmt.Sprintf("Oracle Rejection: Machine HALTS on input '%s' (violates !H(m,w))", w)
	}

	return true, fmt.Sprintf("Accepted: %d-state machine loops forever on '%s'", len(m)/16, w)
}

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

// ============================================================================
// 5. DEMONSTRATION: MULTI-STATE PROOF SEARCH & GÖDEL ENCODING
// ============================================================================

func main() {
	if len(os.Args) > 1 && (os.Args[1] == "train" || os.Args[1] == "-train") {
		if err := trainFromArgs(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "train: %v\n", err)
			os.Exit(1)
		}
		return
	}
	demo()
}

func demo() {
	// Instantiate UTM with a 5,000-step horizon
	utm := &UniversalBoundedSimulator{MaxSteps: 5000}
	grammar := &HyperGrammar{Oracle: utm}

	// Build a 3-State Binary Counter / Proof-Search Machine:
	// It treats the tape to the right of the input as a binary workspace,
	// systematically incrementing binary strings (0 -> 1 -> 10 -> 11 -> 100...)
	// searching for a "proof witness" (which never triggers a halt if unprovable).
	proofSearcher := EncodeMachine(map[int][2]Transition{
		// State 1: Scan right to the end of the current candidate proof string
		1: {
			{WriteSymbol: '1', MoveDir: -1, NextState: 2}, // On '0' (end of number): write '1', carry/reset left
			{WriteSymbol: '1', MoveDir: 1, NextState: 1},  // On '1': keep scanning right
		},
		// State 2: Propagate binary carry leftward across '1's
		2: {
			{WriteSymbol: '0', MoveDir: 1, NextState: 1},  // On '0': reached left boundary, resume forward search
			{WriteSymbol: '0', MoveDir: -1, NextState: 2}, // On '1': flip to '0' and continue carrying left
		},
		// State 3: Unreachable "Proof Found -> Halt" verifier state
		3: {
			{WriteSymbol: '1', MoveDir: 1, NextState: 0}, // HALT
			{WriteSymbol: '1', MoveDir: 1, NextState: 0}, // HALT
		},
	})

	// Build a 2-State Halting Machine (Busy Beaver-style quick terminator)
	haltingMachine := EncodeMachine(map[int][2]Transition{
		1: {
			{WriteSymbol: '1', MoveDir: 1, NextState: 2},
			{WriteSymbol: '1', MoveDir: -1, NextState: 2},
		},
		2: {
			{WriteSymbol: '1', MoveDir: -1, NextState: 1},
			{WriteSymbol: '1', MoveDir: 1, NextState: 0}, // Halts when reading '1' in State 2
		},
	})

	fmt.Println("=== 1. PARSING MULTI-STATE PROOF SEARCHERS IN G^H ===")
	godelCandidate := fmt.Sprintf("%s#%s", proofSearcher, "101")
	haltingCandidate := fmt.Sprintf("%s#%s", haltingMachine, "000")

	valid1, reason1 := grammar.Parse(godelCandidate)
	fmt.Printf("Gödel Proof Searcher | Valid: %-5v | %s\n", valid1, reason1)
	fmt.Printf("  -> Exact String: %s\n\n", godelCandidate)

	valid2, reason2 := grammar.Parse(haltingCandidate)
	fmt.Printf("Halting Machine      | Valid: %-5v | %s\n\n", valid2, reason2)

	fmt.Println("=== 2. HYPERCOMPUTATIONAL RANDOM GENERATION (3-STATE MACHINES) ===")
	generated, err := grammar.Generate(3, 100)
	if err != nil {
		fmt.Println("Error:", err)
	} else {
		fmt.Printf("Generated valid 3-state string:\n%s\n", generated)
	}
}
