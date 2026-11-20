// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bufio"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

// irisFeatures is the number of measurements on each flower.
const irisFeatures = 4

// irisTapeCells is the inscribed tape. Cell 0 is the bias. Each later cell
// is one measurement. The program takes one step per cell.
const irisTapeCells = irisFeatures + 1

// irisTau is the width of the text-machine symbol softmax. It matches the
// default -temp of a spelled string.
const irisTau = 8

// irisByteCenter and irisByteGap place each class on a text byte. The bytes
// sit in the middle of 0..255, sixteen apart, so the Gaussian is away from
// either end and the decision boundaries have room between them.
const irisByteCenter = 128
const irisByteGap = 16

// irisTestFrac is the fraction of each class held out. The split is
// stratified and seeded.
const irisTestFrac = 0.2

type irisRow struct {
	x [irisFeatures]float64
	y int
}

type irisConfig struct {
	Path  string
	Iters int
	Temp  float64
	Seed  int64
	Log   func(string)
}

func (cfg irisConfig) logf(format string, args ...any) {
	if cfg.Log != nil {
		cfg.Log(fmt.Sprintf(format, args...))
	}
}

// loadIris reads a UCI iris file, or iris.data inside a zip.
func loadIris(path string) ([]irisRow, []string, error) {
	var data []byte
	var err error
	if strings.HasSuffix(strings.ToLower(path), ".zip") {
		data, err = irisFromZip(path)
	} else {
		data, err = os.ReadFile(path)
	}
	if err != nil {
		return nil, nil, err
	}
	return parseIris(string(data))
}

func irisFromZip(path string) ([]byte, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	var found *zip.File
	for _, f := range r.File {
		name := f.Name
		if name == "iris.data" || strings.HasSuffix(name, "/iris.data") {
			found = f
			break
		}
	}
	if found == nil {
		return nil, fmt.Errorf("iris.data is not in %s", path)
	}
	rc, err := found.Open()
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	return io.ReadAll(rc)
}

// parseIris reads rows of four measurements and a class name. Blank lines
// are skipped. Class numbers follow the order of first appearance.
func parseIris(text string) ([]irisRow, []string, error) {
	var rows []irisRow
	var names []string
	index := map[string]int{}
	lineNo := 0
	s := bufio.NewScanner(strings.NewReader(text))
	for s.Scan() {
		lineNo++
		line := strings.TrimSpace(s.Text())
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) != irisFeatures+1 {
			return nil, nil, fmt.Errorf("line %d has %d fields", lineNo, len(parts))
		}
		var row irisRow
		for i := 0; i < irisFeatures; i++ {
			v, err := strconv.ParseFloat(strings.TrimSpace(parts[i]), 64)
			if err != nil {
				return nil, nil, fmt.Errorf("line %d feature %d: %w", lineNo, i+1, err)
			}
			row.x[i] = v
		}
		name := strings.TrimSpace(parts[irisFeatures])
		if name == "" {
			return nil, nil, fmt.Errorf("line %d has an empty class", lineNo)
		}
		y, ok := index[name]
		if !ok {
			y = len(names)
			index[name] = y
			names = append(names, name)
		}
		row.y = y
		rows = append(rows, row)
	}
	if err := s.Err(); err != nil {
		return nil, nil, err
	}
	if len(rows) == 0 || len(names) < 2 {
		return nil, nil, fmt.Errorf("iris data needs at least two classes")
	}
	return rows, names, nil
}

// splitIris holds out the same fraction of every class. The shuffle is
// seeded, and the held-out rows are the front of each shuffled class.
func splitIris(rows []irisRow, seed int64) (train, test []irisRow, err error) {
	var order []int
	by := map[int][]irisRow{}
	for _, r := range rows {
		if _, ok := by[r.y]; !ok {
			order = append(order, r.y)
		}
		by[r.y] = append(by[r.y], r)
	}
	rng := rand.New(rand.NewSource(seed))
	for _, y := range order {
		g := by[y]
		rng.Shuffle(len(g), func(i, j int) { g[i], g[j] = g[j], g[i] })
		nTest := int(math.Round(irisTestFrac * float64(len(g))))
		if len(g) > 1 && nTest < 1 {
			nTest = 1
		}
		if nTest >= len(g) {
			nTest = len(g) - 1
		}
		if nTest < 1 {
			return nil, nil, fmt.Errorf("class %d has no held-out row", y)
		}
		test = append(test, g[:nTest]...)
		train = append(train, g[nTest:]...)
	}
	if len(train) == 0 || len(test) == 0 {
		return nil, nil, fmt.Errorf("iris split is empty")
	}
	return train, test, nil
}

// standardizeIris centers each feature on the training mean and scales it
// by the training population deviation. The test rows use those same values.
func standardizeIris(train, test []irisRow) {
	var mean [irisFeatures]float64
	for _, r := range train {
		for i := 0; i < irisFeatures; i++ {
			mean[i] += r.x[i]
		}
	}
	inv := 1 / float64(len(train))
	for i := 0; i < irisFeatures; i++ {
		mean[i] *= inv
	}
	var dev [irisFeatures]float64
	for _, r := range train {
		for i := 0; i < irisFeatures; i++ {
			d := r.x[i] - mean[i]
			dev[i] += d * d
		}
	}
	var scale [irisFeatures]float64
	for i := 0; i < irisFeatures; i++ {
		s := math.Sqrt(dev[i] * inv)
		if s == 0 {
			s = 1
		}
		scale[i] = s
	}
	apply := func(rows []irisRow) {
		for i := range rows {
			for j := 0; j < irisFeatures; j++ {
				rows[i].x[j] = (rows[i].x[j] - mean[j]) / scale[j]
			}
		}
	}
	apply(train)
	apply(test)
}

// irisInscribe writes the bias and the measurements on the tape. Cell 0 is
// the bias. Each later cell is one measurement, in order.
func irisInscribe(bias float64, x [irisFeatures]float64) []float64 {
	tape := make([]float64, irisTapeCells)
	tape[0] = bias
	for i := 0; i < irisFeatures; i++ {
		tape[i+1] = x[i]
	}
	return tape
}

// irisProgram learns the combining writes. The bias and the measurements are
// already on the tape. The head starts on the first measurement and moves
// right. A blank cell, or a mean that rounds to byte 0, selects write 0.
// Any other byte selects write 1. That choice is constant, so it has no
// partial. The step adds the learned write for that byte. The measurement
// stays on the cell. The score is the sum of the tape after those writes.
func irisProgram(bias float64, w0, w1 []float64, x [irisFeatures]float64) (sum float64, rule []int) {
	tape := irisInscribe(bias, x)
	rule = make([]int, irisFeatures)
	head := 1
	for t := 0; t < irisFeatures; t++ {
		pos := wrapCell(head, len(tape))
		r := 0
		if nearestByte(tape[pos]) != 0 {
			r = 1
		}
		rule[t] = r
		w := w0[t]
		if r == 1 {
			w = w1[t]
		}
		tape[pos] += w
		head = pos + 1
	}
	for _, s := range tape {
		sum += s
	}
	return sum, rule
}

// irisClassByte is the text symbol for a class. Class 0 sits at or just
// below byte 128 and each next class is irisByteGap higher.
func irisClassByte(y, nClass int) int {
	if nClass < 2 {
		nClass = 2
	}
	if y < 0 {
		y = 0
	}
	if y >= nClass {
		y = nClass - 1
	}
	return irisByteCenter + (y-(nClass-1)/2)*irisByteGap
}

// irisClassOf is the class whose text byte is nearest the output. An exact
// tie picks the lower class.
func irisClassOf(score float64, nClass int) int {
	best := 0
	bestD := math.Inf(1)
	for c := 0; c < nClass; c++ {
		d := float64(irisClassByte(c, nClass)) - score
		d *= d
		if d < bestD {
			bestD = d
			best = c
		}
	}
	return best
}

// classLoss is the cross-entropy of a class symbol under a Gaussian softmax.
// Class symbols sit at 0, 1, .... The partial with respect to the output
// score is (E[c] - y) / τ^2.
func classLoss(mu, tau float64, y, nClass int) (nll, dmu float64) {
	scale := 1 / (2 * tau * tau)
	z := make([]float64, nClass)
	maxZ := math.Inf(-1)
	for c := 0; c < nClass; c++ {
		d := float64(c) - mu
		z[c] = -d * d * scale
		if z[c] > maxZ {
			maxZ = z[c]
		}
	}
	sum := 0.0
	moment := 0.0
	for c := 0; c < nClass; c++ {
		e := math.Exp(z[c] - maxZ)
		sum += e
		moment += e * float64(c)
	}
	expect := moment / sum
	nll = -z[y] + maxZ + math.Log(sum)
	dmu = (expect - float64(y)) / (tau * tau)
	return nll, dmu
}

// classOf is the class symbol nearest the output score. That is the mode of
// the Gaussian softmax when every class shares τ. An exact tie picks the
// lower symbol.
func classOf(mu float64, nClass int) int {
	best := 0
	bestD := math.Inf(1)
	for c := 0; c < nClass; c++ {
		d := float64(c) - mu
		d *= d
		if d < bestD {
			bestD = d
			best = c
		}
	}
	return best
}

// irisGrad is the mean symbol cross-entropy of the tape sum, the partial of
// that mean for the bias, and the partial for each learned write. The rule
// that did not fire has partial 0.
func irisGrad(bias float64, w0, w1 []float64, rows []irisRow, tau float64, nClass int) (gb float64, g0, g1 []float64, loss float64) {
	g0 = make([]float64, irisFeatures)
	g1 = make([]float64, irisFeatures)
	n := float64(len(rows))
	for _, r := range rows {
		sum, rule := irisProgram(bias, w0, w1, r.x)
		nll, dmu := symbolLoss(sum, tau, irisClassByte(r.y, nClass))
		loss += nll
		gb += dmu
		for t, ruleT := range rule {
			if ruleT == 0 {
				g0[t] += dmu
			} else {
				g1[t] += dmu
			}
		}
	}
	loss /= n
	gb /= n
	for t := 0; t < irisFeatures; t++ {
		g0[t] /= n
		g1[t] /= n
	}
	return gb, g0, g1, loss
}

func irisCorrect(rows []irisRow, bias float64, w0, w1 []float64, nClass int) int {
	n := 0
	for _, r := range rows {
		sum, _ := irisProgram(bias, w0, w1, r.x)
		if irisClassOf(sum, nClass) == r.y {
			n++
		}
	}
	return n
}

// fitIris writes the bias and the measurements on the tape, then learns the
// write each step adds. The text trainer's full step is τ^2 times the
// partial, which moves one write onto its byte. Flowers want different
// bytes, and several writes land in one sum, so this step is two fifths
// of that. Weights start at 0.
func fitIris(train, test []irisRow, nClass, iters int, tau float64, logf func(string, ...any)) (bias float64, w0, w1 []float64, err error) {
	if len(train) == 0 || nClass < 2 || tau <= 0 || iters < 0 {
		return 0, nil, nil, fmt.Errorf("iris fit needs training rows, two classes, and a positive temperature")
	}
	w0 = make([]float64, irisFeatures)
	w1 = make([]float64, irisFeatures)
	lr := 0.4 * tau * tau
	for iter := 0; iter <= iters; iter++ {
		gb, g0, g1, loss := irisGrad(bias, w0, w1, train, tau, nClass)
		tr := irisCorrect(train, bias, w0, w1, nClass)
		te := irisCorrect(test, bias, w0, w1, nClass)
		if logf != nil && (iter == 0 || iter == iters || iter%8 == 0) {
			logf("iter %d  loss %.6f  train %d/%d  test %d/%d", iter, loss, tr, len(train), te, len(test))
		}
		if iter == iters {
			break
		}
		bias -= lr * gb
		for t := 0; t < irisFeatures; t++ {
			w0[t] -= lr * g0[t]
			w1[t] -= lr * g1[t]
		}
	}
	return bias, w0, w1, nil
}

func learnIris(cfg irisConfig) error {
	if cfg.Iters <= 0 {
		cfg.Iters = 32
	}
	rows, names, err := loadIris(cfg.Path)
	if err != nil {
		return err
	}
	train, test, err := splitIris(rows, cfg.Seed)
	if err != nil {
		return err
	}
	standardizeIris(train, test)
	if cfg.Temp <= 0 {
		cfg.Temp = irisTau
	}
	cfg.logf("gaussian machine classifies iris")
	cfg.logf("  file %s", cfg.Path)
	cfg.logf("  %d flowers, %d features, %d classes", len(rows), irisFeatures, len(names))
	cfg.logf("  train %d, test %d, seed %d", len(train), len(test), cfg.Seed)
	cfg.logf("  features are centered and scaled on the training flowers")
	cfg.logf("  the bias and the measurements are written on the tape")
	cfg.logf("  cell 0 is the bias and the next cells are the measurements")
	cfg.logf("  the program learns a write for a zero byte and a write for any other byte")
	cfg.logf("  the head starts on the first measurement and moves right")
	cfg.logf("  each step reads that cell and adds the write for that byte")
	cfg.logf("  a blank cell or a zero byte reads 0, and any other byte reads 1")
	cfg.logf("  the output is the sum of the tape after those writes")
	cfg.logf("  loss is the text machine's Gaussian softmax of that sum")
	cfg.logf("  class bytes are %d apart around byte %d", irisByteGap, irisByteCenter)
	cfg.logf("  the class is the byte nearest that sum")
	cfg.logf("  the write step is two fifths of the temperature squared")
	bias, w0, w1, err := fitIris(train, test, len(names), cfg.Iters, cfg.Temp, cfg.logf)
	if err != nil {
		return err
	}
	cfg.logf("  bias %.3f", bias)
	cfg.logf("  write 0  %.3f %.3f %.3f %.3f", w0[0], w0[1], w0[2], w0[3])
	cfg.logf("  write 1  %.3f %.3f %.3f %.3f", w1[0], w1[1], w1[2], w1[3])
	for y, name := range names {
		var have, hit int
		for _, r := range test {
			if r.y != y {
				continue
			}
			have++
			sum, _ := irisProgram(bias, w0, w1, r.x)
			if irisClassOf(sum, len(names)) == y {
				hit++
			}
		}
		cfg.logf("  test %s  %d/%d  byte %d", name, hit, have, irisClassByte(y, len(names)))
	}
	return nil
}
