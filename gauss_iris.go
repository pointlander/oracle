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

// irisFeatures is the number of measurements on each flower. The machine
// scans one tape cell per measurement.
const irisFeatures = 4

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

// scoreIris scans the four feature cells from the left. State i writes its
// weight times that feature onto the output cell and moves right. A final
// write adds the bias. The score is that sum, and it is read as a text byte.
func scoreIris(w [irisFeatures]float64, b float64, x [irisFeatures]float64) float64 {
	out := b
	for i := range x {
		out += w[i] * x[i]
	}
	return out
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

// irisGrad is the mean text-symbol cross-entropy and the partial of that
// mean with respect to each write weight and the bias. The target is the
// class byte, not the class index.
func irisGrad(w [irisFeatures]float64, b float64, rows []irisRow, tau float64, nClass int) (gw [irisFeatures]float64, gb, loss float64) {
	n := float64(len(rows))
	for _, r := range rows {
		mu := scoreIris(w, b, r.x)
		nll, dmu := symbolLoss(mu, tau, irisClassByte(r.y, nClass))
		loss += nll
		gb += dmu
		for i := 0; i < irisFeatures; i++ {
			gw[i] += dmu * r.x[i]
		}
	}
	loss /= n
	gb /= n
	for i := range gw {
		gw[i] /= n
	}
	return gw, gb, loss
}

func irisCorrect(rows []irisRow, w [irisFeatures]float64, b float64, nClass int) int {
	n := 0
	for _, r := range rows {
		if irisClassOf(scoreIris(w, b, r.x), nClass) == r.y {
			n++
		}
	}
	return n
}

// fitIris follows the partial of the text-symbol cross-entropy through the
// scan. The text trainer's full step is τ^2 times the partial, which moves
// one write onto its byte. Flowers in one scan want different bytes, and
// that full step overshoots, so this step is half of that. Weights start at 0.
func fitIris(train, test []irisRow, nClass, iters int, tau float64, logf func(string, ...any)) (w [irisFeatures]float64, b float64, err error) {
	if len(train) == 0 || nClass < 2 || tau <= 0 || iters < 0 {
		return w, b, fmt.Errorf("iris fit needs training rows, two classes, and a positive temperature")
	}
	lr := 0.5 * tau * tau
	for iter := 0; iter <= iters; iter++ {
		gw, gb, loss := irisGrad(w, b, train, tau, nClass)
		tr := irisCorrect(train, w, b, nClass)
		te := irisCorrect(test, w, b, nClass)
		if logf != nil && (iter == 0 || iter == iters || iter%8 == 0) {
			logf("iter %d  loss %.6f  train %d/%d  test %d/%d", iter, loss, tr, len(train), te, len(test))
		}
		if iter == iters {
			break
		}
		b -= lr * gb
		for i := range w {
			w[i] -= lr * gw[i]
		}
	}
	return w, b, nil
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
	cfg.logf("  the head scans the four feature cells from the left")
	cfg.logf("  each state writes its weight times that feature onto the output cell")
	cfg.logf("  the last write is a bias, and the output cell sums those writes")
	cfg.logf("  loss is the text machine's Gaussian softmax over byte codes 0..255")
	cfg.logf("  class bytes are %d apart around byte %d", irisByteGap, irisByteCenter)
	cfg.logf("  the class is the byte nearest that sum")
	cfg.logf("  the write step is half the temperature squared")
	w, b, err := fitIris(train, test, len(names), cfg.Iters, cfg.Temp, cfg.logf)
	if err != nil {
		return err
	}
	cfg.logf("  weights %.3f %.3f %.3f %.3f  bias %.3f", w[0], w[1], w[2], w[3], b)
	for y, name := range names {
		var have, hit int
		for _, r := range test {
			if r.y != y {
				continue
			}
			have++
			if irisClassOf(scoreIris(w, b, r.x), len(names)) == y {
				hit++
			}
		}
		cfg.logf("  test %s  %d/%d  byte %d", name, hit, have, irisClassByte(y, len(names)))
	}
	return nil
}
