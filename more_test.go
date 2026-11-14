// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func writeTarZip(t *testing.T, path string, order []string, bodies map[string]string) {
	t.Helper()
	var tarBuf bytes.Buffer
	tw := tar.NewWriter(&tarBuf)
	for _, name := range order {
		body := []byte(bodies[name])
		hdr := &tar.Header{Name: name, Mode: 0o644, Size: int64(len(body))}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write(body); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	var zipBuf bytes.Buffer
	zw := zip.NewWriter(&zipBuf)
	w, err := zw.Create("txt-files.tar")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write(tarBuf.Bytes()); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, zipBuf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestReadFirstTextsStopsAtTen(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "txt-files.tar.zip")
	order := make([]string, 12)
	bodies := make(map[string]string, 12)
	for i := range order {
		name := filepath.ToSlash(filepath.Join("cache", "epub", string(rune('a'+i)), "t.txt"))
		order[i] = name
		bodies[name] = string(rune('A'+i)) + "\n"
	}
	writeTarZip(t, path, order, bodies)
	got, names, err := readFirstTexts(path, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 10 {
		t.Fatalf("names %d", len(names))
	}
	for i, name := range names {
		if name != order[i] {
			t.Fatalf("name %d %s", i, name)
		}
	}
	want := []byte("A\nB\nC\nD\nE\nF\nG\nH\nI\nJ\n")
	if !bytes.Equal(got, want) {
		t.Fatalf("got %q", got)
	}
}

func TestTrainMoreKeepsHoldout(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "txt-files.tar.zip")
	writeTarZip(t, path, []string{"one.txt", "two.txt", "three.txt"}, map[string]string{
		"one.txt":   "XXXX",
		"two.txt":   "YYYY",
		"three.txt": "ZZZZ",
	})
	raw := bytes.Repeat([]byte("abcdefg"), 40)
	cfg := defaultConfig()
	cfg.Ctx = 8
	cfg.DModel = 16
	cfg.Layers = 1
	cfg.Heads = 4
	cfg.OracleInput = 4
	cfg.MaxSteps = 8
	res, err := Train(TrainConfig{
		Data:      raw,
		MorePath:  path,
		MoreN:     2,
		Steps:     1,
		Batch:     2,
		Ctx:       cfg.Ctx,
		LR:        1e-3,
		Warmup:    0,
		EvalEvery: 1,
		Seed:      1,
		SampleN:   0,
		Model:     cfg,
		Log:       func(string) {},
	})
	if err != nil {
		t.Fatal(err)
	}
	trainN := (len(raw) * 9) / 10
	// A newline separates the holdout prefix from "XXXX\nYYYY".
	if res.TrainBytes != trainN+1+len("XXXX\nYYYY") {
		t.Fatalf("train bytes %d", res.TrainBytes)
	}
	if res.ValBytes != len(raw)-trainN {
		t.Fatalf("val bytes %d", res.ValBytes)
	}
}

func TestRealArchiveFirstTen(t *testing.T) {
	if _, err := os.Stat("txt-files.tar.zip"); err != nil {
		t.Skip(err)
	}
	_, names, err := readFirstTexts("txt-files.tar.zip", moreTextCount)
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 10 || names[0] != "cache/epub/10000/pg10000.txt" || names[9] != "cache/epub/10009/pg10009.txt" {
		t.Fatalf("names %v", names)
	}
}
