// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/gob"
	"errors"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"
)

// loadCheckpoint rebuilds the architecture stored in a training checkpoint
// and copies the saved weights into it.
func loadCheckpoint(path string) (*OracleTransformer, int, float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()
	var ck checkpoint
	if err := gob.NewDecoder(f).Decode(&ck); err != nil {
		return nil, 0, 0, err
	}
	model, err := NewModel(ck.Config, rand.New(rand.NewSource(1)))
	if err != nil {
		return nil, 0, 0, err
	}
	ps := model.Params()
	if len(ps) != len(ck.Params) {
		return nil, 0, 0, fmt.Errorf("checkpoint has %d tensors, model has %d", len(ck.Params), len(ps))
	}
	for i, p := range ps {
		if len(p.Data) != len(ck.Params[i]) {
			return nil, 0, 0, fmt.Errorf("param %d: checkpoint %d values, model %d", i, len(ck.Params[i]), len(p.Data))
		}
		copy(p.Data, ck.Params[i])
	}
	return model, ck.Step, ck.ValCE, nil
}

func inferFromArgs(args []string) error {
	fs := flag.NewFlagSet("infer", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: oracle infer [flags]\n\nContinue a prompt by sampling the next-byte distribution at a temperature.\nTemperature 1 is the model's softmax. Lower values sharpen it. 0 is greedy.\n\n-mcts draws -sims rollouts from that softmax. Each rollout is scored by the sum of its log probabilities.\nThe rollout with the highest sum is the continuation.\n\n")
		fs.PrintDefaults()
	}
	path := fs.String("checkpoint", "checkpoints/best.gob", "trained weights")
	prompt := fs.String("prompt", "The quality of mercy ", "text to continue")
	n := fs.Int("n", 200, "bytes to generate")
	temp := fs.Float64("temp", 0.8, "softmax temperature")
	seed := fs.Int64("seed", -1, "RNG seed; negative uses the current time")
	mcts := fs.Bool("mcts", false, "sample several rollouts and keep the highest log-probability sum")
	sims := fs.Int("sims", 8, "number of MCTS rollouts")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if *n < 1 {
		return fmt.Errorf("n must be positive")
	}
	if *prompt == "" {
		return fmt.Errorf("prompt is empty")
	}
	model, step, val, err := loadCheckpoint(*path)
	if err != nil {
		return err
	}
	s := *seed
	if s < 0 {
		s = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(s))
	var sample []byte
	if *mcts {
		if *sims < 1 {
			return fmt.Errorf("sims must be positive")
		}
		sample = model.GenerateMCTS([]byte(*prompt), *n, MCTSConfig{
			Sims: *sims, Temp: *temp,
		}, rng)
		fmt.Fprintf(os.Stderr, "checkpoint %s  step %d  val_ce %.4f  temp %.3f  seed %d  mcts sims %d\n", *path, step, val, *temp, s, *sims)
	} else {
		sample = model.Generate([]byte(*prompt), *n, *temp, rng)
		fmt.Fprintf(os.Stderr, "checkpoint %s  step %d  val_ce %.4f  temp %.3f  seed %d\n", *path, step, val, *temp, s)
	}
	if _, err := os.Stdout.Write([]byte(*prompt)); err != nil {
		return err
	}
	if _, err := os.Stdout.Write(sample); err != nil {
		return err
	}
	if len(sample) == 0 || sample[len(sample)-1] != '\n' {
		if _, err := os.Stdout.Write([]byte{'\n'}); err != nil {
			return err
		}
	}
	return nil
}
