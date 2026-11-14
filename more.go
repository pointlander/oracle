// Copyright 2026 The Oracle Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
)

const moreTextCount = 10

// readFirstTexts returns the first n regular files in the tar stored inside
// a zip archive, in archive order, joined by newlines.
func readFirstTexts(path string, n int) ([]byte, []string, error) {
	if n < 1 {
		return nil, nil, fmt.Errorf("more texts must be positive")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	zr, err := zip.NewReader(f, st.Size())
	if err != nil {
		return nil, nil, err
	}
	var member *zip.File
	for _, zf := range zr.File {
		if strings.HasSuffix(zf.Name, ".tar") {
			member = zf
			break
		}
	}
	if member == nil {
		return nil, nil, fmt.Errorf("%s contains no tar member", path)
	}
	rc, err := member.Open()
	if err != nil {
		return nil, nil, err
	}
	defer rc.Close()
	tr := tar.NewReader(rc)
	var buf bytes.Buffer
	names := make([]string, 0, n)
	for len(names) < n {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		if hdr.Typeflag != tar.TypeReg && hdr.Typeflag != tar.TypeRegA {
			continue
		}
		if len(names) > 0 && (buf.Len() == 0 || buf.Bytes()[buf.Len()-1] != '\n') {
			buf.WriteByte('\n')
		}
		if _, err := io.Copy(&buf, tr); err != nil {
			return nil, nil, err
		}
		names = append(names, hdr.Name)
	}
	if len(names) < n {
		return nil, nil, fmt.Errorf("%s has %d texts, want %d", path, len(names), n)
	}
	return buf.Bytes(), names, nil
}
