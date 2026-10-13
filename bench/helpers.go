package main

import (
	"encoding/csv"
	"fmt"
	"math/rand/v2"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
)

func parseSweep(sweep string, fallback int) ([]int, error) {
	if strings.TrimSpace(sweep) == "" {
		return []int{fallback}, nil
	}
	levels := []int{}
	for field := range strings.SplitSeq(sweep, ",") {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		n, err := strconv.Atoi(field)
		if err != nil {
			return nil, fmt.Errorf("%q is not a number", field)
		}
		if n <= 0 {
			return nil, fmt.Errorf("concurrency must be greater than 0, got %d", n)
		}
		levels = append(levels, n)
	}
	// catches a value that is non-empty but has no levels in it, e.g. ",,,"
	if len(levels) == 0 {
		return nil, fmt.Errorf("no levels found")
	}
	return levels, nil
}

func deriveStatsUrl(target string) (string, error) {
	u, err := url.Parse(target)
	if err != nil {
		return "", err
	}
	if u.Host == "" {
		return "", fmt.Errorf("%q has no host", target)
	}
	u.Path = "/stats"
	u.RawQuery = ""
	u.Fragment = ""
	return u.String(), nil
}

func csvHeader(v any) []string {
	header := []string{}
	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	for field := range t.Fields() {
		name := field.Tag.Get("csv")
		if name == "" || name == "-" {
			continue
		}
		header = append(header, name)
	}
	return header
}

func buildRecord(r any) []string {
	record := []string{}
	v := reflect.ValueOf(r)
	if v.Kind() == reflect.Pointer {
		v = v.Elem()
	}
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		if name := t.Field(i).Tag.Get("csv"); name == "" || name == "-" {
			continue
		}
		value := v.Field(i)

		var str string
		switch value.Kind() {
		case reflect.String:
			str = value.String()
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			str = strconv.Itoa(int(value.Int()))
		case reflect.Float32, reflect.Float64:
			str = fmt.Sprintf("%f", value.Float())
		case reflect.Bool:
			str = fmt.Sprintf("%t", value.Bool())
		default:
			// Interface() panics on unexported fields, and every field of Report
			// is unexported -- so anything not handled above must not reach it
			if !value.CanInterface() {
				str = fmt.Sprintf("<unsupported %s>", value.Kind())
				break
			}
			str = fmt.Sprint(value.Interface())
		}
		record = append(record, str)
	}
	return record
}

func writeToCSV(filename string, header []string, records ...[]string) error {
	// MkdirAll, not Mkdir: it is idempotent, so levels 2..n of a sweep don't
	// fail with EEXIST once level 1 has created the directory
	if dir := filepath.Dir(filename); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}

	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	// size rather than existence: O_CREATE means the file is always there by
	// now, so an empty one is the only kind that still needs a header
	info, err := file.Stat()
	if err != nil {
		return err
	}

	writer := csv.NewWriter(file)
	defer writer.Flush()

	if info.Size() == 0 {
		if err := writer.Write(header); err != nil {
			return err
		}
	}
	for _, record := range records {
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	writer.Flush()
	return writer.Error()
}

type mixEntry struct {
	tokens int
	weight float64
}

func parseMix(s string) ([]mixEntry, error) {
	var mix []mixEntry
	for field := range strings.SplitSeq(s, ",") {
		tok, w, ok := strings.Cut(strings.TrimSpace(field), ":")
		if !ok {
			return nil, fmt.Errorf("%q: want tokens:weight", field)
		}
		t, err := strconv.Atoi(tok)
		if err != nil || t <= 0 {
			return nil, fmt.Errorf("%q: bad token count", field)
		}
		wf, err := strconv.ParseFloat(w, 64)
		if err != nil || wf <= 0 {
			return nil, fmt.Errorf("%q: bad weight", field)
		}
		mix = append(mix, mixEntry{t, wf})
	}
	if len(mix) == 0 {
		return nil, fmt.Errorf("empty mix")
	}
	return mix, nil
}

func buildWorkload(mix []mixEntry, n int, seed uint64) []int {
	var total float64
	for _, m := range mix {
		total += m.weight
	}
	r := rand.New(rand.NewPCG(seed, 0))
	out := make([]int, n)
	for i := range out {
		x := r.Float64() * total
		out[i] = mix[len(mix)-1].tokens
		for _, m := range mix {
			if x < m.weight {
				out[i] = m.tokens
				break
			}
			x -= m.weight
		}
	}
	return out
}
