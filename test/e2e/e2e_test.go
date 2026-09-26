package e2e

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

const engineSlots = 2

var (
	root       string
	logDir     string
	baseURL    string
	workerPort int
	workerCmd  *exec.Cmd
)

type completion struct {
	RequestId       string `json:"request_id"`
	GeneratedText   string `json:"generated_text"`
	TokensGenerated int    `json:"tokens_generated"`
	Error           string `json:"error"`
}

func TestMain(m *testing.M) { os.Exit(run(m)) }

func run(m *testing.M) int {
	root, _ = filepath.Abs("../..")
	logDir = filepath.Join(os.TempDir(), "inference-e2e")
	os.MkdirAll(logDir, 0o755)
	log.Printf("e2e logs: %s", logDir)

	bin := filepath.Join(logDir, "server")
	build := exec.Command("go", "build", "-race", "-o", bin, "./cmd/server")
	build.Dir, build.Stdout, build.Stderr = root, os.Stderr, os.Stderr
	if err := build.Run(); err != nil {
		log.Printf("build server: %v", err)
		return 1
	}

	workerPort = freePort()
	if err := startWorker(); err != nil {
		log.Printf("start worker: %v", err)
		return 1
	}
	defer killWorker()

	httpPort := freePort()
	serverLog := filepath.Join(logDir, "server.log")
	f, _ := os.Create(serverLog)
	srv := exec.Command(bin,
		"-httpAddr", fmt.Sprintf("127.0.0.1:%d", httpPort),
		"-workerAddr", fmt.Sprintf("127.0.0.1:%d", workerPort),
		"-continuousBatching",
		"-engineSlots", strconv.Itoa(engineSlots),
		"-requestTimeout", "60s",
		"-queueTimeout", "30s",
	)
	srv.Stdout = f
	srv.Stderr = f
	if err := srv.Start(); err != nil {
		log.Printf("start server: %v", err)
		return 1
	}
	baseURL = fmt.Sprintf("http://127.0.0.1:%d", httpPort)

	if err := waitHealthy(90 * time.Second); err != nil {
		log.Printf("%v (see %s)", err, logDir)
		srv.Process.Kill()
		return 1
	}

	code := m.Run()

	srv.Process.Signal(os.Interrupt)
	srv.Wait()
	f.Close()
	if b, _ := os.ReadFile(serverLog); strings.Contains(string(b), "DATA RACE") {
		log.Printf("data race detected, see %s", serverLog)
		return 1
	}
	return code
}

func startWorker() error {
	python := os.Getenv("WORKER_PYTHON")
	if python == "" {
		python = filepath.Join(root, "cmd/worker/venv/bin/python")
	}
	f, err := os.OpenFile(filepath.Join(logDir, "worker.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	// Assign, don't declare: killWorker reads the package-level workerCmd
	workerCmd = exec.Command(python, "worker.py",
		"--port", strconv.Itoa(workerPort),
		"--engine-slots", strconv.Itoa(engineSlots),
		"--engine-per-seq-ctx", "512",
	)
	workerCmd.Dir = filepath.Join(root, "cmd/worker")
	workerCmd.Stdout = f
	workerCmd.Stderr = f
	// Homebrew's macOS python is a launcher that runs the real interpreter as
	// a child, so killing workerCmd's pid alone orphans the actual worker.
	// Give it its own process group and kill the whole group instead.
	workerCmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	return workerCmd.Start()
}

func killWorker() {
	if workerCmd != nil && workerCmd.Process != nil {
		syscall.Kill(-workerCmd.Process.Pid, syscall.SIGKILL) // negative pid = process group
		workerCmd.Wait()
		workerCmd = nil
	}
}

func freePort() int {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func waitHealthy(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if resp, err := http.Get(baseURL + "/healthz"); err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("server not healthy after %s", timeout)
}

func complete(ctx context.Context, body string) (int, completion, error) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/completions", strings.NewReader(body))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, completion{}, err
	}
	defer resp.Body.Close()
	var c completion
	json.NewDecoder(resp.Body).Decode(&c)
	return resp.StatusCode, c, nil
}

func fanOut(t *testing.T, n int, body string, timeout time.Duration) []int {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	codes := make([]int, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			code, _, err := complete(ctx, body)
			if err != nil {
				t.Errorf("request %d: %v", i, err)
			}
			codes[i] = code
		}()
	}
	wg.Wait()
	return codes
}

const short = `{"prompt":"The capital of France is","max_tokens":8}`

// long must actually run long: an open-ended prompt like "write a story" lets
// TinyLlama hit EOS at token 0. Greedy counting never ends on its own, so this
// reliably generates all 480 tokens (~4.5s solo on an M-series Air).
const long = `{"prompt":"1, 2, 3, 4, 5, 6, 7, 8,","max_tokens":480}`

func TestHappyPath(t *testing.T) {
	code, c, err := complete(context.Background(), short)
	if err != nil || code != http.StatusOK {
		t.Fatalf("code=%d err=%v body=%+v", code, err, c)
	}
	if c.GeneratedText == "" || c.TokensGenerated == 0 || c.RequestId == "" {
		t.Fatalf("empty completion: %+v", c)
	}
}

func TestValidation(t *testing.T) {
	for _, body := range []string{
		`{"prompt":"","max_tokens":8}`,
		`{"prompt":"hi","max_tokens":0}`,
		`{"prompt":"hi","max_tokens":8,"temperature":3}`,
		`{"prompt":"hi","max_tokens":8,"priority":"urgent"}`,
		`not json`,
	} {
		code, _, _ := complete(context.Background(), body)
		if code != http.StatusBadRequest {
			t.Errorf("%s: got %d, want 400", body, code)
		}
	}
}

func TestConcurrencyAboveSlots(t *testing.T) {
	for i, code := range fanOut(t, engineSlots*10, short, 60*time.Second) {
		if code != http.StatusOK {
			t.Errorf("request %d: status %d", i, code)
		}
	}
}

// Fill every slot with a long generation and abandon them. If Cancel reaches
// the worker, a short request gets a slot right away; if not, it waits for
// the 480-token generations to finish
// Calibrate: time one `long` request by hand; the threshold must be well below it.
func TestClientDisconnectFreesSlot(t *testing.T) {
	for range engineSlots {
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 1500)
			defer cancel()
			complete(ctx, long)
		}()
	}
	time.Sleep(2500 * time.Millisecond)

	start := time.Now()
	code, _, err := complete(context.Background(), short)
	elapsed := time.Since(start)
	if err != nil || code != http.StatusOK {
		t.Fatalf("code=%d err=%v", code, err)
	}
	if elapsed > 3*time.Second {
		t.Fatalf("short request took %s; abandoned requests were probably not cancelled", elapsed)
	}
}

// Engine Rejected events must return the sem token. Reject more times than
// there are slots, then prove full capacity is still there.
func TestRejectionDoesNotLeakSlots(t *testing.T) {
	tooLong := `{"prompt":"hi","max_tokens":600}` // passes Go validation, exceeds per_seq_ctx=512
	for range engineSlots + 1 {
		code, c, _ := complete(context.Background(), tooLong)
		if code != http.StatusInternalServerError || !strings.Contains(c.Error, "per-sequence context") {
			t.Fatalf("got %d %q", code, c.Error)
		}
	}
	for i, code := range fanOut(t, engineSlots*2, short, 20*time.Second) {
		if code != http.StatusOK {
			t.Errorf("request %d after rejections: status %d (sem leak?)", i, code)
		}
	}
}

func TestWorkerCrashAndReconnect(t *testing.T) {
	type result struct {
		code int
		c    completion
	}
	inflight := make(chan result, 1)
	go func() {
		code, c, _ := complete(context.Background(), long)
		inflight <- result{code, c}
	}()
	time.Sleep(1500 * time.Millisecond)
	killWorker()

	// failAll must answer in-flight requests promptly, not leave them hanging
	select {
	case r := <-inflight:
		if r.code != http.StatusInternalServerError || !strings.Contains(r.c.Error, "stream failed") {
			// A 200 here means it finished before the kill, i.e. `long` isn't long
			t.Fatalf("in-flight request: %d %q tokens=%d", r.code, r.c.Error, r.c.TokensGenerated)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("in-flight request not failed after worker died")
	}

	if err := startWorker(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(90 * time.Second)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		code, _, _ := complete(ctx, short)
		cancel()
		if code == http.StatusOK {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("never recovered after worker restart")
		}
		time.Sleep(time.Second)
	}

	// failAll must have returned every sem token across the reconnect
	for i, code := range fanOut(t, engineSlots*2, short, 30*time.Second) {
		if code != http.StatusOK {
			t.Errorf("request %d after reconnect: status %d", i, code)
		}
	}
}
