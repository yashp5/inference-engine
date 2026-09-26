# ── Variables ──────────────────────────────────────────────────────────────────
BINARY_DIR       = bin
PROTO_DIR        = proto
GEN_DIR          = gen
PYTHON_GEN_DIR   = cmd/worker
SERVER_BIN       = $(BINARY_DIR)/server
PYTHON           = python3.14
WORKER_PY ?= venv/bin/python

# Automatically picks up every .proto file in proto/ —
# no need to manually list them when you add new ones
PROTO_FILES = $(wildcard $(PROTO_DIR)/*.proto)

# ── Default ────────────────────────────────────────────────────────────────────
.PHONY: all
all: generate build

# ── generate ───────────────────────────────────────────────────────────────────
# Compiles every .proto in proto/ into both Go and Python code.
#
# Go output   → gen/
#   inference.pb.go       (message structs)
#   inference_grpc.pb.go  (server interface + client stub)
#
# Python output → cmd/worker/
#   inference_pb2.py      (message classes)
#   inference_pb2_grpc.py (servicer base class + stub)
#
# Python files live next to the worker so imports are just:
#   import inference_pb2, inference_pb2_grpc
#
# Override the Python interpreter if needed:
#   make generate PYTHON=python3.12
.PHONY: generate
generate:
	@echo "==> Generating Go protobuf files..."
	@mkdir -p $(GEN_DIR)
	protoc \
		--proto_path=$(PROTO_DIR) \
		--go_out=$(GEN_DIR) \
		--go_opt=paths=source_relative \
		--go-grpc_out=$(GEN_DIR) \
		--go-grpc_opt=paths=source_relative \
		$(PROTO_FILES)
	@echo "==> Generating Python protobuf files..."
	@mkdir -p $(PYTHON_GEN_DIR)
	$(PYTHON) -m grpc_tools.protoc \
		--proto_path=$(PROTO_DIR) \
		--python_out=$(PYTHON_GEN_DIR) \
		--pyi_out=$(PYTHON_GEN_DIR) \
		--grpc_python_out=$(PYTHON_GEN_DIR) \
		$(PROTO_FILES)
	@echo "==> Done. Go → $(GEN_DIR)/  Python → $(PYTHON_GEN_DIR)/"

# ── build ──────────────────────────────────────────────────────────────────────
# Compiles the Go server binary into bin/
.PHONY: build
build:
	@echo "==> Building server..."
	@mkdir -p $(BINARY_DIR)
	go build -o $(SERVER_BIN) ./cmd/server
	@echo "==> Binary written to $(SERVER_BIN)"

# ── run ────────────────────────────────────────────────────────────────────────
# Runs the Go server directly without building a binary first
.PHONY: run
run:
	go run ./cmd/server

# ── run-worker ─────────────────────────────────────────────────────────────────
# Runs the Python gRPC worker.
# Override the interpreter if needed:
#   make run-worker PYTHON=python3.12
.PHONY: run-worker
run-worker:
	$(PYTHON) cmd/worker/worker.py


# --- test ------------------------------------------------------------------------
.PHONY: test-engine test-worker test-e2e test
test-engine:
	cd cmd/worker && $(WORKER_PY) -m pytest tests/test_engine.py -v

test-worker:
	cd cmd/worker && $(WORKER_PY) -m pytest tests/test_worker_grpc.py -v -s

test-e2e:
	go test -tags e2e -v -count=1 -timeout 10m ./test/e2e/

test: test-engine test-worker test-e2e

# ── clean ──────────────────────────────────────────────────────────────────────
# Removes all generated files (Go + Python) and compiled binaries
.PHONY: clean
clean:
	@echo "==> Removing Go generated files ($(GEN_DIR)/)..."
	@rm -rf $(GEN_DIR)
	@echo "==> Removing Python generated files from $(PYTHON_GEN_DIR)/..."
	@rm -f $(PYTHON_GEN_DIR)/inference_pb2.py
	@rm -f $(PYTHON_GEN_DIR)/inference_pb2.pyi
	@rm -f $(PYTHON_GEN_DIR)/inference_pb2_grpc.py
	@echo "==> Removing binaries ($(BINARY_DIR)/)..."
	@rm -rf $(BINARY_DIR)
	@echo "==> Clean complete."

# ── help ───────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "  generate      Compile all .proto files → Go (gen/) + Python (cmd/worker/)"
	@echo "  build         Build the Go server binary into bin/"
	@echo "  run           Run the Go server with go run"
	@echo "  run-worker    Run the Python gRPC worker"
	@echo "  test          Run test-engine, test-worker, then test-e2e"
	@echo "  test-engine   Run the in-process engine tests (tests/test_engine.py)"
	@echo "  test-worker   Start the worker and run the gRPC tests (tests/test_worker_grpc.py)"
	@echo "  test-e2e      Run the Go end-to-end tests (test/e2e/)"
	@echo "  clean         Remove all generated files and binaries"
	@echo "  all           generate + build  (default)"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON        Python interpreter to use (default: python3.14)"
	@echo "                Example: make generate PYTHON=python3.12"
	@echo "  WORKER_PY     Python used by the test targets, relative to cmd/worker/"
	@echo "                (default: venv/bin/python, i.e. cmd/worker/venv)"
	@echo "                Example: make test WORKER_PY=../../venv/bin/python"
	@echo ""
