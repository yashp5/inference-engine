package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/yashp5/inference-engine/internal/api"
	"github.com/yashp5/inference-engine/internal/config"
	"github.com/yashp5/inference-engine/internal/registry"
)

func main() {
	cfg := config.Load()

	ctx := context.Background()

	// N=0 is NewRateLimiter's off switch: it returns the no-op limiter
	rateLimitN := 0
	if cfg.RateLimit {
		rateLimitN = cfg.RateLimitN
	}
	r := api.NewRateLimiter(ctx, "TOKEN_BUCKET", rateLimitN, cfg.RateLimitWindow, cfg.RateLimitBucketTTL, cfg.RateLimitSweepInterval)
	log.Printf("rate limiter: enabled=%t n=%d window=%s", cfg.RateLimit, cfg.RateLimitN, cfg.RateLimitWindow)

	catalog := registry.LegacyCatalog(cfg)
	if cfg.Models != "" {
		var err error
		catalog, err = registry.LoadCatalog(cfg.Models, cfg.DefaultModel)
		if err != nil {
			log.Fatalf("load catalog %s: %v", cfg.Models, err)
		}
	}
	reg, err := registry.New(ctx, catalog, cfg)
	if err != nil {
		log.Fatalf("build registry: %v", err)
	}
	for _, m := range catalog {
		log.Printf("model %q: worker=%s continuous=%t", m.Name, m.WorkerAddr, m.ContinuousBatching)
	}

	h := api.NewHandler(r, reg, cfg.QueueTimeout, cfg.RequestTimeout, cfg.AgingInterval)
	mux := api.NewMux(h)

	srv := &http.Server{Addr: cfg.HTTPAddr, Handler: mux}
	go srv.ListenAndServe()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	srv.Shutdown(ctx)
	reg.Close()
}
