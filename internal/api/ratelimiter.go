package api

import (
	"context"
	"sync"
	"time"
)

type RateLimiter interface {
	allow(userId string) bool
}

func NewRateLimiter(ctx context.Context, limiterType string, N int, window time.Duration, bucketTtl time.Duration, sweepInterval time.Duration) RateLimiter {
	if N <= 0 || window <= 0 {
		return noopRateLimiter{}
	}
	switch limiterType {
	default:
		return NewTokenBucketRateLimiter(ctx, N, window, bucketTtl, sweepInterval)
	}
}

type noopRateLimiter struct{}

func (noopRateLimiter) allow(string) bool { return true }

type Bucket struct {
	tokens        float64
	ratePerNano   float64
	lastTimestamp time.Time
}

type TokenBucketRateLimiter struct {
	ctx           context.Context
	mu            sync.Mutex
	clients       map[string]*Bucket
	N             int
	window        time.Duration
	bucketTTL     time.Duration
	sweepInterval time.Duration
}

func NewTokenBucketRateLimiter(ctx context.Context, N int, D time.Duration, bucketTtl time.Duration, sweepInterval time.Duration) *TokenBucketRateLimiter {
	r := &TokenBucketRateLimiter{
		ctx:           ctx,
		clients:       make(map[string]*Bucket),
		N:             N,
		window:        D,
		bucketTTL:     bucketTtl,
		sweepInterval: sweepInterval,
	}
	r.sweep()
	return r
}

func (r *TokenBucketRateLimiter) allow(userId string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	b, ok := r.clients[userId]
	if !ok {
		b = &Bucket{
			tokens:        float64(r.N),
			ratePerNano:   float64(r.N) / float64(r.window),
			lastTimestamp: time.Now(),
		}
		r.clients[userId] = b
	}

	now := time.Now()
	accrual := b.ratePerNano * (float64(now.Sub(b.lastTimestamp)))
	b.tokens = min(b.tokens+accrual, float64(r.N))
	b.lastTimestamp = now

	if b.tokens-1.0 < 0 {
		return false
	}

	b.tokens = max(b.tokens-1.0, 0.0)
	return true
}

func (r *TokenBucketRateLimiter) sweep() {
	go func() {
		ticker := time.NewTicker(r.sweepInterval)
		defer ticker.Stop()
		for {
			select {
			case <-r.ctx.Done():
				return
			case now := <-ticker.C:
				r.mu.Lock()
				for k, v := range r.clients {
					if now.Sub(v.lastTimestamp) > r.bucketTTL {
						delete(r.clients, k)
					}
				}
				r.mu.Unlock()
			}
		}
	}()
}
