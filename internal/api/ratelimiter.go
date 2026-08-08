package api

import (
	"context"
	"sync"
	"time"
)

type RateLimiter interface {
	allow(userId string) bool
	sweep()
}

func NewRateLimiter(ctx context.Context, limiterType string, N int, D int, bucketTtl time.Duration, sweepInterval time.Duration) RateLimiter {
	switch limiterType {
	default:
		return NewTokenBucketRateLimiter(ctx, N, D, bucketTtl, sweepInterval)
	}
}

type Bucket struct {
	tokens        float64
	ratePerMs     float64
	lastTimestamp time.Time
}

type TokenBucketRateLimiter struct {
	ctx           context.Context
	mu            sync.Mutex
	clients       map[string]*Bucket
	N             int
	D             int
	bucketTTL     time.Duration
	sweepInterval time.Duration
}

func NewTokenBucketRateLimiter(ctx context.Context, N int, D int, bucketTtl time.Duration, sweepInterval time.Duration) *TokenBucketRateLimiter {
	r := &TokenBucketRateLimiter{
		ctx:           ctx,
		clients:       make(map[string]*Bucket),
		N:             N,
		D:             D,
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
			ratePerMs:     float64(r.N) / float64(r.D),
			lastTimestamp: time.Now(),
		}
		r.clients[userId] = b
	}

	now := time.Now()
	accrual := b.ratePerMs * (float64(now.Sub(b.lastTimestamp).Milliseconds()))
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
