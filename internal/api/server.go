package api

import "net/http"

func NewMux(h *Handler) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/completions", h.Completions)
	mux.HandleFunc("GET /healthz", h.Healthz)
	mux.HandleFunc("GET /stats", h.Stats)
	return mux
}
