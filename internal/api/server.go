package api

import "net/http"

func NewMux(h *Handler) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /infer", h.Infer)
	mux.HandleFunc("GET /health", h.Health)
	mux.HandleFunc("GET /stats", h.Stats)
	return mux
}
