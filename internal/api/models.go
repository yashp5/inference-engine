package api

import "net/http"

// Admin endpoints for Phase 5 (docs/phase5-multi-model.md, section 8). Not
// routed yet; register them in NewMux once the registry is wired into Handler:
//
//	mux.HandleFunc("GET /v1/models", h.ListModels)
//	mux.HandleFunc("POST /v1/models/load", h.LoadModel)
//	mux.HandleFunc("DELETE /v1/models/{name}", h.UnloadModel)

// ListModels returns registry.List(): state, refs, memory, last use and pid
// for every catalog model.
func (h *Handler) ListModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]string{"error": "not implemented"})
}

// LoadModel takes {"model": "..."} and blocks until the model is Ready or
// -loadTimeout. Idempotent: an already Ready model returns 200.
func (h *Handler) LoadModel(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]string{"error": "not implemented"})
}

// UnloadModel drains r.PathValue("name") and stops its worker.
func (h *Handler) UnloadModel(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]string{"error": "not implemented"})
}
