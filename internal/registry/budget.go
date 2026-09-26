package registry

// budget tracks memory reserved by models that are Loading, Ready or Draining.
// Like lru, it relies on Registry.mu.
//
// Two rules (docs/phase5-multi-model.md, section 5):
//   - reserve before starting the worker, never after, or two concurrent loads
//     can both pass a "does it fit?" check
//   - release when the process has exited, not when its eviction starts
type budget struct {
	limitMB    int
	reservedMB int
}

func newBudget(limitMB int) *budget { return &budget{limitMB: limitMB} }

// TODO(phase5)
func (b *budget) fits(mb int) bool { return false }
func (b *budget) reserve(mb int)   {}
func (b *budget) release(mb int)   {}
