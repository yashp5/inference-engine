package registry

// lru orders models by last use. Not safe on its own: every call happens under
// Registry.mu, so that picking victims and marking them Draining is one atomic
// step with respect to Acquire.
//
// TODO(phase5): container/list plus map[string]*list.Element. touch moves to
// the front; victims walks from the back.
type lru struct{}

func newLRU() *lru { return &lru{} }

func (l *lru) touch(name string)  {}
func (l *lru) remove(name string) {}

// victims returns least-recently-used first, skipping anything eligible
// rejects. The registry passes "Ready and refs == 0".
func (l *lru) victims(eligible func(name string) bool) []string {
	return nil
}
