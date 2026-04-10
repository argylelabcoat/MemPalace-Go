// Package search provides vector similarity search for memory retrieval.
// It uses embeddings to find relevant drawers in the palace.
package search

import (
	"context"

	"github.com/argylelabcoat/mempalace-go/internal/palace"
	"github.com/argylelabcoat/mempalace-go/storage/govector"
)

type Store interface {
	Search(query []float32, limit int, filter map[string]any) ([]govector.SearchResult, error)
	Add(id string, vector []float32, payload map[string]any) error
	Delete(id string) error
	ListAll(limit int) ([]govector.SearchResult, error)
}

type Embedder interface {
	CreateEmbedding(ctx context.Context, text string) ([]float32, error)
}

type LlamaClient interface {
	CreateEmbedding(ctx context.Context, text string) ([]float32, error)
}

type Drawer struct {
	ID       string
	Wing     string
	Room     string
	Content  string
	Metadata map[string]string
}

type WingInfo struct {
	Name        string `json:"name"`
	DrawerCount int    `json:"drawer_count"`
}

type RoomInfo struct {
	Name        string `json:"name"`
	Wing        string `json:"wing"`
	DrawerCount int    `json:"drawer_count"`
}

type TaxonomyNode struct {
	Name  string                   `json:"name"`
	Rooms map[string]*TaxonomyNode `json:"rooms,omitempty"`
	Count int                      `json:"count"`
}

type Searcher struct {
	store    Store
	embedder Embedder
}

func NewSearcher(store Store, embedder Embedder) *Searcher {
	return &Searcher{store: store, embedder: embedder}
}

func (s *Searcher) Search(ctx context.Context, query string, wing, room string, nResults int) ([]Drawer, error) {
	vector, err := s.embedder.CreateEmbedding(ctx, query)
	if err != nil {
		return nil, err
	}

	filter := map[string]any{}
	if wing != "" {
		filter["wing"] = wing
	}
	if room != "" {
		filter["room"] = room
	}

	results, err := s.store.Search(vector, nResults, filter)
	if err != nil {
		return nil, err
	}

	var drawers []Drawer
	for _, r := range results {
		d := Drawer{
			ID:       r.ID,
			Metadata: map[string]string{},
		}
		if wingVal, ok := r.Payload["wing"].(string); ok {
			d.Wing = wingVal
			d.Metadata["wing"] = wingVal
		}
		if roomVal, ok := r.Payload["room"].(string); ok {
			d.Room = roomVal
			d.Metadata["room"] = roomVal
		}
		if contentVal, ok := r.Payload["content"].(string); ok {
			d.Content = contentVal
		}
		drawers = append(drawers, d)
	}
	return drawers, nil
}

func (s *Searcher) Store(ctx context.Context, drawer palace.Drawer) error {
	vector, err := s.embedder.CreateEmbedding(ctx, drawer.Content)
	if err != nil {
		return err
	}

	payload := map[string]any{
		"wing":    drawer.Wing,
		"room":    drawer.Room,
		"source":  drawer.SourceFile,
		"content": drawer.Content,
	}

	return s.store.Add(drawer.ID, vector, payload)
}

func (s *Searcher) ListWings(ctx context.Context) ([]WingInfo, error) {
	results, err := s.store.ListAll(10000)
	if err != nil {
		return nil, err
	}

	wingCounts := make(map[string]int)
	for _, r := range results {
		if wing, ok := r.Payload["wing"].(string); ok {
			wingCounts[wing]++
		}
	}

	var wings []WingInfo
	for wing, count := range wingCounts {
		wings = append(wings, WingInfo{Name: wing, DrawerCount: count})
	}
	return wings, nil
}

func (s *Searcher) ListRooms(ctx context.Context, wingFilter string) ([]RoomInfo, error) {
	results, err := s.store.ListAll(10000)
	if err != nil {
		return nil, err
	}

	type roomKey struct {
		Wing string
		Room string
	}
	roomCounts := make(map[roomKey]int)
	for _, r := range results {
		wing, wingOk := r.Payload["wing"].(string)
		room, roomOk := r.Payload["room"].(string)
		if wingOk && roomOk {
			if wingFilter == "" || wing == wingFilter {
				roomCounts[roomKey{Wing: wing, Room: room}]++
			}
		}
	}

	var rooms []RoomInfo
	for key, count := range roomCounts {
		rooms = append(rooms, RoomInfo{
			Name:        key.Room,
			Wing:        key.Wing,
			DrawerCount: count,
		})
	}
	return rooms, nil
}

func (s *Searcher) GetTaxonomy(ctx context.Context) (map[string]*TaxonomyNode, error) {
	results, err := s.store.ListAll(10000)
	if err != nil {
		return nil, err
	}

	taxonomy := make(map[string]*TaxonomyNode)
	for _, r := range results {
		wing, wingOk := r.Payload["wing"].(string)
		room, roomOk := r.Payload["room"].(string)
		if !wingOk || !roomOk {
			continue
		}

		if _, exists := taxonomy[wing]; !exists {
			taxonomy[wing] = &TaxonomyNode{
				Name:  wing,
				Rooms: make(map[string]*TaxonomyNode),
			}
		}

		if _, exists := taxonomy[wing].Rooms[room]; !exists {
			taxonomy[wing].Rooms[room] = &TaxonomyNode{
				Name:  room,
				Count: 0,
			}
		}
		taxonomy[wing].Rooms[room].Count++
		taxonomy[wing].Count++
	}

	return taxonomy, nil
}

func (s *Searcher) Delete(ctx context.Context, id string) error {
	return s.store.Delete(id)
}
