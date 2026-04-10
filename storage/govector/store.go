// Package govector provides a GoVector-based vector store implementation.
// GoVector is a pure Go HNSW vector database for embeddings storage.
package govector

import (
	"github.com/DotNetAge/govector/core"
)

type Store struct {
	collection *core.Collection
	storage    *core.Storage
}

type SearchResult struct {
	ID      string
	Score   float32
	Payload map[string]any
}

func NewStore(dbPath string, dimension int) (*Store, error) {
	storage, err := core.NewStorage(dbPath)
	if err != nil {
		return nil, err
	}
	col, err := core.NewCollection("mempalace_drawers", dimension, core.Cosine, storage, true)
	if err != nil {
		return nil, err
	}
	return &Store{collection: col, storage: storage}, nil
}

func (s *Store) Add(id string, vector []float32, payload map[string]any) error {
	return s.collection.Upsert([]core.PointStruct{{
		ID:      id,
		Vector:  vector,
		Payload: core.Payload(payload),
	}})
}

// Point is a pre-constructed vector with its payload.
type Point struct {
	ID      string
	Vector  []float32
	Payload map[string]any
}

// AddBatch stores multiple points in a single upsert call.
func (s *Store) AddBatch(points []Point) error {
	ps := make([]core.PointStruct, len(points))
	for i, p := range points {
		ps[i] = core.PointStruct{
			ID:      p.ID,
			Vector:  p.Vector,
			Payload: core.Payload(p.Payload),
		}
	}
	return s.collection.Upsert(ps)
}

func (s *Store) Search(query []float32, limit int, filter map[string]any) ([]SearchResult, error) {
	var cf *core.Filter
	if len(filter) > 0 {
		cf = &core.Filter{
			Must: buildConditions(filter),
		}
	}
	results, err := s.collection.Search(query, cf, limit)
	if err != nil {
		return nil, err
	}
	var searchResults []SearchResult
	for _, r := range results {
		searchResults = append(searchResults, SearchResult{
			ID:      r.ID,
			Score:   r.Score,
			Payload: r.Payload,
		})
	}
	return searchResults, nil
}

func buildConditions(filter map[string]any) []core.Condition {
	var conditions []core.Condition
	for key, val := range filter {
		// Check if it's a range filter
		if rangeFilter, ok := val.(map[string]any); ok {
			conditions = append(conditions, buildRangeConditions(key, rangeFilter)...)
		} else {
			conditions = append(conditions, core.Condition{
				Key:   key,
				Type:  core.MatchTypeExact,
				Match: core.MatchValue{Value: val},
			})
		}
	}
	return conditions
}

func buildRangeConditions(key string, filter map[string]any) []core.Condition {
	var conditions []core.Condition

	rangeVal := &core.RangeValue{}
	hasRange := false

	if val, ok := filter["gt"]; ok {
		rangeVal.GT = val
		hasRange = true
	}
	if val, ok := filter["gte"]; ok {
		rangeVal.GTE = val
		hasRange = true
	}
	if val, ok := filter["lt"]; ok {
		rangeVal.LT = val
		hasRange = true
	}
	if val, ok := filter["lte"]; ok {
		rangeVal.LTE = val
		hasRange = true
	}

	if hasRange {
		conditions = append(conditions, core.Condition{
			Key:   key,
			Type:  core.MatchTypeRange,
			Range: rangeVal,
		})
	}

	if val, ok := filter["eq"]; ok {
		conditions = append(conditions, core.Condition{
			Key:   key,
			Type:  core.MatchTypeExact,
			Match: core.MatchValue{Value: val},
		})
	}

	return conditions
}

func (s *Store) Close() error {
	if s.storage != nil {
		return s.storage.Close()
	}
	return nil
}

func (s *Store) Delete(id string) error {
	_, err := s.collection.Delete([]string{id}, nil)
	return err
}

func (s *Store) ListAll(limit int) ([]SearchResult, error) {
	zeroVector := make([]float32, s.collection.VectorLen)
	results, err := s.collection.Search(zeroVector, nil, limit)
	if err != nil {
		return nil, err
	}
	var searchResults []SearchResult
	for _, r := range results {
		searchResults = append(searchResults, SearchResult{
			ID:      r.ID,
			Score:   r.Score,
			Payload: r.Payload,
		})
	}
	return searchResults, nil
}
