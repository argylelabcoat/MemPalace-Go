package embedder

import (
	"context"
	"testing"
)

// TestCreateEmbeddings_LargeBatch verifies that batches larger than the internal
// chunk boundary are handled correctly — all texts get embeddings, none are dropped.
// This exercises the chunk-splitting logic across the boundary (currently 64).
func TestCreateEmbeddings_LargeBatch(t *testing.T) {
	emb, err := New("", "")
	if err != nil {
		t.Fatalf("create embedder: %v", err)
	}
	defer emb.Close()

	ctx := context.Background()
	// 65 texts: one more than the chunk size so we exercise the split.
	texts := make([]string, 65)
	for i := range texts {
		texts[i] = "the quick brown fox jumps over the lazy dog"
	}

	vecs, err := emb.CreateEmbeddings(ctx, texts)
	if err != nil {
		t.Fatalf("CreateEmbeddings: %v", err)
	}
	if len(vecs) != len(texts) {
		t.Errorf("expected %d embeddings, got %d", len(texts), len(vecs))
	}
	for i, v := range vecs {
		if len(v) == 0 {
			t.Errorf("embedding %d is empty", i)
		}
	}
}

// BenchmarkSingleEmbed measures single-text embedding latency.
// Run with: go test -bench=BenchmarkSingleEmbed -benchtime=10s ./internal/embedder/
// With ORT: CGO_LDFLAGS="-L${HOME}/lib" go test -tags ORT -bench=BenchmarkSingleEmbed -benchtime=10s ./internal/embedder/
func BenchmarkSingleEmbed(b *testing.B) {
	emb, err := New("", "")
	if err != nil {
		b.Fatalf("create embedder: %v", err)
	}
	defer emb.Close()

	ctx := context.Background()
	text := "the quick brown fox jumps over the lazy dog"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := emb.CreateEmbedding(ctx, text)
		if err != nil {
			b.Fatalf("embed: %v", err)
		}
	}
}

// BenchmarkBatchEmbed measures batch embedding throughput (50 texts).
// Run with: go test -bench=BenchmarkBatchEmbed -benchtime=10s ./internal/embedder/
// With ORT: CGO_LDFLAGS="-L${HOME}/lib" go test -tags ORT -bench=BenchmarkBatchEmbed -benchtime=10s ./internal/embedder/
func BenchmarkBatchEmbed(b *testing.B) {
	emb, err := New("", "")
	if err != nil {
		b.Fatalf("create embedder: %v", err)
	}
	defer emb.Close()

	ctx := context.Background()
	texts := make([]string, 50)
	for i := range texts {
		texts[i] = "the quick brown fox jumps over the lazy dog"
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := emb.CreateEmbeddings(ctx, texts)
		if err != nil {
			b.Fatalf("batch embed: %v", err)
		}
	}
}
