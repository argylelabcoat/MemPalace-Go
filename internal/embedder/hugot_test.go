package embedder

import (
	"context"
	"testing"
)

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
