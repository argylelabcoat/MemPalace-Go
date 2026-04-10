package embedder

import (
	"context"
	"strings"
	"testing"
)

// TestTruncateText verifies that truncateText keeps output within maxRunes
// even for pathological inputs (long URLs, Unicode math symbols).
func TestTruncateText(t *testing.T) {
	// Normal short text — should pass through unchanged.
	short := "hello world"
	if got := truncateText(short); got != short {
		t.Errorf("short text modified: got %q, want %q", got, short)
	}

	// Text exactly at the limit — must not be truncated.
	atLimit := strings.Repeat("a", maxRunes)
	if got := truncateText(atLimit); got != atLimit {
		t.Errorf("text at limit was modified")
	}

	// Text longer than the limit — must be truncated to exactly maxRunes runes.
	long := strings.Repeat("a", maxRunes+100)
	got := truncateText(long)
	if len([]rune(got)) != maxRunes {
		t.Errorf("truncated text has %d runes, want %d", len([]rune(got)), maxRunes)
	}

	// Long URL (single "word") — must be truncated to maxRunes runes.
	longURL := "https://" + strings.Repeat("x", maxRunes+50) + ".com/path/to/image.jpg"
	gotURL := truncateText(longURL)
	if len([]rune(gotURL)) > maxRunes {
		t.Errorf("URL not truncated: %d runes > %d", len([]rune(gotURL)), maxRunes)
	}

	// Unicode math symbols — each rune is a single code-point.
	mathText := strings.Repeat("𝐴", maxRunes+10)
	gotMath := truncateText(mathText)
	if len([]rune(gotMath)) > maxRunes {
		t.Errorf("unicode math not truncated: %d runes > %d", len([]rune(gotMath)), maxRunes)
	}
}

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

// BenchmarkCorpusEmbed mirrors the real LongMemEval workload: one question's
// worth of corpus (~50 sessions, each ~80 words). Profile with:
//
//	go test -bench=BenchmarkCorpusEmbed -benchtime=5x -cpuprofile=cpu.prof ./internal/embedder/
//	go tool pprof -top cpu.prof
func BenchmarkCorpusEmbed(b *testing.B) {
	emb, err := New("", "")
	if err != nil {
		b.Fatalf("create embedder: %v", err)
	}
	defer emb.Close()

	ctx := context.Background()

	// ~80-word sentence — realistic LongMemEval session length.
	single := "the quick brown fox jumped over the lazy dog near the river bank on a warm summer afternoon while birds sang in the trees and children played nearby along the winding path through the ancient forest full of tall oaks and whispering pines under a clear blue sky"
	texts := make([]string, 50)
	for i := range texts {
		texts[i] = single
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := emb.CreateEmbeddings(ctx, texts)
		if err != nil {
			b.Fatalf("corpus embed: %v", err)
		}
	}
}
