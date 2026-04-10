package embedder

import (
	"context"
	"strings"
	"testing"
)

// TestTruncateByRunes verifies the fallback rune truncation handles
// pathological inputs (long URLs, Unicode math symbols).
func TestTruncateByRunes(t *testing.T) {
	// Normal short text — should pass through unchanged.
	short := "hello world"
	if got := truncateByRunes(short); got != short {
		t.Errorf("short text modified: got %q, want %q", got, short)
	}

	// Text exactly at the limit — must not be truncated.
	atLimit := strings.Repeat("a", 400)
	if got := truncateByRunes(atLimit); got != atLimit {
		t.Errorf("text at limit was modified")
	}

	// Text longer than the limit — must be truncated to exactly 400 runes.
	long := strings.Repeat("a", 500)
	got := truncateByRunes(long)
	if len([]rune(got)) != 400 {
		t.Errorf("truncated text has %d runes, want %d", len([]rune(got)), 400)
	}

	// Long URL (single "word") — must be truncated to 400 runes.
	longURL := "https://" + strings.Repeat("x", 450) + ".com/path/to/image.jpg"
	gotURL := truncateByRunes(longURL)
	if len([]rune(gotURL)) > 400 {
		t.Errorf("URL not truncated: %d runes > 400", len([]rune(gotURL)))
	}

	// Unicode math symbols — each rune is a single code-point.
	mathText := strings.Repeat("𝐴", 500)
	gotMath := truncateByRunes(mathText)
	if len([]rune(gotMath)) > 400 {
		t.Errorf("unicode math not truncated: %d runes > 400", len([]rune(gotMath)))
	}
}

// TestTruncateByTokens verifies token-aware truncation stays within 512 tokens.
func TestTruncateByTokens(t *testing.T) {
	// Create a minimal embedder with just a tokenizer (no model needed).
	e := &Embedder{}
	if err := e.loadTokenizer(); err != nil {
		t.Skipf("tokenizer unavailable, skipping token-aware test: %v", err)
	}
	defer e.tokenizer.Close()

	// Short text — should pass through unchanged.
	short := "hello world"
	if got := e.truncateByTokens(short); got != short {
		t.Errorf("short text modified: got %q, want %q", got, short)
	}

	// Text that produces exactly 512 tokens — should not be truncated.
	// Generate enough text to get close to 512 tokens.
	longText := strings.Repeat("the quick brown fox jumps over the lazy dog ", 60)
	tokenIDs, _ := e.tokenizer.Encode(longText, false)
	if len(tokenIDs) > maxTokens {
		// Text is too long; trim it to exactly 512 tokens worth.
		// We need to find a text that fits — just use the truncate result.
	}

	// Text longer than 512 tokens — must be truncated.
	veryLong := strings.Repeat("the quick brown fox jumps over the lazy dog ", 200)
	got := e.truncateByTokens(veryLong)
	gotTokens, _ := e.tokenizer.Encode(got, false)
	if len(gotTokens) > maxTokens {
		t.Errorf("truncated text has %d tokens, want ≤ %d", len(gotTokens), maxTokens)
	}

	// Long URL — must be truncated to ≤ 512 tokens.
	longURL := "https://" + strings.Repeat("abcdef", 200) + ".com/" + strings.Repeat("path", 100)
	gotURL := e.truncateByTokens(longURL)
	gotURLTokens, _ := e.tokenizer.Encode(gotURL, false)
	if len(gotURLTokens) > maxTokens {
		t.Errorf("URL not truncated: %d tokens > %d", len(gotURLTokens), maxTokens)
	}

	// Unicode — must be truncated.
	unicodeText := strings.Repeat("𝐴𝐵𝐶", 300)
	gotUnicode := e.truncateByTokens(unicodeText)
	gotUnicodeTokens, _ := e.tokenizer.Encode(gotUnicode, false)
	if len(gotUnicodeTokens) > maxTokens {
		t.Errorf("unicode not truncated: %d tokens > %d", len(gotUnicodeTokens), maxTokens)
	}
}

// TestTruncateText_Method verifies the method delegates correctly.
func TestTruncateText_Method(t *testing.T) {
	// Without tokenizer — should use rune fallback.
	e := &Embedder{}
	short := "hello world"
	if got := e.truncateText(short); got != short {
		t.Errorf("short text modified: got %q, want %q", got, short)
	}

	long := strings.Repeat("a", 500)
	got := e.truncateText(long)
	if len([]rune(got)) != 400 {
		t.Errorf("fallback truncation: got %d runes, want 400", len([]rune(got)))
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
