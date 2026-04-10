package miner

import (
	"strings"
	"testing"
	"time"
)

func TestGenerateID(t *testing.T) {
	id1 := generateID()
	// Add a small delay to ensure different timestamps
	time.Sleep(1 * time.Millisecond)
	id2 := generateID()
	if id1 == id2 {
		t.Errorf("generateID should produce unique IDs, got %s twice", id1)
	}
	if len(id1) < 5 || id1[:2] != "d_" {
		t.Errorf("generateID should start with 'd_', got %s", id1)
	}
}

func TestDetectFormat(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		expected string
	}{
		{"claude format", `{"role": "user", "content": "hello"}`, "claude"},
		{"chatgpt format", `{"message": {"role": "user", "content": "hello"}}`, "chatgpt"},
		{"unknown format", `{"data": "test"}`, "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := DetectFormat([]byte(tt.content))
			if result != tt.expected {
				t.Errorf("DetectFormat(%q) = %q, want %q", tt.content, result, tt.expected)
			}
		})
	}
}

func TestNormalizeClaudeExport(t *testing.T) {
	valid := []byte(`{"messages": [{"role": "user", "content": "hello"}], "source": "test"}`)
	result, err := NormalizeClaudeExport(valid)
	if err != nil {
		t.Errorf("NormalizeClaudeExport with valid JSON returned error: %v", err)
	}
	if len(result.Messages) != 1 || result.Messages[0].Role != "user" {
		t.Errorf("NormalizeClaudeExport parsed incorrectly: %+v", result)
	}
}

func TestNormalizeClaudeExportInvalid(t *testing.T) {
	invalid := []byte(`{invalid json}`)
	_, err := NormalizeClaudeExport(invalid)
	if err == nil {
		t.Errorf("NormalizeClaudeExport with invalid JSON should return error")
	}
}

func TestChunkContent(t *testing.T) {
	// Test short content
	content := "short"
	chunks := chunkContent(content, 100, 20)
	if len(chunks) != 1 || chunks[0] != "short" {
		t.Errorf("chunkContent(\"short\", 100, 20) = %v, want [%q]", chunks, "short")
	}

	// Test content that fits exactly
	content = strings.Repeat("x", 100)
	chunks = chunkContent(content, 100, 20)
	if len(chunks) != 1 || chunks[0] != content {
		t.Errorf("chunkContent(exact fit, 100, 20) = %v, want [%q]", chunks, content)
	}

	// Test content requiring chunking without boundaries
	content = strings.Repeat("x", 250)       // 250 chars
	chunks = chunkContent(content, 800, 100) // Use same values as actual call in Miner.MineProject
	if len(chunks) != 1 {                    // 250 < 800, so should be 1 chunk
		t.Errorf("chunkContent(250x, 800, 100) = %v, want 1 chunk", chunks)
	}
	if len(chunks[0]) != 250 {
		t.Errorf("chunkContent first chunk incorrect length: got %d, want 250", len(chunks[0]))
	}

	// Test with smaller chunk size to force splitting
	content = strings.Repeat("x", 900) // 900 chars
	chunks = chunkContent(content, 800, 100)
	if len(chunks) != 2 {
		t.Errorf("chunkContent(900x, 800, 100) = %v, want 2 chunks", chunks)
	}
	// First chunk should be ~800 chars (trying to break at boundary)
	// Second chunk should be remainder
	if len(chunks[0]) < 700 || len(chunks[0]) > 900 {
		t.Errorf("chunkContent first chunk length unreasonable: got %d", len(chunks[0]))
	}
	if len(chunks[1]) < 0 || len(chunks[1]) > 200 {
		t.Errorf("chunkContent second chunk length unreasonable: got %d", len(chunks[1]))
	}

	// Test empty content
	content = ""
	chunks = chunkContent(content, 800, 100)
	if len(chunks) != 0 {
		t.Errorf("chunkContent empty string = %v, want []", chunks)
	}

	// Test content with only spaces
	content = "   \n\t\n   "
	chunks = chunkContent(content, 800, 100)
	if len(chunks) != 0 {
		t.Errorf("chunkContent whitespace only = %v, want []", chunks)
	}
}
