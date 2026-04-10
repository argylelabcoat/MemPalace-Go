package miner

import (
	"testing"
)

func TestGenerateID(t *testing.T) {
	id1 := generateID()
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
