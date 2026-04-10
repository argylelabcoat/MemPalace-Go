// Package embedder provides text embedding using Hugging Face transformers via hugot.
// It uses ONNX models for efficient, native Go embeddings without external processes.
package embedder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/daulet/tokenizers"
	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

// maxTokens is the model's position embedding limit. all-MiniLM-L6-v2
// and most sentence-transformer models support up to 512 subword tokens.
const maxTokens = 512

// Embedder generates text embeddings using hugot (Hugging Face ONNX runtime).
type Embedder struct {
	pipeline    *pipelines.FeatureExtractionPipeline
	session     *hugot.Session
	tokenizer   *tokenizers.Tokenizer
	modelPath   string
	modelsDir   string
	modelName   string
	initialized bool
}

// New creates a new Embedder. It downloads (if needed) and loads the model.
func New(modelName string, modelsDir string) (*Embedder, error) {
	if modelName == "" {
		modelName = "sentence-transformers/all-MiniLM-L6-v2"
	}

	downloadOpts := hugot.NewDownloadOptions()
	downloadOpts.OnnxFilePath = "onnx/model.onnx"
	downloadOpts.Verbose = false

	modelPath, err := hugot.DownloadModel(modelName, modelsDir, downloadOpts)
	if err != nil {
		return nil, fmt.Errorf("download model: %w", err)
	}

	e := &Embedder{
		modelPath: modelPath,
		modelsDir: modelsDir,
		modelName: modelName,
	}

	if err := e.loadTokenizer(); err != nil {
		// Non-fatal: fall back to rune-based truncation.
		// This can happen if tokenizer.json is missing or corrupted.
		fmt.Fprintf(os.Stderr, "warning: tokenizer unavailable, falling back to rune truncation: %v\n", err)
	}

	// Try ORT + CoreML (Apple Silicon GPU/ANE acceleration).
	session, err := hugot.NewORTSession(ortDylibOption(), options.WithCoreML(coreMLFlags()))
	if err != nil {
		// CoreML unavailable (non-darwin, older macOS, or missing EP).
		// Try plain ORT (CPU).
		session, err = hugot.NewORTSession(ortDylibOption())
		if err != nil {
			// Final fallback: pure-Go (no CGo, works everywhere).
			session, err = hugot.NewGoSession()
			if err != nil {
				return nil, fmt.Errorf("create session: %w", err)
			}
		}
	}

	config := hugot.FeatureExtractionConfig{
		ModelPath: modelPath,
		Name:      "mempalace-embeddings",
	}

	pipeline, err := hugot.NewPipeline[*pipelines.FeatureExtractionPipeline](session, config)
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("create pipeline: %w", err)
	}

	e.pipeline = pipeline
	e.session = session
	e.initialized = true

	return e, nil
}

// loadTokenizer loads the HuggingFace tokenizer from the model directory.
func (e *Embedder) loadTokenizer() error {
	// Try the model's tokenizer.json first.
	tokenizerPath := filepath.Join(e.modelPath, "tokenizer.json")
	if _, err := os.Stat(tokenizerPath); err == nil {
		tok, err := tokenizers.FromFile(tokenizerPath)
		if err == nil {
			e.tokenizer = tok
			return nil
		}
	}

	// Fallback: check the sibling sentence-transformers directory (downloaded separately).
	altPath := filepath.Join(e.modelsDir, "sentence-transformers_all-MiniLM-L6-v2", "tokenizer.json")
	if _, err := os.Stat(altPath); err == nil {
		tok, err := tokenizers.FromFile(altPath)
		if err == nil {
			e.tokenizer = tok
			return nil
		}
	}

	return fmt.Errorf("tokenizer.json not found in %s or %s", tokenizerPath, altPath)
}

// truncateText limits text to fit within the model's token limit.
// Uses the actual tokenizer when available; falls back to rune-count truncation.
func (e *Embedder) truncateText(text string) string {
	if e.tokenizer != nil {
		return e.truncateByTokens(text)
	}
	return truncateByRunes(text)
}

// truncateByTokens encodes text, truncates to maxTokens, and decodes back to string.
// This uses the full 512-token window instead of the conservative 400-rune limit.
func (e *Embedder) truncateByTokens(text string) string {
	tokenIDs, _ := e.tokenizer.Encode(text, false)
	if len(tokenIDs) <= maxTokens {
		return text
	}
	// Decode the first maxTokens tokens back to a string.
	truncated := e.tokenizer.Decode(tokenIDs[:maxTokens], false)
	if truncated == "" {
		return truncateByRunes(text)
	}
	return truncated
}

// truncateByRunes is the fallback: limits text to ~400 Unicode code-points.
// Word-count limits are unreliable because URLs and Unicode math symbols can
// produce far more than 4 subword tokens per word; character limits are safe.
func truncateByRunes(text string) string {
	runes := []rune(text)
	if len(runes) <= 400 {
		return text
	}
	return string(runes[:400])
}

// CreateEmbedding generates a float32 vector for the given text.
func (e *Embedder) CreateEmbedding(ctx context.Context, text string) ([]float32, error) {
	output, err := e.pipeline.RunPipeline([]string{e.truncateText(text)})
	if err != nil {
		return nil, err
	}

	if len(output.Embeddings) == 0 {
		return nil, fmt.Errorf("empty embedding output")
	}

	return output.Embeddings[0], nil
}

// CreateEmbeddings batch-embeds multiple texts in a single forward pass.
// It processes texts in chunks of 64 (recommended by hugot) and handles
// GoMLX graph shape mismatches by falling back to single embeddings when needed.
func (e *Embedder) CreateEmbeddings(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	// Truncate all texts first
	truncated := make([]string, len(texts))
	for i, t := range texts {
		truncated[i] = e.truncateText(t)
	}

	const chunkSize = 64
	allEmbeddings := make([][]float32, 0, len(texts))

	for i := 0; i < len(truncated); i += chunkSize {
		end := min(i+chunkSize, len(truncated))
		chunk := truncated[i:end]

		output, err := e.pipeline.RunPipeline(chunk)
		if err != nil {
			// GoMLX graph shape mismatch: fall back to single embeddings
			for _, text := range chunk {
				single, err2 := e.pipeline.RunPipeline([]string{text})
				if err2 != nil {
					return nil, fmt.Errorf("batch+fallback embed: %w (single: %v)", err, err2)
				}
				if len(single.Embeddings) > 0 {
					allEmbeddings = append(allEmbeddings, single.Embeddings[0])
				}
			}
			continue
		}
		allEmbeddings = append(allEmbeddings, output.Embeddings...)
	}
	return allEmbeddings, nil
}

// coreMLFlags returns CoreML execution provider options that allow the ANE,
// GPU, and CPU to all be used. See:
// https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
func coreMLFlags() map[string]string {
	return map[string]string{
		// MLComputeUnitsAll (0) = CPU+GPU+ANE, CoreML decides optimal placement.
		"MLComputeUnits": "0",
	}
}

// ortDylibOption returns the hugot option that points ORT at the directory
// containing the arm64 dylib bundled inside yalue/onnxruntime_go when running
// on darwin/arm64. options.WithOnnxLibraryPath expects a directory; it appends
// the platform-specific filename internally.
// On other platforms returns a no-op option (ORT searches PATH/LD_LIBRARY_PATH).
func ortDylibOption() options.WithOption {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		dir, err := findOrtDylibDir()
		if err == nil {
			return options.WithOnnxLibraryPath(dir)
		}
	}
	return func(o *options.Options) error { return nil }
}

// findOrtDylibDir locates the test_data directory inside yalue/onnxruntime_go
// in the Go module cache. The arm64 dylib there already links CoreML + Metal.
func findOrtDylibDir() (string, error) {
	gopath := os.Getenv("GOPATH")
	if gopath == "" {
		gopath = filepath.Join(os.Getenv("HOME"), "go")
	}
	dir := filepath.Join(gopath, "pkg", "mod",
		"github.com", "yalue", "onnxruntime_go@v1.27.0",
		"test_data")
	if _, err := os.Stat(dir); err != nil {
		return "", err
	}
	return dir, nil
}

// Close releases the hugot session and tokenizer resources.
func (e *Embedder) Close() {
	if e.tokenizer != nil {
		e.tokenizer.Close()
	}
	if e.session != nil {
		e.session.Destroy()
	}
}
