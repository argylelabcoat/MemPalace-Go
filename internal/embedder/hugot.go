// Package embedder provides text embedding using Hugging Face transformers via hugot.
// It uses ONNX models for efficient, native Go embeddings without external processes.
package embedder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

// maxTokens is set to match ChromaDB's ONNXMiniLM_L6_V2 which uses
// tokenizer.enable_truncation(max_length=256). We use 170 words because
// BERT's WordPiece tokenizer produces ~1.5 subword tokens per English word.
// 170 × 1.5 ≈ 255 tokens, safely within the 256 limit.
const maxTokens = 170

// Embedder generates text embeddings using hugot (Hugging Face ONNX runtime).
type Embedder struct {
	pipeline *pipelines.FeatureExtractionPipeline
	session  *hugot.Session
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

	return &Embedder{
		pipeline: pipeline,
		session:  session,
	}, nil
}

// truncateText limits text to maxTokens words to stay within the model's
// 512-token position embedding limit.
func truncateText(text string) string {
	words := strings.Fields(text)
	if len(words) <= maxTokens {
		return text
	}
	return strings.Join(words[:maxTokens], " ")
}

// CreateEmbedding generates a float32 vector for the given text.
func (e *Embedder) CreateEmbedding(ctx context.Context, text string) ([]float32, error) {
	output, err := e.pipeline.RunPipeline([]string{truncateText(text)})
	if err != nil {
		return nil, err
	}

	if len(output.Embeddings) == 0 {
		return nil, fmt.Errorf("empty embedding output")
	}

	return output.Embeddings[0], nil
}

// CreateEmbeddings batch-embeds multiple texts in a single forward pass.
// It processes texts in chunks of 32 (recommended by hugot) and handles
// GoMLX graph shape mismatches by falling back to single embeddings when needed.
func (e *Embedder) CreateEmbeddings(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	// Truncate all texts first
	truncated := make([]string, len(texts))
	for i, t := range texts {
		truncated[i] = truncateText(t)
	}

	const chunkSize = 32
	allEmbeddings := make([][]float32, 0, len(texts))

	for i := 0; i < len(truncated); i += chunkSize {
		end := i + chunkSize
		if end > len(truncated) {
			end = len(truncated)
		}
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

// Close releases the hugot session resources.
func (e *Embedder) Close() {
	if e.session != nil {
		e.session.Destroy()
	}
}
