// Package embedder provides text embedding using Hugging Face transformers via hugot.
// It uses ONNX models for efficient, native Go embeddings without external processes.
package embedder

import (
	"context"
	"fmt"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
)

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

	modelPath, err := hugot.DownloadModel(modelName, modelsDir, hugot.NewDownloadOptions())
	if err != nil {
		return nil, fmt.Errorf("download model: %w", err)
	}

	session, err := hugot.NewGoSession()
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
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

// CreateEmbedding generates a float32 vector for the given text.
func (e *Embedder) CreateEmbedding(ctx context.Context, text string) ([]float32, error) {
	output, err := e.pipeline.RunPipeline([]string{text})
	if err != nil {
		return nil, err
	}

	if len(output.Embeddings) == 0 {
		return nil, fmt.Errorf("empty embedding output")
	}

	return output.Embeddings[0], nil
}

// Close releases the hugot session resources.
func (e *Embedder) Close() {
	if e.session != nil {
		e.session.Destroy()
	}
}
