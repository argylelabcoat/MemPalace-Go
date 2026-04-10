// Package benchmarks provides evaluation metrics and utilities for memory retrieval benchmarks.
package benchmarks

import (
	"math"
)

// RecallAny computes binary recall: did at least one correct item appear in top-K?
func RecallAny(topK []string, correctIDs []string) float64 {
	correctSet := make(map[string]bool)
	for _, id := range correctIDs {
		correctSet[id] = true
	}
	for _, id := range topK {
		if correctSet[id] {
			return 1.0
		}
	}
	return 0.0
}

// RecallAll computes binary recall: did ALL correct items appear in top-K?
func RecallAll(topK []string, correctIDs []string) float64 {
	correctSet := make(map[string]bool)
	for _, id := range correctIDs {
		correctSet[id] = true
	}
	for _, id := range correctIDs {
		found := false
		for _, tid := range topK {
			if tid == id {
				found = true
				break
			}
		}
		if !found {
			return 0.0
		}
	}
	return 1.0
}

// FractionalRecall computes fractional recall: what fraction of evidence items were found?
func FractionalRecall(topK []string, evidenceIDs []string) float64 {
	if len(evidenceIDs) == 0 {
		return 1.0
	}
	topKSet := make(map[string]bool)
	for _, id := range topK {
		topKSet[id] = true
	}
	found := 0
	for _, id := range evidenceIDs {
		if topKSet[id] {
			found++
		}
	}
	return float64(found) / float64(len(evidenceIDs))
}

// NDCG computes Normalized Discounted Cumulative Gain with binary relevance.
func NDCG(topK []string, correctIDs []string, corpusIDs []string, k int) float64 {
	correctSet := make(map[string]bool)
	for _, id := range correctIDs {
		correctSet[id] = true
	}

	// Compute relevances for top-K
	relevances := make([]float64, 0, min(k, len(topK)))
	for i := 0; i < min(k, len(topK)); i++ {
		if correctSet[topK[i]] {
			relevances = append(relevances, 1.0)
		} else {
			relevances = append(relevances, 0.0)
		}
	}

	// DCG
	dcgVal := computeDCG(relevances, k)

	// Ideal DCG
	ideal := make([]float64, len(relevances))
	copy(ideal, relevances)
	// Sort descending
	for i := 0; i < len(ideal); i++ {
		for j := i + 1; j < len(ideal); j++ {
			if ideal[j] > ideal[i] {
				ideal[i], ideal[j] = ideal[j], ideal[i]
			}
		}
	}
	idcgVal := computeDCG(ideal, k)

	if idcgVal == 0 {
		return 0.0
	}
	return dcgVal / idcgVal
}

func computeDCG(relevances []float64, k int) float64 {
	score := 0.0
	for i, rel := range relevances[:min(k, len(relevances))] {
		score += rel / math.Log2(float64(i+2))
	}
	return score
}

// F1Score computes token-level F1 with normalization.
func F1Score(prediction, groundTruth string) float64 {
	predTokens := normalizeTokens(prediction)
	truthTokens := normalizeTokens(groundTruth)

	if len(predTokens) == 0 || len(truthTokens) == 0 {
		if len(predTokens) == len(truthTokens) {
			return 1.0
		}
		return 0.0
	}

	predCount := make(map[string]int)
	for _, t := range predTokens {
		predCount[t]++
	}
	truthCount := make(map[string]int)
	for _, t := range truthTokens {
		truthCount[t]++
	}

	numSame := 0
	for token, count := range predCount {
		if truthCount[token] > 0 {
			numSame += min(count, truthCount[token])
		}
	}

	if numSame == 0 {
		return 0.0
	}

	precision := float64(numSame) / float64(len(predTokens))
	recall := float64(numSame) / float64(len(truthTokens))
	return 2 * precision * recall / (precision + recall)
}

func normalizeTokens(s string) []string {
	// Simple normalization: lowercase, remove punctuation, split
	var result []string
	var current []rune
	for _, r := range s {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' {
			current = append(current, r)
		} else {
			if len(current) > 0 {
				result = append(result, string(current))
				current = nil
			}
		}
	}
	if len(current) > 0 {
		result = append(result, string(current))
	}

	// Lowercase and remove stopwords
	stopwords := map[string]bool{
		"a": true, "an": true, "the": true, "and": true, "or": true,
	}
	var filtered []string
	for _, t := range result {
		t = toLower(t)
		if !stopwords[t] {
			filtered = append(filtered, t)
		}
	}
	return filtered
}

func toLower(s string) string {
	result := make([]rune, len(s))
	for i, r := range s {
		if r >= 'A' && r <= 'Z' {
			result[i] = r + 32
		} else {
			result[i] = r
		}
	}
	return string(result)
}
