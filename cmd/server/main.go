// Package server provides the MCP server implementation for mempalace-go.
// It runs as a stdio-based MCP server exposing memory tools to AI clients.
package server

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/argylelabcoat/mempalace-go/internal/config"
	"github.com/argylelabcoat/mempalace-go/internal/diary"
	"github.com/argylelabcoat/mempalace-go/internal/embedder"
	"github.com/argylelabcoat/mempalace-go/internal/kg"
	"github.com/argylelabcoat/mempalace-go/internal/layers"
	"github.com/argylelabcoat/mempalace-go/internal/palace"
	"github.com/argylelabcoat/mempalace-go/internal/sanitizer"
	"github.com/argylelabcoat/mempalace-go/internal/search"
	"github.com/argylelabcoat/mempalace-go/pkg/mcp"
	"github.com/argylelabcoat/mempalace-go/pkg/wal"
	govector "github.com/argylelabcoat/mempalace-go/storage/govector"
)

func runServer(cmd *cobra.Command, args []string) error {
	ctx := context.Background()
	cfg, err := config.Load("")
	if err != nil {
		return fmt.Errorf("config: %w", err)
	}

	modelsDir := cfg.GetModelsDir()
	emb, err := embedder.New("", modelsDir)
	if err != nil {
		return fmt.Errorf("embedder: %w", err)
	}
	defer emb.Close()

	palacePath := os.ExpandEnv(cfg.PalacePath)
	if palacePath == cfg.PalacePath && strings.HasPrefix(cfg.PalacePath, "~") {
		home, _ := os.UserHomeDir()
		palacePath = strings.Replace(cfg.PalacePath, "~", home, 1)
	}

	// all-MiniLM-L6-v2 produces 384-dimensional embeddings
	vectorDB, err := govector.NewStore(palacePath+"/vectors.db", 384)
	if err != nil {
		return fmt.Errorf("vector store: %w", err)
	}

	kgDB, err := kg.New(palacePath + "/knowledge_graph.sqlite3")
	if err != nil {
		return fmt.Errorf("knowledge graph: %w", err)
	}
	defer kgDB.Close()

	searcher := search.NewSearcher(vectorDB, emb)
	stack := layers.NewMemoryStack(cfg, searcher)

	walInstance, err := wal.NewWAL(palacePath)
	if err != nil {
		return fmt.Errorf("wal: %w", err)
	}

	taxonomy, err := searcher.GetTaxonomy(ctx)
	if err != nil {
		return fmt.Errorf("taxonomy: %w", err)
	}

	palaceGraph := palace.NewGraph()
	for wingName, wingNode := range taxonomy {
		for roomName, roomNode := range wingNode.Rooms {
			for i := 0; i < roomNode.Count; i++ {
				palaceGraph.AddDrawer(wingName, roomName, "")
			}
		}
	}
	palaceGraph.BuildEdges()

	agentDiary, err := diary.New(palacePath)
	if err != nil {
		return fmt.Errorf("diary: %w", err)
	}

	server := mcp.NewServer(bufio.NewReader(os.Stdin), bufio.NewWriter(os.Stdout), bufio.NewWriter(os.Stderr))

	if err := server.Initialize(ctx, stack, searcher, kgDB); err != nil {
		return fmt.Errorf("initialize server: %w", err)
	}
	defer server.Shutdown(ctx)

	registerTools(server, stack, searcher, kgDB, walInstance, palacePath, palaceGraph, agentDiary)

	return server.Run()
}

func NewCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "server",
		Short: "Run the MCP memory server",
		RunE:  runServer,
	}
	return cmd
}

func registerTools(server *mcp.Server, stack *layers.MemoryStack, searcher *search.Searcher, kgDB *kg.KnowledgeGraph, walInstance *wal.WAL, palacePath string, palaceGraph *palace.Graph, agentDiary *diary.Diary) {
	server.RegisterTool("mempalace_search", "Search memories", mcp.SearchToolSchema, func(params map[string]any) (any, error) {
		query, _ := params["query"].(string)
		wing, _ := params["wing"].(string)
		room, _ := params["room"].(string)

		ctx := context.Background()
		results, err := stack.Search(ctx, query, wing, room, 10)
		if err != nil {
			return nil, err
		}

		var lines []string
		for _, r := range results {
			lines = append(lines, fmt.Sprintf("[%s/%s] %s", r.Wing, r.Room, r.Content))
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: fmt.Sprintf("Found %d results:\n%s", len(results), strings.Join(lines, "\n"))}},
		}, nil
	})

	server.RegisterTool("mempalace_wake", "Wake up memory with wing context", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"wing": map[string]any{"type": "string"},
		},
	}), func(params map[string]any) (any, error) {
		wing, _ := params["wing"].(string)
		ctx := context.Background()
		text, err := stack.WakeUp(ctx, wing)
		if err != nil {
			return nil, err
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: text}},
		}, nil
	})

	server.RegisterTool("mempalace_recall", "Recall memories from wing/room", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"wing":  map[string]any{"type": "string"},
			"room":  map[string]any{"type": "string"},
			"count": map[string]any{"type": "integer", "default": 10},
		},
	}), func(params map[string]any) (any, error) {
		wing, _ := params["wing"].(string)
		room, _ := params["room"].(string)
		count, _ := params["count"].(int)
		if count == 0 {
			count = 10
		}
		ctx := context.Background()
		text, err := stack.Recall(ctx, wing, room, count)
		if err != nil {
			return nil, err
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: text}},
		}, nil
	})

	server.RegisterTool("mempalace_kg_query", "Query knowledge graph", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"entity":    map[string]any{"type": "string"},
			"as_of":     map[string]any{"type": "string"},
			"direction": map[string]any{"type": "string", "default": "outgoing"},
		},
	}), func(params map[string]any) (any, error) {
		entity, _ := params["entity"].(string)
		asOf, _ := params["as_of"].(string)
		direction, _ := params["direction"].(string)

		results, err := kgDB.QueryEntity(entity, asOf, direction)
		if err != nil {
			return nil, err
		}

		var lines []string
		for _, r := range results {
			lines = append(lines, fmt.Sprintf("%s -> %s (%s)", r.Predicate, r.Object, r.ValidFrom))
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: strings.Join(lines, "\n")}},
		}, nil
	})

	server.RegisterTool("mempalace_status", "Get palace status and overview", mcp.SchemaToJSON(map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}), func(params map[string]any) (any, error) {
		ctx := context.Background()
		taxonomy, err := searcher.GetTaxonomy(ctx)
		if err != nil {
			return nil, err
		}

		totalDrawers := 0
		var wingLines []string
		for wingName, wingNode := range taxonomy {
			totalDrawers += wingNode.Count
			roomCount := len(wingNode.Rooms)
			wingLines = append(wingLines, fmt.Sprintf("  %s: %d drawers, %d rooms", wingName, wingNode.Count, roomCount))
		}

		text := fmt.Sprintf("Palace Status:\n"+
			"  Total drawers: %d\n"+
			"  Total wings: %d\n\n"+
			"Wings:\n%s",
			totalDrawers, len(taxonomy), strings.Join(wingLines, "\n"))

		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: text}},
		}, nil
	})

	server.RegisterTool("mempalace_list_wings", "List all wings with drawer counts", mcp.SchemaToJSON(map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}), func(params map[string]any) (any, error) {
		ctx := context.Background()
		wings, err := searcher.ListWings(ctx)
		if err != nil {
			return nil, err
		}

		var lines []string
		for _, w := range wings {
			lines = append(lines, fmt.Sprintf("%s: %d drawers", w.Name, w.DrawerCount))
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: fmt.Sprintf("Wings (%d):\n%s", len(wings), strings.Join(lines, "\n"))}},
		}, nil
	})

	server.RegisterTool("mempalace_list_rooms", "List rooms within a wing (or all rooms)", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"wing": map[string]any{"type": "string"},
		},
	}), func(params map[string]any) (any, error) {
		wing, _ := params["wing"].(string)
		ctx := context.Background()
		rooms, err := searcher.ListRooms(ctx, wing)
		if err != nil {
			return nil, err
		}

		var lines []string
		for _, r := range rooms {
			lines = append(lines, fmt.Sprintf("%s/%s: %d drawers", r.Wing, r.Name, r.DrawerCount))
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: fmt.Sprintf("Rooms (%d):\n%s", len(rooms), strings.Join(lines, "\n"))}},
		}, nil
	})

	server.RegisterTool("mempalace_get_taxonomy", "Get full wing -> room -> count tree structure", mcp.SchemaToJSON(map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}), func(params map[string]any) (any, error) {
		ctx := context.Background()
		taxonomy, err := searcher.GetTaxonomy(ctx)
		if err != nil {
			return nil, err
		}

		var lines []string
		for wingName, wingNode := range taxonomy {
			lines = append(lines, fmt.Sprintf("%s (%d)", wingName, wingNode.Count))
			for roomName, roomNode := range wingNode.Rooms {
				lines = append(lines, fmt.Sprintf("  %s: %d", roomName, roomNode.Count))
			}
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: strings.Join(lines, "\n")}},
		}, nil
	})

	server.RegisterTool("mempalace_check_duplicate", "Check if content already exists", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"content": map[string]any{"type": "string"},
			"wing":    map[string]any{"type": "string"},
			"room":    map[string]any{"type": "string"},
		},
		"required": []string{"content"},
	}), func(params map[string]any) (any, error) {
		content, _ := params["content"].(string)
		wing, _ := params["wing"].(string)
		room, _ := params["room"].(string)

		ctx := context.Background()
		results, err := stack.Search(ctx, content, wing, room, 5)
		if err != nil {
			return nil, err
		}

		var duplicates []string
		for _, r := range results {
			if len(r.Content) > 0 && similarContent(content, r.Content) {
				duplicates = append(duplicates, fmt.Sprintf("[%s/%s] ID: %s", r.Wing, r.Room, r.ID))
			}
		}

		if len(duplicates) > 0 {
			return mcp.ToolCallResult{
				Content: []mcp.ToolContent{{Type: "text", Text: fmt.Sprintf("Possible duplicates found:\n%s", strings.Join(duplicates, "\n"))}},
			}, nil
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: "No duplicates found. Content appears to be unique."}},
		}, nil
	})

	server.RegisterTool("mempalace_add_drawer", "Add content to a wing/room", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"content": map[string]any{"type": "string"},
			"wing":    map[string]any{"type": "string"},
			"room":    map[string]any{"type": "string"},
			"source":  map[string]any{"type": "string"},
		},
		"required": []string{"content", "wing", "room"},
	}), func(params map[string]any) (any, error) {
		content, _ := params["content"].(string)
		wing, _ := params["wing"].(string)
		room, _ := params["room"].(string)
		source, _ := params["source"].(string)

		// Sanitize inputs
		var err error
		wing, err = sanitizer.SanitizeName(wing, "wing")
		if err != nil {
			return nil, err
		}
		room, err = sanitizer.SanitizeName(room, "room")
		if err != nil {
			return nil, err
		}
		content, err = sanitizer.SanitizeContent(content, "content")
		if err != nil {
			return nil, err
		}

		ctx := context.Background()
		drawer := palace.Drawer{
			Content:    content,
			Wing:       wing,
			Room:       room,
			SourceFile: source,
		}

		if err := searcher.Store(ctx, drawer); err != nil {
			return nil, err
		}

		if err := walInstance.LogAdd(wal.Entry{
			DrawerID: drawer.ID,
			Wing:     wing,
			Room:     room,
			Content:  content,
		}); err != nil {
			return nil, err
		}

		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: fmt.Sprintf("Successfully added drawer to %s/%s", wing, room)}},
		}, nil
	})

	server.RegisterTool("mempalace_delete_drawer", "Delete a drawer by ID", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"id": map[string]any{"type": "string"},
		},
		"required": []string{"id"},
	}), func(params map[string]any) (any, error) {
		id, _ := params["id"].(string)

		ctx := context.Background()
		if err := searcher.Delete(ctx, id); err != nil {
			return nil, err
		}

		if err := walInstance.LogDelete(id); err != nil {
			return nil, err
		}

		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: fmt.Sprintf("Successfully deleted drawer %s", id)}},
		}, nil
	})

	server.RegisterTool("mempalace_kg_add", "Add fact to knowledge graph", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"subject":    map[string]any{"type": "string"},
			"predicate":  map[string]any{"type": "string"},
			"object":     map[string]any{"type": "string"},
			"valid_from": map[string]any{"type": "string"},
			"valid_to":   map[string]any{"type": "string"},
			"confidence": map[string]any{"type": "number"},
		},
		"required": []string{"subject", "predicate", "object"},
	}), func(params map[string]any) (any, error) {
		subject, _ := params["subject"].(string)
		predicate, _ := params["predicate"].(string)
		obj, _ := params["object"].(string)
		validFrom, _ := params["valid_from"].(string)
		validTo, _ := params["valid_to"].(string)
		confidence := 1.0
		if c, ok := params["confidence"].(float64); ok {
			confidence = c
		}

		tripleID, err := kgDB.AddTriple(subject, predicate, obj, validFrom, validTo, confidence)
		if err != nil {
			return nil, err
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: fmt.Sprintf("Added triple: %s", tripleID)}},
		}, nil
	})

	server.RegisterTool("mempalace_kg_invalidate", "Mark facts as ended", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"subject":   map[string]any{"type": "string"},
			"predicate": map[string]any{"type": "string"},
			"object":    map[string]any{"type": "string"},
			"valid_to":  map[string]any{"type": "string"},
		},
		"required": []string{"subject", "predicate", "object", "valid_to"},
	}), func(params map[string]any) (any, error) {
		subject, _ := params["subject"].(string)
		predicate, _ := params["predicate"].(string)
		obj, _ := params["object"].(string)
		validTo, _ := params["valid_to"].(string)

		if err := kgDB.Invalidate(subject, predicate, obj, validTo); err != nil {
			return nil, err
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: "Fact invalidated successfully"}},
		}, nil
	})

	server.RegisterTool("mempalace_kg_timeline", "Get chronological entity story", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"entity": map[string]any{"type": "string"},
		},
		"required": []string{"entity"},
	}), func(params map[string]any) (any, error) {
		entity, _ := params["entity"].(string)

		entries, err := kgDB.Timeline(entity)
		if err != nil {
			return nil, err
		}

		var lines []string
		for _, e := range entries {
			line := fmt.Sprintf("%s: %s -> %s", e.ValidFrom, e.Predicate, e.Object)
			if e.ValidTo != "" {
				line += fmt.Sprintf(" (until %s)", e.ValidTo)
			}
			lines = append(lines, line)
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: fmt.Sprintf("Timeline for %s:\n%s", entity, strings.Join(lines, "\n"))}},
		}, nil
	})

	server.RegisterTool("mempalace_kg_stats", "Get knowledge graph statistics", mcp.SchemaToJSON(map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}), func(params map[string]any) (any, error) {
		stats, err := kgDB.Stats()
		if err != nil {
			return nil, err
		}

		var lines []string
		lines = append(lines, fmt.Sprintf("Entities: %d", stats.EntityCount))
		lines = append(lines, fmt.Sprintf("Triples: %d", stats.TripleCount))
		lines = append(lines, "Relationship types:")
		for pred, count := range stats.RelationshipTypes {
			lines = append(lines, fmt.Sprintf("  %s: %d", pred, count))
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: strings.Join(lines, "\n")}},
		}, nil
	})

	server.RegisterTool("mempalace_traverse", "Walk the palace graph from a room", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"room":     map[string]any{"type": "string"},
			"max_hops": map[string]any{"type": "integer", "default": 3},
		},
		"required": []string{"room"},
	}), func(params map[string]any) (any, error) {
		room, _ := params["room"].(string)
		maxHops := 3
		if h, ok := params["max_hops"].(float64); ok {
			maxHops = int(h)
		}

		results := palaceGraph.Traverse(room, maxHops)
		if results == nil {
			return mcp.ToolCallResult{
				Content: []mcp.ToolContent{{Type: "text", Text: fmt.Sprintf("Room '%s' not found", room)}},
			}, nil
		}

		var lines []string
		for _, r := range results {
			wings := r["wings"].([]string)
			line := fmt.Sprintf("%s (hops: %d, wings: %s)", r["room"], r["hop"], strings.Join(wings, ", "))
			lines = append(lines, line)
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: fmt.Sprintf("Traversal from '%s':\n%s", room, strings.Join(lines, "\n"))}},
		}, nil
	})

	server.RegisterTool("mempalace_find_tunnels", "Find rooms bridging two wings", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"wing_a": map[string]any{"type": "string"},
			"wing_b": map[string]any{"type": "string"},
		},
	}), func(params map[string]any) (any, error) {
		wingA, _ := params["wing_a"].(string)
		wingB, _ := params["wing_b"].(string)

		tunnels := palaceGraph.FindTunnels(wingA, wingB)
		if len(tunnels) == 0 {
			return mcp.ToolCallResult{
				Content: []mcp.ToolContent{{Type: "text", Text: "No tunnels found"}},
			}, nil
		}

		var lines []string
		for _, t := range tunnels {
			lines = append(lines, fmt.Sprintf("  %s", t))
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: fmt.Sprintf("Tunnels (%d):\n%s", len(tunnels), strings.Join(lines, "\n"))}},
		}, nil
	})

	server.RegisterTool("mempalace_graph_stats", "Get palace graph connectivity", mcp.SchemaToJSON(map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}), func(params map[string]any) (any, error) {
		nodes := palaceGraph.GetNodes()
		edges := palaceGraph.GetEdges()
		rooms := palaceGraph.GetRooms()
		tunnels := palaceGraph.FindTunnels("", "")

		var lines []string
		lines = append(lines, fmt.Sprintf("Nodes: %d", len(nodes)))
		lines = append(lines, fmt.Sprintf("Edges: %d", len(edges)))
		lines = append(lines, fmt.Sprintf("Rooms: %d", len(rooms)))
		lines = append(lines, fmt.Sprintf("Tunnels: %d", len(tunnels)))
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: strings.Join(lines, "\n")}},
		}, nil
	})

	server.RegisterTool("mempalace_diary_write", "Write AAAK diary entry for a specialist agent", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"agent":   map[string]any{"type": "string"},
			"wing":    map[string]any{"type": "string"},
			"content": map[string]any{"type": "string"},
		},
		"required": []string{"agent", "content"},
	}), func(params map[string]any) (any, error) {
		agent, _ := params["agent"].(string)
		wing, _ := params["wing"].(string)
		content, _ := params["content"].(string)

		entry := diary.Entry{
			Agent:   agent,
			Wing:    wing,
			Content: content,
		}

		if err := agentDiary.Write(entry); err != nil {
			return nil, err
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: fmt.Sprintf("Diary entry written for agent: %s", agent)}},
		}, nil
	})

	server.RegisterTool("mempalace_diary_read", "Read recent diary entries", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"agent": map[string]any{"type": "string"},
			"wing":  map[string]any{"type": "string"},
			"limit": map[string]any{"type": "integer", "default": 10},
			"hours": map[string]any{"type": "integer", "default": 24},
		},
	}), func(params map[string]any) (any, error) {
		agent, _ := params["agent"].(string)
		wing, _ := params["wing"].(string)
		limit := 10
		if l, ok := params["limit"].(float64); ok {
			limit = int(l)
		}
		hours := 24
		if h, ok := params["hours"].(float64); ok {
			hours = int(h)
		}

		since := time.Now().Add(-time.Duration(hours) * time.Hour)
		entries, err := agentDiary.Read(agent, wing, limit, since)
		if err != nil {
			return nil, err
		}

		if len(entries) == 0 {
			return mcp.ToolCallResult{
				Content: []mcp.ToolContent{{Type: "text", Text: "No diary entries found"}},
			}, nil
		}

		var lines []string
		for _, e := range entries {
			line := fmt.Sprintf("[%s] %s/%s: %s",
				e.Timestamp.Format("2006-01-02 15:04"),
				e.Agent,
				e.Wing,
				e.Content)
			lines = append(lines, line)
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: fmt.Sprintf("Diary entries (%d):\n%s", len(entries), strings.Join(lines, "\n"))}},
		}, nil
	})

	server.RegisterTool("mempalace_get_aaak_spec", "Get AAAK dialect reference specification", mcp.SchemaToJSON(map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}), func(params map[string]any) (any, error) {
		spec := `# AAAK Dialect Specification

AAAK (Aphantix Abstraction Annotating Kit) is a lossy structured symbolic summary format.

## Format
0:ENTITIES|topics|"key_quote"|emotions|flags

## Entity Codes
3-letter uppercase codes derived from names (e.g., KAI, MAX, PRI)
Multiple entities joined with + (e.g., KAI+PRI)

## Topics
Frequency-based extraction with proper noun and technical term boosting
Max 3 topics, joined with _

## Key Quote
Extracted from emotional quotes or key sentences (max 55 chars)

## Emotion Codes
vul (vulnerability), joy, fear, trust, grief, wonder, rage, love, hope, despair, peace, humor, tender, raw, doubt, relief, anx (anxiety), exhaust, convict, passion, warmth, curious, grat, frust, confuse, satis, excite, determ, surprise

## Flag Codes
DECISION, ORIGIN, CORE, PIVOT, TECHNICAL

## Example
0:KAI+PRI|backend_auth|"we decided to switch"|determ|DECISION_TECHNICAL`

		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: spec}},
		}, nil
	})

	server.RegisterTool("mempalace_deep_search", "L3 deep semantic search with full results", mcp.SchemaToJSON(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{"type": "string"},
			"wing":  map[string]any{"type": "string"},
			"room":  map[string]any{"type": "string"},
			"count": map[string]any{"type": "integer", "default": 20},
		},
		"required": []string{"query"},
	}), func(params map[string]any) (any, error) {
		query, _ := params["query"].(string)
		wing, _ := params["wing"].(string)
		room, _ := params["room"].(string)
		count := 20
		if c, ok := params["count"].(float64); ok {
			count = int(c)
		}

		ctx := context.Background()
		text, err := stack.DeepSearchFormatted(ctx, query, wing, room, count)
		if err != nil {
			return nil, err
		}
		return mcp.ToolCallResult{
			Content: []mcp.ToolContent{{Type: "text", Text: text}},
		}, nil
	})
}

func similarContent(content1, content2 string) bool {
	content1 = strings.ToLower(content1)
	content2 = strings.ToLower(content2)

	if len(content1) > 200 {
		content1 = content1[:200]
	}
	if len(content2) > 200 {
		content2 = content2[:200]
	}

	return strings.Contains(content1, content2[:min(50, len(content2))]) ||
		strings.Contains(content2, content1[:min(50, len(content1))])
}
