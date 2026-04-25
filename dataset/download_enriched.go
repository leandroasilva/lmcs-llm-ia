package dataset

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"sort"
	"time"
)

// ConversationWithMetadata representa uma conversa com metadados
type ConversationWithMetadata struct {
	Messages  []Message              `json:"messages"`
	Metadata  map[string]interface{} `json:"metadata"`
	Intent    string
	Sentiment string
	Sector    string
	Turns     int
}

// Message representa uma mensagem
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// EnrichedDownloader downloader com metadados
type EnrichedDownloader struct {
	BaseURL    string
	Dataset    string
	Config     string
	Split      string
	OutputFile string
}

// NewEnrichedDownloader cria novo downloader
func NewEnrichedDownloader() *EnrichedDownloader {
	return &EnrichedDownloader{
		BaseURL:    "https://datasets-server.huggingface.co/rows",
		Dataset:    "RichardSakaguchiMS/brazilian-customer-service-conversations",
		Config:     "default",
		Split:      "train",
		OutputFile: "dataset/data/train_enriched.txt",
	}
}

// DownloadWithMetadata baixa dataset COM metadados completos
func (d *EnrichedDownloader) DownloadWithMetadata() error {
	fmt.Println("📦 Baixando dataset enriquecido com metadados...")
	fmt.Println("=")

	allConversations := []ConversationWithMetadata{}
	offset := 0
	length := 100
	page := 0

	for {
		url := fmt.Sprintf("%s?dataset=%s&config=%s&split=%s&offset=%d&length=%d",
			d.BaseURL, d.Dataset, d.Config, d.Split, offset, length)

		fmt.Printf("⏳ Baixando página %d (offset=%d)...\n", page+1, offset)

		conversations, err := d.fetchPage(url)
		if err != nil {
			return fmt.Errorf("erro ao baixar página %d: %v", page+1, err)
		}

		if len(conversations) == 0 {
			fmt.Println("✅ Não há mais dados para baixar")
			break
		}

		allConversations = append(allConversations, conversations...)

		fmt.Printf("   ✓ %d conversas baixadas (total: %d)\n", len(conversations), len(allConversations))

		// Verificar se é a última página
		if len(conversations) < length {
			fmt.Println("✅ Última página atingida")
			break
		}

		offset += length
		page++

		// Pausa para não sobrecarregar a API
		time.Sleep(500 * time.Millisecond)
	}

	fmt.Println("\n" + "=")
	fmt.Printf("✅ Total de conversas baixadas: %d\n", len(allConversations))

	// Estatísticas por metadados
	d.printStatistics(allConversations)

	// Criar arquivo de treinamento enriquecido
	if err := d.saveEnrichedToFile(allConversations); err != nil {
		return fmt.Errorf("erro ao salvar arquivo: %v", err)
	}

	// Mostrar exemplo
	d.showExample()

	return nil
}

// fetchPage busca uma página de dados
func (d *EnrichedDownloader) fetchPage(url string) ([]ConversationWithMetadata, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var apiResp APIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, err
	}

	conversations := []ConversationWithMetadata{}

	for _, row := range apiResp.Rows {
		conv := ConversationWithMetadata{
			Metadata: make(map[string]interface{}),
		}

		// Extrair metadados
		if metadata, ok := row.Row["metadata"].(map[string]interface{}); ok {
			conv.Metadata = metadata
			if intent, ok := metadata["intent"].(string); ok {
				conv.Intent = intent
			}
			if sentiment, ok := metadata["sentiment"].(string); ok {
				conv.Sentiment = sentiment
			}
			if sector, ok := metadata["sector"].(string); ok {
				conv.Sector = sector
			}
			if turns, ok := metadata["turns"].(float64); ok {
				conv.Turns = int(turns)
			}
		}

		// Extrair mensagens
		if messages, ok := row.Row["messages"].([]interface{}); ok {
			for _, msg := range messages {
				if msgMap, ok := msg.(map[string]interface{}); ok {
					role, _ := msgMap["role"].(string)
					content, _ := msgMap["content"].(string)

					if role != "" && content != "" {
						conv.Messages = append(conv.Messages, Message{
							Role:    role,
							Content: content,
						})
					}
				}
			}
		}

		if len(conv.Messages) > 0 {
			conversations = append(conversations, conv)
		}
	}

	return conversations, nil
}

// printStatistics exibe estatísticas dos metadados
func (d *EnrichedDownloader) printStatistics(conversations []ConversationWithMetadata) {
	intents := make(map[string]int)
	sentiments := make(map[string]int)
	sectors := make(map[string]int)

	for _, conv := range conversations {
		intents[conv.Intent]++
		sentiments[conv.Sentiment]++
		sectors[conv.Sector]++
	}

	fmt.Printf("\n📊 Distribuição de Intents:\n")
	d.printTopMap(intents, 10)

	fmt.Printf("\n📊 Distribuição de Sentiments:\n")
	d.printSortedMap(sentiments)

	fmt.Printf("\n📊 Distribuição de Sectors:\n")
	d.printSortedMap(sectors)
}

func (d *EnrichedDownloader) printTopMap(data map[string]int, top int) {
	type kv struct {
		Key   string
		Value int
	}

	var sorted []kv
	for k, v := range data {
		sorted = append(sorted, kv{k, v})
	}

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Value > sorted[j].Value
	})

	for i := 0; i < len(sorted) && i < top; i++ {
		fmt.Printf("   %s: %d\n", sorted[i].Key, sorted[i].Value)
	}
}

func (d *EnrichedDownloader) printSortedMap(data map[string]int) {
	type kv struct {
		Key   string
		Value int
	}

	var sorted []kv
	for k, v := range data {
		sorted = append(sorted, kv{k, v})
	}

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Value > sorted[j].Value
	})

	for _, item := range sorted {
		fmt.Printf("   %s: %d\n", item.Key, item.Value)
	}
}

// saveEnrichedToFile salva as conversas com metadados
func (d *EnrichedDownloader) saveEnrichedToFile(conversations []ConversationWithMetadata) error {
	file, err := os.Create(d.OutputFile)
	if err != nil {
		return err
	}
	defer file.Close()

	userCount := 0
	assistantCount := 0
	metadataCount := 0

	for _, conv := range conversations {
		// Escrever metadados
		fmt.Fprintf(file, "[INTENT:%s] [SENTIMENT:%s] [SECTOR:%s]\n",
			conv.Intent, conv.Sentiment, conv.Sector)
		metadataCount++

		// Escrever mensagens
		for _, msg := range conv.Messages {
			switch msg.Role {
			case "customer", "user", "human":
				fmt.Fprintf(file, "Usuário: %s\n", msg.Content)
				userCount++
			case "agent", "assistant", "ai":
				fmt.Fprintf(file, "Assistente: %s\n", msg.Content)
				assistantCount++
			}
		}

		// Separador entre conversas
		fmt.Fprintln(file)
	}

	fmt.Printf("\n📊 Dataset enriquecido criado:\n")
	fmt.Printf("   Arquivo: %s\n", d.OutputFile)
	fmt.Printf("   Conversas: %d\n", metadataCount)
	fmt.Printf("   Diálogos Usuário: %d\n", userCount)
	fmt.Printf("   Diálogos Assistente: %d\n", assistantCount)
	fmt.Printf("   Total de linhas: %d\n", metadataCount+userCount+assistantCount)

	return nil
}

// showExample mostra exemplo do formato
func (d *EnrichedDownloader) showExample() {
	fmt.Printf("\n📋 Exemplo do formato enriquecido:\n")
	fmt.Println("=")

	file, err := os.Open(d.OutputFile)
	if err != nil {
		return
	}
	defer file.Close()

	// Ler primeiras 15 linhas
	buf := make([]byte, 1024)
	n, _ := file.Read(buf)
	content := string(buf[:n])

	// Contar linhas e mostrar até 15
	lines := 0
	for _, line := range content {
		if line == '\n' {
			lines++
			if lines >= 15 {
				break
			}
		}
	}

	fmt.Println(content)
	fmt.Println("=")
}

// DownloadEnrichedDataset exporta função para download com metadados
func DownloadEnrichedDataset() error {
	downloader := NewEnrichedDownloader()
	return downloader.DownloadWithMetadata()
}
