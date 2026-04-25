package dataset

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"
)

// DatasetDownloader gerencia o download de datasets do HuggingFace
type DatasetDownloader struct {
	BaseURL    string
	Dataset    string
	Config     string
	Split      string
	OutputFile string
}

// NewDatasetDownloader cria um novo downloader
func NewDatasetDownloader() *DatasetDownloader {
	return &DatasetDownloader{
		BaseURL:    "https://datasets-server.huggingface.co/rows",
		Dataset:    "RichardSakaguchiMS/brazilian-customer-service-conversations",
		Config:     "default",
		Split:      "train",
		OutputFile: "dataset/data/train.txt",
	}
}

// Row representa uma linha do dataset
type Row struct {
	Row map[string]interface{} `json:"row"`
}

// APIResponse representa a resposta da API
type APIResponse struct {
	Rows []Row `json:"rows"`
}

// DownloadDataset baixa o dataset completo e cria arquivo de treinamento
func DownloadDataset() error {
	downloader := NewDatasetDownloader()
	return downloader.Download()
}

// Download executa o download
func (d *DatasetDownloader) Download() error {
	fmt.Println("📥 Baixando dataset completo do HuggingFace...")
	fmt.Println("=")

	allMessages := []map[string]string{}
	offset := 0
	length := 100
	page := 0

	for {
		url := fmt.Sprintf("%s?dataset=%s&config=%s&split=%s&offset=%d&length=%d",
			d.BaseURL, d.Dataset, d.Config, d.Split, offset, length)

		fmt.Printf("⏳ Baixando página %d (offset=%d)...\n", page+1, offset)

		rows, err := d.fetchPage(url)
		if err != nil {
			return fmt.Errorf("erro ao baixar página %d: %v", page+1, err)
		}

		if len(rows) == 0 {
			fmt.Println("✅ Não há mais dados para baixar")
			break
		}

		// Processar mensagens
		for _, row := range rows {
			if messages, ok := row.Row["messages"].([]interface{}); ok {
				for _, msg := range messages {
					if msgMap, ok := msg.(map[string]interface{}); ok {
						role, _ := msgMap["role"].(string)
						content, _ := msgMap["content"].(string)

						if role != "" && content != "" {
							allMessages = append(allMessages, map[string]string{
								"role":    role,
								"content": content,
							})
						}
					}
				}
			}
		}

		fmt.Printf("   ✓ %d conversas baixadas (total: %d mensagens)\n", len(rows), len(allMessages))

		// Verificar se é a última página
		if len(rows) < length {
			fmt.Println("✅ Última página atingida")
			break
		}

		offset += length
		page++

		// Pausa para não sobrecarregar a API
		time.Sleep(500 * time.Millisecond)
	}

	fmt.Println("\n" + "=")
	fmt.Printf("✅ Total de mensagens baixadas: %d\n", len(allMessages))

	// Criar arquivo de treinamento
	if err := d.saveToFile(allMessages); err != nil {
		return fmt.Errorf("erro ao salvar arquivo: %v", err)
	}

	return nil
}

// fetchPage busca uma página da API
func (d *DatasetDownloader) fetchPage(url string) ([]Row, error) {
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("User-Agent", "LMCS-LLM/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var apiResp APIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, err
	}

	return apiResp.Rows, nil
}

// saveToFile salva as mensagens no arquivo de treinamento
func (d *DatasetDownloader) saveToFile(messages []map[string]string) error {
	file, err := os.Create(d.OutputFile)
	if err != nil {
		return err
	}
	defer file.Close()

	userCount := 0
	assistantCount := 0

	for _, msg := range messages {
		role := msg["role"]
		content := msg["content"]

		switch role {
		case "customer", "user", "human":
			fmt.Fprintf(file, "Usuário: %s\n", content)
			userCount++
		case "agent", "assistant", "ai":
			fmt.Fprintf(file, "Assistente: %s\n", content)
			assistantCount++
		}
	}

	// Estatísticas
	fmt.Printf("\n📊 Dataset criado:\n")
	fmt.Printf("   Arquivo: %s\n", d.OutputFile)
	fmt.Printf("   Diálogos Usuário: %d\n", userCount)
	fmt.Printf("   Diálogos Assistente: %d\n", assistantCount)
	fmt.Printf("   Total de linhas: %d\n", userCount+assistantCount)

	// Mostrar exemplos
	fmt.Printf("\n📋 Exemplos do dataset:\n")
	fmt.Println("=")

	// Ler primeiras linhas para mostrar exemplos
	d.showExamples(d.OutputFile, 3)

	// Copiar para livro.txt
	if err := d.copyToLivroTxt(); err != nil {
		log.Printf("Aviso: Não foi possível copiar para livro.txt: %v", err)
	} else {
		fmt.Println("✅ Dataset disponível em:")
		fmt.Println("   - dataset/data/train.txt (principal)")
		fmt.Println("   - livro.txt (cópia compatibilidade)")
	}

	return nil
}

// showExamples mostra exemplos do arquivo
func (d *DatasetDownloader) showExamples(filename string, count int) {
	file, err := os.Open(filename)
	if err != nil {
		return
	}
	defer file.Close()

	reader := make([]byte, 2000)
	n, _ := file.Read(reader)
	content := string(reader[:n])

	fmt.Println(content)
}

// copyToLivroTxt copia o arquivo para livro.txt
func (d *DatasetDownloader) copyToLivroTxt() error {
	src, err := os.Open(d.OutputFile)
	if err != nil {
		return err
	}
	defer src.Close()

	dst, err := os.Create("livro.txt")
	if err != nil {
		return err
	}
	defer dst.Close()

	_, err = io.Copy(dst, src)
	if err != nil {
		return err
	}

	fmt.Println("✅ livro.txt atualizado!")
	return nil
}
