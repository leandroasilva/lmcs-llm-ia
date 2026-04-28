package download

import (
	"log"

	"github.com/leandroasilva/lmcs-llm-ia/internal/dataset"
)

func RunDownload(enriched bool) error {
	if enriched {
		log.Println("📥 Modo download de dataset enriquecido com metadados")
		if err := dataset.DownloadEnrichedDataset(); err != nil {
			return err
		}
		log.Println("✅ Dataset enriquecido baixado com sucesso!")
	} else {
		log.Println("📥 Modo download de dataset")
		if err := dataset.DownloadDataset(); err != nil {
			return err
		}
		log.Println("✅ Dataset baixado com sucesso!")
	}
	return nil
}
