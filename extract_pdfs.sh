#!/bin/bash
# Script para extrair texto de todos os PDFs e criar livro.txt combinado

echo "=== Extraindo texto dos PDFs ==="
echo ""

LIVROS_DIR="/Volumes/Dock/workspace/lmcs/lmcs-llm-ia/livros"
OUTPUT_FILE="/Volumes/Dock/workspace/lmcs/lmcs-llm-ia/livro.txt"
TEMP_DIR="/tmp/pdf_extraction"

# Criar diretório temporário
mkdir -p "$TEMP_DIR"

# Limpar arquivos temporários antigos
rm -f "$TEMP_DIR"/*.txt

# Contador
count=0
total_chars=0

# Extrair texto de cada PDF
for pdf in "$LIVROS_DIR"/*.pdf; do
    if [ -f "$pdf" ]; then
        filename=$(basename "$pdf" .pdf)
        echo "Extraindo: $filename.pdf..."
        
        # Extrair texto
        pdftotext -layout "$pdf" "$TEMP_DIR/$filename.txt"
        
        # Verificar se foi extraído texto
        if [ -f "$TEMP_DIR/$filename.txt" ]; then
            chars=$(wc -c < "$TEMP_DIR/$filename.txt")
            echo "  ✓ Extraído: $chars caracteres"
            count=$((count + 1))
            total_chars=$((total_chars + chars))
        else
            echo "  ✗ Falha na extração"
        fi
    fi
done

echo ""
echo "=== Combinando todos os textos ==="

# Criar arquivo combinado
COMBINED_FILE="$TEMP_DIR/combined_all.txt"

# Adicionar livro original primeiro (se existir)
if [ -f "$OUTPUT_FILE" ]; then
    echo "Adicionando livro.txt original..."
    cp "$OUTPUT_FILE" "$COMBINED_FILE"
    original_chars=$(wc -c < "$OUTPUT_FILE")
    echo "  ✓ Livro original: $original_chars caracteres"
else
    echo "" > "$COMBINED_FILE"
fi

# Adicionar textos extraídos dos PDFs
echo "Adicionando textos dos PDFs..."
for txt_file in "$TEMP_DIR"/*.txt; do
    if [ -f "$txt_file" ] && [ "$(basename "$txt_file")" != "combined_all.txt" ]; then
        filename=$(basename "$txt_file")
        echo "" >> "$COMBINED_FILE"
        echo "=== INÍCIO DO ARQUIVO: $filename ===" >> "$COMBINED_FILE"
        echo "" >> "$COMBINED_FILE"
        cat "$txt_file" >> "$COMBINED_FILE"
        echo "" >> "$COMBINED_FILE"
        echo "=== FIM DO ARQUIVO: $filename ===" >> "$COMBINED_FILE"
        echo "" >> "$COMBINED_FILE"
    fi
done

# Copiar para arquivo final
cp "$COMBINED_FILE" "$OUTPUT_FILE"

# Estatísticas finais
final_chars=$(wc -c < "$OUTPUT_FILE")
final_lines=$(wc -l < "$OUTPUT_FILE")

echo ""
echo "=== RESUMO ==="
echo "PDFs processados: $count"
echo "Caracteres extraídos dos PDFs: $total_chars"
echo "Caracteres totais (combinado): $final_chars"
echo "Linhas totais: $final_lines"
echo "Arquivo salvo em: $OUTPUT_FILE"
echo ""

# Limpar diretório temporário
rm -rf "$TEMP_DIR"

echo "=== Concluído! ==="
