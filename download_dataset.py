#!/usr/bin/env python3
"""
Script para baixar dataset completo do HuggingFace
Brazilian Customer Service Conversations
"""

import json
import urllib.request
import time

def download_dataset():
    """Baixa todas as páginas do dataset e cria arquivo de treinamento"""
    
    base_url = "https://datasets-server.huggingface.co/rows"
    dataset = "RichardSakaguchiMS/brazilian-customer-service-conversations"
    config = "default"
    split = "train"
    offset = 0
    length = 100
    
    all_conversations = []
    page = 0
    
    print("📥 Baixando dataset completo...")
    print("=" * 60)
    
    while True:
        url = f"{base_url}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
        
        try:
            print(f"⏳ Baixando página {page + 1} (offset={offset})...")
            
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            rows = data.get('rows', [])
            
            if not rows:
                print("✅ Não há mais dados para baixar")
                break
            
            # Processar cada conversa
            for row in rows:
                conversation = row.get('row', {})
                all_conversations.append(conversation)
            
            print(f"   ✓ {len(rows)} conversas baixadas (total: {len(all_conversations)})")
            
            # Verificar se há mais páginas
            if len(rows) < length:
                print("✅ Última página atingida")
                break
            
            offset += length
            page += 1
            
            # Pausa para não sobrecarregar a API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Erro ao baixar página {page + 1}: {e}")
            break
    
    print("\n" + "=" * 60)
    print(f"✅ Total de conversas baixadas: {len(all_conversations)}")
    
    # Criar arquivo de treinamento
    output_file = "conversas.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, conv in enumerate(all_conversations):
            # Dataset usa formato: {"messages": [{"role": "customer/agent", "content": "..."}]}
            messages = conv.get('messages', [])
            
            if messages:
                # Escrever cada mensagem no formato conversacional
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    
                    if role == 'customer':
                        f.write(f"Usuário: {content}\n")
                    elif role == 'agent':
                        f.write(f"Assistente: {content}\n")
                
                f.write("\n")  # Separador entre conversas
    
    # Estatísticas
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        user_lines = sum(1 for line in lines if line.startswith('Usuário:'))
        assistant_lines = sum(1 for line in lines if line.startswith('Assistente:'))
    
    print(f"\n📊 Dataset criado:")
    print(f"   Arquivo: {output_file}")
    print(f"   Diálogos Usuário: {user_lines}")
    print(f"   Diálogos Assistente: {assistant_lines}")
    print(f"   Total de linhas: {len(lines)}")
    
    # Mostrar exemplos
    print(f"\n📋 Exemplos do dataset:")
    print("=" * 60)
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        dialogs = content.strip().split('\n\n')
        for i, dialog in enumerate(dialogs[:3]):
            print(f"\nExemplo {i + 1}:")
            print(dialog)
            print("-" * 60)
    
    return output_file

if __name__ == "__main__":
    download_dataset()
