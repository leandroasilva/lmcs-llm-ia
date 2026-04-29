#!/bin/bash
# AMD RX 550 GPU Detection & Compatibility Check for tinygpu

echo "🔍 Verificando compatibilidade da GPU AMD RX 550 com tinygpu"
echo "============================================================"
echo

# 1. Verificar se GPU AMD está conectada
echo "1️⃣ Verificando GPUs conectadas..."
if system_profiler SPDisplaysDataType | grep -q "AMD\|Radeon"; then
    echo "✅ GPU AMD detectada!"
    system_profiler SPDisplaysDataType | grep -A 5 "AMD\|Radeon"
else
    echo "❌ GPU AMD NÃO detectada pelo macOS"
    echo
    echo "Possíveis problemas:"
    echo "  • GPU conectada via USB (não USB4/Thunderbolt)"
    echo "  • Enclosure não é Thunderbolt 3/4"
    echo "  • GPU não está recebendo energia"
    echo
    echo "Verificando dispositivos USB..."
    system_profiler SPUSBDataType | grep -A 3 "USB" | head -20
fi

echo
echo "2️⃣ Arquitetura da AMD RX 550"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "GPU: AMD Radeon RX 550"
echo "Arquitetura: Polaris (GCN 4.0)"
echo "VRAM: 4GB GDDR5"
echo
echo "⚠️ Compatibilidade com tinygpu:"
echo "  • tinygpu suporta: RDNA3+ (RX 7000 series)"
echo "  • Sua GPU: Polaris (RX 500 series)"
echo "  • Status: ❌ NÃO SUPORTADA nativamente"
echo
echo "📊 GPUs AMD Suportadas pelo tinygpu:"
echo "  ✅ RDNA3: RX 7900 XTX, RX 7900 XT, RX 7800 XT"
echo "  ✅ RDNA3: RX 7700 XT, RX 7600"
echo "  ✅ RDNA4: RX 9070 XT, RX 9070"
echo "  ❌ Polaris: RX 580, RX 570, RX 560, RX 550"
echo "  ❌ Vega: Vega 56, Vega 64"
echo "  ❌ RDNA1: RX 5700 XT"
echo "  ❌ RDNA2: RX 6000 series"

echo
echo "3️⃣ Verificando conexão USB/Thunderbolt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if system_profiler SPThunderboltDataType 2>/dev/null | grep -q "Device"; then
    echo "✅ Dispositivos Thunderbolt detectados:"
    system_profiler SPThunderboltDataType | grep "Device" | head -5
else
    echo "❌ Nenhum dispositivo Thunderbolt detectado"
fi

echo
echo "4️⃣ Alternativas"
echo "━━━━━━━━━━━━━━"
echo
echo "Opção A: Usar GPU M2 integrada (Atual)"
echo "  • Status: ✅ Funcionando"
echo "  • Performance: ~16s/epoch"
echo "  • Tempo total: ~1h 20m (300 epochs)"
echo
echo "Opção B: Upgrade para GPU RDNA3+"
echo "  • Exemplos: RX 7600, RX 7700 XT"
echo "  • Custo: ~R$ 2000-4000 + enclosure TB3"
echo "  • Suporte tinygpu: ✅"
echo
echo "Opção C: Usar ROCm no Linux"
echo "  • Boot Linux ou VM"
echo "  • ROCm suporta Polaris (limitado)"
echo "  • Performance: ✅"
echo
echo "Opção D: Google Colab (Grátis)"
echo "  • GPU T4 grátis"
echo "  • Upload do código"
echo "  • Tempo: ~1h 40m"

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Conclusão:"
echo "  • AMD RX 550 (Polaris) ❌ não é compatível com tinygpu"
echo "  • tinygpu requer RDNA3+ (RX 7000+)"
echo "  • Continue usando GPU M2 via PyTorch MPS ✅"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
