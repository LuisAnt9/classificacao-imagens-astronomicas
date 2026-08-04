# Classificação de Imagens Astronômicas com Deep Learning

Projeto final da disciplina **Tópicos Especiais em Programação (TEC.1053)** — IFPI Campus Picos.
**Grupo 4 — Tema: Classificação**

## 1. Descrição do Projeto

Este projeto implementa um pipeline completo de classificação de imagens astronômicas em
**6 categorias**: constelações, cosmos, galáxias, nebulosas, planetas e estrelas, utilizando
Redes Neurais Convolucionais (CNN).

Dataset utilizado: [Astronomy Image Classification Dataset](https://www.kaggle.com/datasets/abhikalpsrivastava15/space-images-category) (Kaggle), com aproximadamente 1.100 imagens.

O projeto oferece **duas abordagens** de modelo:

1. **CNN treinada do zero** (`--model scratch`): funciona em qualquer ambiente, inclusive sem
   GPU ou acesso à internet. Treinada localmente em CPU.
2. **Transfer Learning com ResNet18** (`--model transfer`): exige internet (para baixar os pesos
   pré-treinados da ImageNet) e idealmente uma GPU. Treinada no Google Colab (GPU T4) usando o
   notebook `notebooks/transfer_learning_colab.ipynb`, em duas etapas: primeiro com o backbone
   congelado (*linear probe*), depois com fine-tuning das últimas camadas — a versão com melhor
   acurácia entre as três.

Os pesos treinados e os resultados (gráficos, métricas, matriz de confusão) de **ambas** as versões
estão incluídos neste repositório, em `weights/` e `results/`.

## 2. Pré-requisitos

- Python 3.10 ou superior
- pip
- (Opcional, recomendado) GPU com CUDA para treinar mais rápido ou usar transfer learning

## 3. Passo a Passo para Execução

### 3.1. Clonar o repositório

```bash
git clone <URL_DESTE_REPOSITORIO>.git
cd <nome-do-repositorio>
```

### 3.2. Criar ambiente virtual (recomendado)

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3.3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3.4. Obter o dataset

O dataset **não está incluído no repositório** por questão de tamanho. Baixe em:
https://www.kaggle.com/datasets/abhikalpsrivastava15/space-images-category

Após baixar e extrair, organize as imagens na seguinte estrutura dentro da pasta `data/`:

```
data/
├── train/
│   ├── constellations/
│   ├── cosmos/
│   ├── galaxies/
│   ├── nebulae/
│   ├── planets/
│   └── stars/
├── val/
│   └── (mesmas 6 pastas)
└── test/
    └── (mesmas 6 pastas)
```

Um script auxiliar de organização automática do dataset original do Kaggle está disponível em
`src/prepare_data.py` (ver seção 4).

### 3.5. Treinar o modelo

**Opção A — CNN do zero (funciona em qualquer ambiente, inclusive CPU):**

```bash
python src/train.py --model scratch --epochs 15 --batch_size 32 --img_size 96
```

**Opção B — Transfer Learning (recomendado, requer internet/GPU):**

```bash
python src/train.py --model transfer --epochs 15 --batch_size 32 --img_size 128
```

Ou utilize o notebook pronto no Google Colab: `notebooks/transfer_learning_colab.ipynb`.

O treinamento salva automaticamente:
- Os pesos do melhor modelo em `weights/best_model_<tipo>.pt`
- O histórico de treinamento em `results/history_<tipo>.json`
- O gráfico de loss/acurácia por época em `results/training_curves_<tipo>.png`

### 3.6. Avaliar o modelo no conjunto de teste

```bash
python src/evaluate.py --model scratch
```

Gera em `results/`:
- `classification_report_<tipo>.txt` (precisão, recall, F1-score por classe)
- `confusion_matrix_<tipo>.png` (matriz de confusão)
- `sample_predictions_<tipo>.png` (amostras visuais de acertos e erros)

### 3.7. Classificar uma nova imagem

```bash
python src/predict.py --image caminho/para/imagem.jpg --model scratch
```

## 4. Estrutura do Repositório

```
.
├── README.md
├── requirements.txt
├── data/                        # dataset (não incluído — ver seção 3.4)
├── src/
│   ├── dataset.py                # carregamento de dados e data augmentation
│   ├── model.py                  # arquiteturas (CNN do zero e transfer learning)
│   ├── train.py                  # script de treinamento
│   ├── evaluate.py                # avaliação, métricas e matriz de confusão
│   └── predict.py                 # inferência em uma única imagem
├── notebooks/
│   └── transfer_learning_colab.ipynb   # notebook para treinar com GPU no Colab
├── weights/                       # pesos treinados (scratch, transfer e transfer+fine-tuning)
└── results/                        # gráficos, métricas e relatórios gerados
```

## 5. Metodologia

- **Pré-processamento**: redimensionamento das imagens, normalização (estatísticas da ImageNet)
  e data augmentation no conjunto de treino (flip horizontal, rotação, ajuste de brilho/contraste/saturação).
- **Divisão do dataset**: 70% treino / 15% validação / 15% teste, com split estratificado por classe.
- **Arquitetura (CNN do zero)**: 4 blocos convolucionais (Conv2D + BatchNorm + ReLU + MaxPooling),
  seguidos de pooling adaptativo e camadas totalmente conectadas com Dropout para regularização.
- **Arquitetura (Transfer Learning)**: ResNet18 pré-treinada na ImageNet, com backbone congelado
  e a camada final substituída para as 6 classes do problema.
- **Otimização**: Adam, com `ReduceLROnPlateau` para redução automática da taxa de aprendizado.
- **Métricas de avaliação**: acurácia, precisão, recall e F1-score por classe, além de matriz de confusão.

## 6. Resultados Obtidos

Foram treinadas e avaliadas três versões do modelo:

### 6.1. Comparativo geral

| Modelo | Ambiente | Acurácia no teste |
|---|---|---|
| CNN treinada do zero | CPU (sem GPU/internet) | 51,2% |
| Transfer Learning — *linear probe* (backbone congelado) | Google Colab (GPU T4) | 58,7% |
| **Transfer Learning + Fine-tuning** (`layer4` + `fc` destravados) | **Google Colab (GPU T4)** | **63,4%** |
| Acerto aleatório (6 classes) | — | 16,7% |

O melhor resultado veio do **fine-tuning**: partindo do modelo de transfer learning já treinado (camada
final apenas), destravamos o último bloco convolucional da ResNet18 (`layer4`) e a camada final (`fc`),
e os ajustamos por mais 10 épocas com taxa de aprendizado bem menor (1e-5 para `layer4`, 1e-4 para `fc`),
para não destruir o conhecimento já aprendido. Esse ajuste elevou a acurácia de 58,7% para **63,4%** —
um ganho de +4,7 pontos percentuais sobre o transfer learning "simples", e de +12,2 pontos sobre a CNN
treinada do zero.

### 6.2. Resultados por classe — melhor modelo (Transfer Learning + Fine-tuning)

| Classe | Precisão | Recall | F1-score |
|---|---|---|---|
| Planetas | 0,8276 | 0,8889 | **0,8571** (melhor) |
| Constelações | 0,7692 | 0,7143 | 0,7407 |
| Galáxias | 0,6047 | 0,7027 | 0,6500 |
| Nebulosas | 0,6296 | 0,6296 | 0,6296 |
| Estrelas | 0,4444 | 0,4444 | 0,4444 |
| Cosmos | 0,5000 | 0,3846 | **0,4348** (pior) |
| **Acurácia geral** | | | **0,6337** (172 imagens de teste) |

**Análise:** o fine-tuning trouxe ganhos em quase todas as classes, com destaque para *galáxias* (F1 de
0,53 para 0,65) e *nebulosas* (F1 de 0,51 para 0,63). *Planetas* seguiu como a classe mais bem reconhecida
em todas as versões do modelo — possui elementos visuais mais distintivos (formas circulares nítidas).
*Cosmos* permanece a classe mais desafiadora: a matriz de confusão mostra que suas imagens ainda se
confundem com *nebulae*, *stars* e *planets*, refletindo a natureza genérica dessas imagens de "céu
profundo" no dataset, sem um elemento visual único que as diferencie claramente das demais categorias
de céu noturno.

Considerando 6 classes (acerto aleatório ≈ 16,7%), todos os modelos tiveram desempenho consideravelmente
acima do acaso, mesmo com poucos dados por classe (~120–235 imagens). O padrão se repete nas três
versões: *planetas* e *constelações* entre as classes mais bem reconhecidas; *cosmos*, *nebulosas* e
*estrelas* concentrando a maior parte da confusão entre si, por sua alta similaridade visual.

## 7. Dificuldades Encontradas

- **Dados**: dataset pequeno (~1.100 imagens no total, ~150–235 por classe) e obtido via scraping
  do Google Imagens, com imagens de tamanhos, qualidades e proporções muito variadas — algumas
  pouco representativas da classe (ex.: imagens de interface do Google Imagens, thumbnails, etc.),
  o que introduz ruído no treinamento.
- **Similaridade entre classes**: "cosmos", "nebulosas" e "estrelas" apresentam sobreposição visual
  significativa (céu estrelado, nuvens de gás e poeira coloridas), dificultando a distinção mesmo
  para um observador humano em alguns casos.
- **Hardware**: o ambiente de desenvolvimento inicial não tinha GPU nem acesso à internet, o que
  limitou o tamanho de imagem (96×96) e impediu baixar os pesos pré-treinados da ImageNet para a
  versão de transfer learning. Esse modelo foi então treinado separadamente no Google Colab (GPU
  T4), usando o notebook `notebooks/transfer_learning_colab.ipynb` — o treinamento levou poucos
  minutos com GPU, contra ~11 minutos da CNN do zero em CPU para um número semelhante de épocas.
- **Hiperparâmetros**: ajuste do tamanho de imagem, batch size e taxa de aprendizado para equilibrar
  tempo de treinamento (em CPU) e desempenho do modelo. No fine-tuning, foi necessário usar taxas de
  aprendizado bem baixas (1e-5 para as camadas convolucionais destravadas, 1e-4 para a camada final)
  para ajustar o modelo sem destruir o que já havia sido aprendido no treinamento anterior — taxas
  mais altas prejudicavam o desempenho já alcançado.

## 8. Autores — Grupo 4

Luís Antônio Santos, Luis Antônio Souza, Caíque Wesley, Francisco Emerson, Tiago Rodrigues

## 9. Licença/Uso

Projeto acadêmico desenvolvido para a disciplina TEC.1053 — IFPI Campus Picos.
