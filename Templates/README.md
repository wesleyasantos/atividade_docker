# Templates para Atividade Docker + MLOps

## Estrutura dos Templates

```
Templates/
├── batch-ml/           # Scripts de treino e inferência
│   ├── Dockerfile
│   ├── train.py
│   ├── predict.py
│   └── requirements.txt
├── jupyter/           # Jupyter em container
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── mlflow/            # MLflow tracking
│   └── docker-compose.yml
├── requirements.txt   # Dependências globais
├── .dockerignore      # Arquivos ignorados no build
└── README.md          # Este arquivo
```

## Como usar

1. **Copie os templates** para seu projeto:
   ```bash
   cp -r Aula\ 09/Templates/* atividade-docker-mlops/
   ```

2. **Execute a atividade** seguindo o notebook `Atividade_Docker_MLOps.ipynb`

## Templates disponíveis

### batch-ml/
- **Dockerfile**: Imagem para treino e inferência
- **train.py**: Script de treino (compatível com atividade)
- **predict.py**: Script de inferência (compatível com atividade)
- **requirements.txt**: Dependências ML

### jupyter/
- **Dockerfile**: Imagem do Jupyter
- **docker-compose.yml**: Orquestração do Jupyter
- **requirements.txt**: Dependências Jupyter + ML

### mlflow/
- **docker-compose.yml**: MLflow server

## Compatibilidade

Todos os templates foram ajustados para funcionar com:
- Caminhos `/workspace/` (não `/data/`, `/models/`)
- Dataset sintético da atividade
- MLflow tracking
- Volumes mapeados corretamente
