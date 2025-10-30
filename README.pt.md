# Detector de Fraude

Este projeto implementa um pipeline de Machine Learning que classifica transações de cartão de crédito como "Normais" ou "Fraudulentas", usando Scikit-learn.

## Resultado

Comparamos dois modelos usando Validação Cruzada para avaliar o equilíbrio entre o Precision e o Recall:

A Pipeline com SMOTE + LogisticRegression resultava em um Precision médio extremamente baixo de 6.6%, significando que 93% dos alertas dele foram falsos positivos.

A Pipeline com SMOTE + RandomForest retornou um resultado melhor, com Precision de 85.1% e Recall de 81.9%.

## Instalação

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/ArthurDOli/fraud-detector.git
    cd fraud-detector
    ```

2.  **Crie e ative um ambiente virtual:**

    ```bash
    python -m venv venv
    # No Windows
    venv\Scripts\activate
    # No macOS/Linux
    source venv/bin/activate
    ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Baixe o dataset** em: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

5.  **Mova o arquivo `creditcard.csv`** para a pasta `/data/` na raiz do projeto.

6.  **Execução do Projeto**

Após a configuração, você pode rodar o pipeline principal ou os testes.

1.  **Para rodar o pipeline principal:**

    - Este comando irá carregar os dados, pré-processar, treinar o modelo (SMOTE + RandomForest) e exibir os scores de Validação Cruzada e os scores finais do teste no terminal.

    ```bash
    python src/main.py
    ```

2.  **Para rodar os testes unitários:**
    - Este comando executará todos os testes (`test_*.py`) localizados na pasta `/tests/` usando `pytest`.
    ```bash
    pytest
    ```

## Estrutura do Projeto

```bash
/fraud-detector
├── /notebooks/
|   └── fraud.ipynb
├── /src/
|   ├── data_processing.py
|   └── evaluation.py
|   └── main.py
|   └── model_training.py
├── /tests/
|   ├── test_data_processing.py
|   ├── test_evaluation.py
|   ├── test_model_training.py
```

- notebooks/: Contém os Jupyter Notebooks usados para análise de dados inicial e experimentação.
- src/: Contém toda a lógica do pipeline.
  - data_processing.py: Funções para carregar e pré-processar os dados.
  - model_training.py: Funções para dividir os dados (`split_data`) e criar os pipelines de modelo (ex: `create_smote_rf_pipeline`).
  - evaluation.py: Funções para avaliação (`get_cv_scores`, `get_final_metrics`).
  - main.py: O orquestrador que chama todas as funções na ordem correta.
- tests/: Contém os testes unitários feitos com PyTest.
