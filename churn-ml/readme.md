# Previsão de Churn de Clientes (Telecom)

Modelo de machine learning que prevê se um cliente de telecom vai cancelar o serviço.

## Dataset
[Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
Baixe o CSV e coloque em `01-churn-ml/data/`.

## Abordagem
- Limpeza dos dados (conversão de `TotalCharges`, tratamento de nulos)
- Pré-processamento com `ColumnTransformer` (StandardScaler + OneHotEncoder)
- Comparação entre Regressão Logística e Random Forest
- Classes balanceadas com `class_weight="balanced"`
- Escolha do melhor modelo pelo F1 da classe "churn"

## Como rodar
```bash
pip install -r requirements.txt
python train.py
```

## Resultados
| Modelo | Accuracy | F1 (churn) |
|---|---|---|
| Regressão Logística | XX | XX |
| Random Forest | XX | XX |

![Matriz de confusão](confusion_matrix.png)

## Stack
Python, pandas, scikit-learn, matplotlib