======================================================================
    MODELO SALVO: ElasticNet
    ======================================================================

    Data de Salvamento: 2026-03-06 02:11:28
    Versão: 20260306_021128

    TIPO: Pipeline (Scaler + Modelo)

    ARQUIVOS:
    - Modelo: ElasticNet_20260306_021128_model.pkl
    - Parâmetros: ElasticNet_20260306_021128_params.json
    - Metadata: ElasticNet_20260306_021128_metadata.json
    - Diagnósticos: ElasticNet_20260306_021128_diagnostics.json

    INFORMAÇÕES DO MODELO:
    - MAE (CV): 0.34373433889590005
    - Features: 12
    - Amostras de Treino: 5022
    - Features Selecionadas: 8

    COMO CARREGAR E USAR:
    ```python
    from training_model import ModelTrainer
    import joblib

    # Opção 1: Carregar apenas o modelo
    model = joblib.load("trained_models\elasticnet_full\ElasticNet_20260306_021128_model.pkl")
    
    # ⚠️ IMPORTANTE: Este é um Pipeline (inclui normalização automática)
    # Use dados BRUTOS para predição - o Pipeline normaliza automaticamente!
    predictions = model.predict(X_test)  # X_test em dados BRUTOS
    

    # Opção 2: Carregar modelo completo com metadata
    model, metadata = ModelTrainer.load_model_complete(
        base_name="ElasticNet_20260306_021128",
        model_dir="trained_models/elasticnet_full"
    )
    ```

    ======================================================================