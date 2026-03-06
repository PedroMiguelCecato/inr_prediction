======================================================================
    MODELO SALVO: RandomForest
    ======================================================================

    Data de Salvamento: 2026-03-06 02:11:21
    Versão: 20260306_021121

    TIPO: Modelo

    ARQUIVOS:
    - Modelo: RandomForest_20260306_021121_model.pkl
    - Parâmetros: RandomForest_20260306_021121_params.json
    - Metadata: RandomForest_20260306_021121_metadata.json
    

    INFORMAÇÕES DO MODELO:
    - MAE (CV): 0.30450073781904774
    - Features: 12
    - Amostras de Treino: 5022
    

    COMO CARREGAR E USAR:
    ```python
    from training_model import ModelTrainer
    import joblib

    # Opção 1: Carregar apenas o modelo
    model = joblib.load("trained_models\randomforest_full\RandomForest_20260306_021121_model.pkl")
    
    predictions = model.predict(X_test)
    

    # Opção 2: Carregar modelo completo com metadata
    model, metadata = ModelTrainer.load_model_complete(
        base_name="RandomForest_20260306_021121",
        model_dir="trained_models/randomforest_full"
    )
    ```

    ======================================================================