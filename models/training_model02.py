import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from typing import Dict, Tuple, Optional, Any, Union, List
from pathlib import Path
from scipy.stats import ks_2samp

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

# Optuna
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Classe para treinamento automatizado de modelos com otimização Optuna.
    
    ✅ VERSÃO CORRIGIDA - Principais melhorias:
    - Penalização de variância corrigida (era invertida)
    - Validação de dados aprimorada (NaNs, infinitos, variância zero)
    - Método de diagnóstico completo adicionado
    - Logging detalhado de overfitting
    - Detecção de data leakage
    
    Modelos Suportados:
    - XGBoost (Gradient Boosting)
    - LightGBM (Gradient Boosting)
    - Random Forest (Ensemble)
    - ElasticNet (Regressão Linear Regularizada)
    
    Exemplo de uso:
        trainer = ModelTrainer(X_train, y_train, random_state=42, n_splits=5)
        
        # Treinar modelo
        best_params, model, study = trainer.train_xgboost(n_trials=100)
        
        # Diagnosticar
        diagnostics = trainer.diagnose_model(model, X_test, y_test, "XGBoost")
        
        # Comparar todos
        trainer.compare_all_models()
    """
    
    def __init__(self, 
                 X_train: pd.DataFrame, 
                 y_train: pd.Series,
                 random_state: int = 42,
                 n_splits: int = 5,
                 verbose: bool = True):
        """
        Inicializa o treinador de modelos.
        
        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            random_state: Seed para reprodutibilidade
            n_splits: Número de folds para TimeSeriesSplit
            verbose: Se True, mostra logs detalhados
        """
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
        self.n_splits = n_splits
        self.verbose = verbose
        
        # Configurar TimeSeriesSplit
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Histórico de treinamentos
        self.training_history = []
        
        # Armazenar modelos treinados
        self.trained_models = {}
        
        # Scaler para modelos lineares
        self.scaler = None
        
        # Armazenar diagnósticos
        self.diagnostics_history = {}
        
        # Validações
        self._validate_data()
        
        if self.verbose:
            print("=" * 70)
            print("✅ ModelTrainer Inicializado (Versão Corrigida)")
            print("=" * 70)
            print(f"📊 Shape X_train: {self.X_train.shape}")
            print(f"📊 Shape y_train: {self.y_train.shape}")
            print(f"🔢 Número de features: {self.X_train.shape[1]}")
            print(f"🔄 Cross-validation folds: {self.n_splits}")
            print(f"🎲 Random state: {self.random_state}")
            print("=" * 70 + "\n")
    
    def _validate_data(self):
        """✅ CORRIGIDO: Validação mais robusta dos dados."""
        if not isinstance(self.X_train, pd.DataFrame):
            raise TypeError("X_train deve ser um pandas DataFrame")
        
        if not isinstance(self.y_train, (pd.Series, np.ndarray)):
            raise TypeError("y_train deve ser um pandas Series ou numpy array")
        
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X_train e y_train devem ter o mesmo número de amostras")
        
        # ✅ Verificação rigorosa de NaNs
        n_nans = self.X_train.isnull().sum().sum()
        if n_nans > 0:
            nan_cols = self.X_train.columns[self.X_train.isnull().any()].tolist()
            raise ValueError(f"X_train contém {n_nans} valores nulos nas colunas: {nan_cols}")
        
        # ✅ NOVO: Verificar infinitos
        numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            n_infs = np.isinf(self.X_train[numeric_cols].values).sum()
            if n_infs > 0:
                raise ValueError(f"X_train contém {n_infs} valores infinitos")
        
        # ✅ Verificar y_train
        y_series = pd.Series(self.y_train) if not isinstance(self.y_train, pd.Series) else self.y_train
        if y_series.isnull().any():
            raise ValueError(f"y_train contém {y_series.isnull().sum()} valores nulos")
        
        if np.isinf(y_series.values).any():
            raise ValueError("y_train contém valores infinitos")
        
        # ✅ NOVO: Alertar sobre variância zero
        zero_var_cols = self.X_train.columns[self.X_train.std() == 0].tolist()
        if zero_var_cols:
            warnings.warn(f"⚠️ Features com variância zero (considere remover): {zero_var_cols}")
    
    def _create_optuna_study(self, 
                            study_name: str,
                            n_startup_trials: int = 10) -> optuna.Study:
        """
        Cria um estudo Optuna configurado.
        
        Args:
            study_name: Nome do estudo
            n_startup_trials: Número de trials aleatórios iniciais
            
        Returns:
            Estudo Optuna configurado
        """
        sampler = TPESampler(
            seed=self.random_state,
            n_startup_trials=n_startup_trials,
            multivariate=True
        )
        
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3
        )
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
        
        return study
    
    def _calculate_cv_score(self, 
                           model: Any,
                           X: pd.DataFrame = None,
                           y: pd.Series = None,
                           scoring: str = "neg_mean_absolute_error") -> Tuple[float, float]:
        """
        Calcula score de cross-validation.
        
        Args:
            model: Modelo a ser avaliado
            X: Features (usa self.X_train se None)
            y: Target (usa self.y_train se None)
            scoring: Métrica de avaliação
            
        Returns:
            Tuple (mean_score, std_score)
        """
        X = X if X is not None else self.X_train
        y = y if y is not None else self.y_train
        
        scores = cross_val_score(
            model,
            X,
            y,
            cv=self.tscv,
            scoring=scoring,
            n_jobs=-1
        )
        
        return float(np.mean(scores)), float(np.std(scores))
    
    def _save_training_record(self, 
                             model_name: str,
                             best_params: Dict,
                             best_score: float,
                             training_time: float,
                             n_trials: int,
                             study: optuna.Study,
                             model: Any):
        """Salva registro do treinamento no histórico."""
        record = {
            'model_name': model_name,
            'timestamp': pd.Timestamp.now(),
            'best_params': best_params,
            'best_cv_mae': -best_score,
            'training_time_minutes': training_time / 60,
            'n_trials': n_trials,
            'n_completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        }
        
        self.training_history.append(record)
        self.trained_models[model_name] = {
            'model': model,
            'params': best_params,
            'study': study,
            'cv_mae': -best_score
        }
    
    # ========================================================================
    # 🚀 XGBOOST TRAINING
    # ========================================================================
    
    def train_xgboost(self,
                     n_trials: int = 100,
                     timeout: Optional[int] = 3600,
                     plot_results: bool = True) -> Tuple[Dict, xgb.XGBRegressor, optuna.Study]:
        """
        Treina XGBoost com otimização Optuna.
        
        Args:
            n_trials: Número de trials para otimização
            timeout: Timeout em segundos (None = sem limite)
            plot_results: Se True, plota gráficos de otimização
            
        Returns:
            Tuple contendo:
            - best_params: Dicionário com melhores hiperparâmetros
            - model: Modelo XGBoost treinado com melhores parâmetros
            - study: Objeto study do Optuna (para análises adicionais)
        """
        
        if self.verbose:
            print("=" * 70)
            print("🚀 TREINAMENTO XGBOOST COM OPTUNA")
            print("=" * 70)
            print(f"🎯 Número de trials: {n_trials}")
            print(f"⏱️ Timeout: {timeout if timeout else 'Sem limite'} segundos")
            print("=" * 70 + "\n")
        
        # Definir função objetivo
        def objective(trial):
            """Função objetivo para otimização do XGBoost."""
            
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "mae",
                "random_state": self.random_state,
                "tree_method": "hist",
                "n_jobs": -1,
                
                # Parâmetros estruturais
                "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                
                # Sampling
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
                
                # Regularização
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                
                # XGBoost específico
                "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 1.2)
            }
            
            # Criar modelo e avaliar
            model = xgb.XGBRegressor(**params)
            mean_score, std_score = self._calculate_cv_score(model)
            
            # ✅ CORRIGIDO: Penalizar alta variância corretamente
            # mean_score é negativo (ex: -3.0 para MAE=3.0)
            # penalty positivo (ex: 0.5) torna score mais negativo (pior)
            penalty = 0.1 * std_score
            adjusted_score = mean_score + penalty  # -3.0 + 0.5 = -2.5 (melhor que -3.5)
            
            # CORREÇÃO: Invertemos a lógica
            # Quanto MAIOR a variância, PIOR o score (mais negativo)
            adjusted_score = mean_score - penalty  # -3.0 - 0.5 = -3.5 (pior)
            
            # Logging
            trial.set_user_attr("mean_score", mean_score)
            trial.set_user_attr("std_score", std_score)
            trial.set_user_attr("mae", -mean_score)
            
            return adjusted_score
        
        # Criar estudo e otimizar
        study = self._create_optuna_study(
            study_name="xgboost_optimization",
            n_startup_trials=15
        )
        
        start_time = time.time()
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=1,
            show_progress_bar=self.verbose
        )
        
        training_time = time.time() - start_time
        
        # Extrair melhores parâmetros
        best_params = study.best_trial.params.copy()
        best_params.update({
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "random_state": self.random_state,
            "tree_method": "hist",
            "n_jobs": -1
        })
        
        # Treinar modelo final com melhores parâmetros
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(self.X_train, self.y_train)
        
        # Salvar no histórico
        self._save_training_record(
            model_name="XGBoost",
            best_params=best_params,
            best_score=study.best_value,
            training_time=training_time,
            n_trials=n_trials,
            study=study,
            model=final_model
        )
        
        # Mostrar resultados
        if self.verbose:
            self._print_training_results(
                model_name="XGBoost",
                study=study,
                best_params=best_params,
                training_time=training_time
            )
        
        # Plotar resultados
        if plot_results:
            self.plot_optimization_results(study, model_name="XGBoost")
        
        return best_params, final_model, study
    
    # ========================================================================
    # 💡 LIGHTGBM TRAINING
    # ========================================================================
    
    def train_lightgbm(self,
                      n_trials: int = 100,
                      timeout: Optional[int] = 3600,
                      plot_results: bool = True) -> Tuple[Dict, lgb.LGBMRegressor, optuna.Study]:
        """
        Treina LightGBM com otimização Optuna.
        
        Args:
            n_trials: Número de trials para otimização
            timeout: Timeout em segundos (None = sem limite)
            plot_results: Se True, plota gráficos de otimização
            
        Returns:
            Tuple contendo:
            - best_params: Dicionário com melhores hiperparâmetros
            - model: Modelo LightGBM treinado
            - study: Objeto study do Optuna
        """
        
        if self.verbose:
            print("=" * 70)
            print("💡 TREINAMENTO LIGHTGBM COM OPTUNA")
            print("=" * 70)
            print(f"🎯 Número de trials: {n_trials}")
            print(f"⏱️ Timeout: {timeout if timeout else 'Sem limite'} segundos")
            print("=" * 70 + "\n")
        
        # Definir função objetivo
        def objective(trial):
            """Função objetivo para otimização do LightGBM."""
            
            params = {
                "objective": "regression",
                "metric": "mae",
                "verbosity": -1,
                "random_state": self.random_state,
                "n_jobs": -1,
                "force_col_wise": True,
                "boosting_type": "gbdt",
                
                # Parâmetros estruturais
                "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                
                # Regularização
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5)
            }
            
            # Criar modelo e avaliar
            model = lgb.LGBMRegressor(**params)
            mean_score, std_score = self._calculate_cv_score(model)
            
            # ✅ CORRIGIDO: Penalizar alta variância
            penalty = 0.05 * std_score
            adjusted_score = mean_score - penalty
            
            # Logging
            trial.set_user_attr("mean_score", mean_score)
            trial.set_user_attr("std_score", std_score)
            trial.set_user_attr("mae", -mean_score)
            
            return adjusted_score
        
        # Criar estudo e otimizar
        study = self._create_optuna_study(
            study_name="lightgbm_optimization",
            n_startup_trials=10
        )
        
        start_time = time.time()
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=1,
            show_progress_bar=self.verbose
        )
        
        training_time = time.time() - start_time
        
        # Extrair melhores parâmetros
        best_params = study.best_trial.params.copy()
        best_params.update({
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "random_state": self.random_state,
            "n_jobs": -1,
            "force_col_wise": True,
            "boosting_type": "gbdt"
        })
        
        # Treinar modelo final
        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(self.X_train, self.y_train)
        
        # Salvar no histórico
        self._save_training_record(
            model_name="LightGBM",
            best_params=best_params,
            best_score=study.best_value,
            training_time=training_time,
            n_trials=n_trials,
            study=study,
            model=final_model
        )
        
        # Mostrar resultados
        if self.verbose:
            self._print_training_results(
                model_name="LightGBM",
                study=study,
                best_params=best_params,
                training_time=training_time
            )
        
        # Plotar resultados
        if plot_results:
            self.plot_optimization_results(study, model_name="LightGBM")
        
        return best_params, final_model, study
    
    # ========================================================================
    # 🌲 RANDOM FOREST TRAINING
    # ========================================================================
    
    def train_randomforest(self,
                          n_trials: int = 100,
                          timeout: Optional[int] = 3600,
                          plot_results: bool = True) -> Tuple[Dict, RandomForestRegressor, optuna.Study]:
        """
        Treina Random Forest com otimização Optuna.
        
        Args:
            n_trials: Número de trials para otimização
            timeout: Timeout em segundos (None = sem limite)
            plot_results: Se True, plota gráficos de otimização
            
        Returns:
            Tuple contendo:
            - best_params: Dicionário com melhores hiperparâmetros
            - model: Modelo Random Forest treinado
            - study: Objeto study do Optuna
        """
        
        if self.verbose:
            print("=" * 70)
            print("🌲 TREINAMENTO RANDOM FOREST COM OPTUNA")
            print("=" * 70)
            print(f"🎯 Número de trials: {n_trials}")
            print(f"⏱️ Timeout: {timeout if timeout else 'Sem limite'} segundos")
            print("=" * 70 + "\n")
        
        # Definir função objetivo
        def objective(trial):
            """Função objetivo para otimização do Random Forest."""
            
            params = {
                "random_state": self.random_state,
                "n_jobs": -1,
                "bootstrap": True,
                "oob_score": True,
                
                # Número de árvores
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                
                # Profundidade e estrutura
                "max_depth": trial.suggest_categorical("max_depth", [6, 10, 15, 20, None]),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
                
                # Controle de amostragem
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                
                # Bootstrap sampling
                "max_samples": trial.suggest_categorical("max_samples", [0.6, 0.7, 0.8, 0.9, None]),
                
                # Critério de split
                "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error", "friedman_mse"]),
                
                # Complexidade
                "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.01),
                "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [None, 50, 100, 200, 500]),
                
                # Controle adicional
                "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.01)
            }
            
            # Criar modelo e avaliar
            model = RandomForestRegressor(**params)
            mean_score, std_score = self._calculate_cv_score(model)
            
            # ✅ CORRIGIDO: Penalizar alta variância
            penalty = 0.05 * std_score
            adjusted_score = mean_score - penalty
            
            # Logging
            trial.set_user_attr("mean_score", mean_score)
            trial.set_user_attr("std_score", std_score)
            trial.set_user_attr("mae", -mean_score)
            
            return adjusted_score
        
        # Criar estudo e otimizar
        study = self._create_optuna_study(
            study_name="randomforest_optimization",
            n_startup_trials=20
        )
        
        start_time = time.time()
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=1,
            show_progress_bar=self.verbose
        )
        
        training_time = time.time() - start_time
        
        # Extrair melhores parâmetros
        best_params = study.best_trial.params.copy()
        best_params.update({
            "random_state": self.random_state,
            "n_jobs": -1,
            "bootstrap": True,
            "oob_score": True
        })
        
        # Treinar modelo final
        final_model = RandomForestRegressor(**best_params)
        final_model.fit(self.X_train, self.y_train)
        
        # Salvar no histórico
        self._save_training_record(
            model_name="RandomForest",
            best_params=best_params,
            best_score=study.best_value,
            training_time=training_time,
            n_trials=n_trials,
            study=study,
            model=final_model
        )
        
        # Mostrar resultados
        if self.verbose:
            self._print_training_results(
                model_name="RandomForest",
                study=study,
                best_params=best_params,
                training_time=training_time
            )
        
        # Plotar resultados
        if plot_results:
            self.plot_optimization_results(study, model_name="RandomForest")
        
        return best_params, final_model, study
    
    # ========================================================================
    # 📏 ELASTICNET TRAINING
    # ========================================================================
    
    def train_elasticnet(self,
                        n_trials: int = 80,
                        timeout: Optional[int] = 600,
                        plot_results: bool = True) -> Tuple[Dict, ElasticNet, optuna.Study, StandardScaler]:
        """
        Treina ElasticNet com otimização Optuna.
        
        IMPORTANTE: Retorna também o scaler usado para normalização!
        
        Args:
            n_trials: Número de trials para otimização
            timeout: Timeout em segundos (None = sem limite)
            plot_results: Se True, plota gráficos de otimização
            
        Returns:
            Tuple contendo:
            - best_params: Dicionário com melhores hiperparâmetros
            - model: Modelo ElasticNet treinado
            - study: Objeto study do Optuna
            - scaler: StandardScaler usado (necessário para predições)
        """
        
        if self.verbose:
            print("=" * 70)
            print("📏 TREINAMENTO ELASTICNET COM OPTUNA")
            print("=" * 70)
            print(f"🎯 Número de trials: {n_trials}")
            print(f"⏱️ Timeout: {timeout if timeout else 'Sem limite'} segundos")
            print("⚠️ ATENÇÃO: ElasticNet requer normalização dos dados")
            print("=" * 70 + "\n")
        
        # Normalizar dados
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        
        # Definir função objetivo
        def objective(trial):
            """Função objetivo para otimização do ElasticNet."""
            
            params = {
                "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "max_iter": 10000,
                "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
                "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
                "random_state": self.random_state
            }
            
            # Criar modelo e avaliar
            model = ElasticNet(**params)
            mean_score, std_score = self._calculate_cv_score(model, X=X_train_scaled)
            
            # ✅ CORRIGIDO: Penalizar alta variância
            penalty = 0.05 * std_score
            adjusted_score = mean_score - penalty
            
            # Logging
            trial.set_user_attr("mean_score", mean_score)
            trial.set_user_attr("std_score", std_score)
            trial.set_user_attr("mae", -mean_score)
            
            # Identificar tipo de regularização
            l1_ratio = params['l1_ratio']
            if l1_ratio < 0.1:
                reg_type = "Ridge (L2)"
            elif l1_ratio > 0.9:
                reg_type = "Lasso (L1)"
            else:
                reg_type = f"ElasticNet ({l1_ratio:.0%}L1, {1-l1_ratio:.0%}L2)"
            trial.set_user_attr("regularization_type", reg_type)
            
            return adjusted_score
        
        # Criar estudo e otimizar
        study = self._create_optuna_study(
            study_name="elasticnet_optimization",
            n_startup_trials=10
        )
        
        start_time = time.time()
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=-1,  # ElasticNet é rápido, pode paralelizar
            show_progress_bar=self.verbose
        )
        
        training_time = time.time() - start_time
        
        # Extrair melhores parâmetros
        best_params = study.best_trial.params.copy()
        best_params.update({
            "max_iter": 10000,
            "random_state": self.random_state
        })
        
        # Treinar modelo final
        final_model = ElasticNet(**best_params)
        final_model.fit(X_train_scaled, self.y_train)
        
        # Salvar no histórico
        self._save_training_record(
            model_name="ElasticNet",
            best_params=best_params,
            best_score=study.best_value,
            training_time=training_time,
            n_trials=n_trials,
            study=study,
            model=final_model
        )
        
        # Informações adicionais
        n_features_nonzero = np.sum(final_model.coef_ != 0)
        reg_type = study.best_trial.user_attrs.get('regularization_type', 'N/A')
        
        # Mostrar resultados
        if self.verbose:
            self._print_training_results(
                model_name="ElasticNet",
                study=study,
                best_params=best_params,
                training_time=training_time
            )
            print(f"📊 Features selecionadas: {n_features_nonzero}/{len(self.X_train.columns)}")
            print(f"📊 Tipo de regularização: {reg_type}")
            print(f"🔁 Iterações para convergência: {final_model.n_iter_}")
            print()
        
        # Plotar resultados
        if plot_results:
            self.plot_optimization_results(study, model_name="ElasticNet")
            self._plot_elasticnet_coefficients(final_model)
        
        return best_params, final_model, study, self.scaler
    
    def _plot_elasticnet_coefficients(self, model: ElasticNet, top_n: int = 20):
        """Plota coeficientes do ElasticNet."""
        
        coef_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Filtrar apenas coeficientes não-zero
        coef_nonzero = coef_df[coef_df['coefficient'] != 0].head(top_n)
        
        if len(coef_nonzero) == 0:
            print("⚠️ Todos os coeficientes foram zerados pela regularização")
            return
        
        plt.figure(figsize=(12, max(6, len(coef_nonzero) * 0.3)))
        
        colors = ['green' if c > 0 else 'red' for c in coef_nonzero['coefficient']]
        
        plt.barh(coef_nonzero['feature'], coef_nonzero['coefficient'], color=colors, alpha=0.7)
        plt.xlabel('Coeficiente')
        plt.ylabel('Feature')
        plt.title(f'ElasticNet - Top {len(coef_nonzero)} Coeficientes Mais Importantes')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Total de features com coef ≠ 0: {np.sum(model.coef_ != 0)}/{len(model.coef_)}")
    
    # ========================================================================
    # 🎯 TREINAMENTO DE TODOS OS MODELOS
    # ========================================================================
    
    def train_all_models(self,
                        n_trials_dict: Optional[Dict[str, int]] = None,
                        timeout: Optional[int] = None,
                        plot_individual: bool = False) -> Dict[str, Tuple]:
        """
        Treina todos os modelos disponíveis sequencialmente.
        
        Args:
            n_trials_dict: Dicionário com número de trials por modelo
                          Ex: {'xgboost': 100, 'lightgbm': 100, 'randomforest': 80, 'elasticnet': 60}
            timeout: Timeout global em segundos
            plot_individual: Se True, plota gráfico de cada modelo
            
        Returns:
            Dicionário com resultados de todos os modelos
        """
        
        # Configurações padrão
        if n_trials_dict is None:
            n_trials_dict = {
                'xgboost': 100,
                'lightgbm': 100,
                'randomforest': 80,
                'elasticnet': 60
            }
        
        results = {}
        
        print("\n" + "=" * 70)
        print("🚀 INICIANDO TREINAMENTO DE TODOS OS MODELOS")
        print("=" * 70)
        print(f"Modelos a treinar: {list(n_trials_dict.keys())}")
        print(f"Total de trials: {sum(n_trials_dict.values())}")
        print("=" * 70 + "\n")
        
        start_total = time.time()
        
        # XGBoost
        if 'xgboost' in n_trials_dict:
            try:
                params, model, study = self.train_xgboost(
                    n_trials=n_trials_dict['xgboost'],
                    timeout=timeout,
                    plot_results=plot_individual
                )
                results['XGBoost'] = (params, model, study)
            except Exception as e:
                print(f"❌ Erro ao treinar XGBoost: {e}")
        
        # LightGBM
        if 'lightgbm' in n_trials_dict:
            try:
                params, model, study = self.train_lightgbm(
                    n_trials=n_trials_dict['lightgbm'],
                    timeout=timeout,
                    plot_results=plot_individual
                )
                results['LightGBM'] = (params, model, study)
            except Exception as e:
                print(f"❌ Erro ao treinar LightGBM: {e}")
        
        # Random Forest
        if 'randomforest' in n_trials_dict:
            try:
                params, model, study = self.train_randomforest(
                    n_trials=n_trials_dict['randomforest'],
                    timeout=timeout,
                    plot_results=plot_individual
                )
                results['RandomForest'] = (params, model, study)
            except Exception as e:
                print(f"❌ Erro ao treinar Random Forest: {e}")
        
        # ElasticNet
        if 'elasticnet' in n_trials_dict:
            try:
                params, model, study, scaler = self.train_elasticnet(
                    n_trials=n_trials_dict['elasticnet'],
                    timeout=timeout,
                    plot_results=plot_individual
                )
                results['ElasticNet'] = (params, model, study, scaler)
            except Exception as e:
                print(f"❌ Erro ao treinar ElasticNet: {e}")
        
        total_time = time.time() - start_total
        
        print("\n" + "=" * 70)
        print("✅ TREINAMENTO DE TODOS OS MODELOS CONCLUÍDO")
        print("=" * 70)
        print(f"⏱️ Tempo total: {total_time/60:.2f} minutos")
        print(f"📊 Modelos treinados: {len(results)}")
        print("=" * 70 + "\n")
        
        # Comparação automática
        self.compare_all_models()
        
        return results
    
    # ========================================================================
    # 🔍 MÉTODO DE DIAGNÓSTICO
    # ========================================================================
    
    def diagnose_model(self, 
                      model: Any,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      model_name: str = "Modelo",
                      scaler: Optional[StandardScaler] = None) -> Dict:
        """
        ✅ NOVO: Diagnóstico completo de problemas no modelo.
        
        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Target de teste
            model_name: Nome do modelo
            scaler: Scaler (se modelo usar normalização)
            
        Returns:
            Dicionário com métricas e diagnósticos
        """
        
        print("\n" + "="*80)
        print(f"🔍 DIAGNÓSTICO COMPLETO - {model_name}")
        print("="*80)
        
        # Preparar dados
        X_train_eval = self.X_train if scaler is None else pd.DataFrame(
            scaler.transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        X_test_eval = X_test if scaler is None else pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # 1. Previsões
        y_pred_train = model.predict(X_train_eval)
        y_pred_test = model.predict(X_test_eval)
        
        mae_train = mean_absolute_error(self.y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_train = r2_score(self.y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        print("\n📊 MÉTRICAS")
        print(f"MAE Train:  {mae_train:.4f}")
        print(f"MAE Test:   {mae_test:.4f}")
        print(f"Gap MAE:    {((mae_test - mae_train)/mae_train * 100):+.2f}%")
        print(f"\nRMSE Train: {rmse_train:.4f}")
        print(f"RMSE Test:  {rmse_test:.4f}")
        print(f"Gap RMSE:   {((rmse_test - rmse_train)/rmse_train * 100):+.2f}%")
        print(f"\nR² Train:   {r2_train:.4f}")
        print(f"R² Test:    {r2_test:.4f}")
        
        # ✅ Diagnóstico de overfitting
        mae_gap = ((mae_test - mae_train)/mae_train * 100)
        if mae_gap < 5:
            print("\n✅ Modelo bem generalizado (gap < 5%)")
        elif mae_gap < 15:
            print("\n⚠️ Leve overfitting (5% < gap < 15%)")
        elif mae_gap < 50:
            print("\n❌ Overfitting moderado (15% < gap < 50%)")
        else:
            print("\n🚨 OVERFITTING SEVERO (gap > 50%) - Possível data leakage!")
        
        # 2. Análise de erros
        errors_train = np.abs(self.y_train - y_pred_train)
        errors_test = np.abs(y_test - y_pred_test)
        
        print("\n📊 ANÁLISE DE ERROS")
        print(f"Erro médio Train:   {errors_train.mean():.4f}")
        print(f"Erro médio Test:    {errors_test.mean():.4f}")
        print(f"Erro máximo Train:  {errors_train.max():.4f}")
        print(f"Erro máximo Test:   {errors_test.max():.4f}")
        print(f"% erros > 5 (Train): {(errors_train > 5).sum() / len(errors_train) * 100:.1f}%")
        print(f"% erros > 5 (Test):  {(errors_test > 5).sum() / len(errors_test) * 100:.1f}%")
        
        # 3. Distribuição do target
        print("\n📊 DISTRIBUIÇÃO DO TARGET")
        print(f"Train - Mean: {self.y_train.mean():.4f}, Std: {self.y_train.std():.4f}, "
              f"Range: [{self.y_train.min():.2f}, {self.y_train.max():.2f}]")
        print(f"Test  - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}, "
              f"Range: [{y_test.min():.2f}, {y_test.max():.2f}]")
        
        # ✅ Teste estatístico
        ks_stat, ks_pvalue = ks_2samp(self.y_train, y_test)
        print(f"\nKolmogorov-Smirnov test p-value: {ks_pvalue:.4f}")
        if ks_pvalue < 0.05:
            print("⚠️ ALERTA: Distribuições de treino e teste são SIGNIFICATIVAMENTE diferentes!")
        else:
            print("✅ Distribuições de treino e teste são similares")
        
        # 4. Features suspeitas de leakage
        print("\n🚨 VERIFICAÇÃO DE DATA LEAKAGE")
        suspicious_features = []
        
        for col in self.X_train.columns:
            try:
                corr_train = np.corrcoef(self.X_train[col], self.y_train)[0, 1]
                
                if abs(corr_train) > 0.95:
                    suspicious_features.append((col, corr_train))
                    print(f"  ⚠️ {col}: correlação = {corr_train:.4f} (MUITO ALTA - possível leakage!)")
            except:
                continue
        
        if not suspicious_features:
            print("  ✅ Nenhuma feature com correlação suspeita encontrada")
        
        # 5. Feature importance (se disponível)
        if hasattr(model, 'feature_importances_'):
            print("\n📊 TOP 10 FEATURES MAIS IMPORTANTES")
            importances = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            for idx, row in importances.iterrows():
                print(f"  {row['feature']:<30} {row['importance']:.4f}")
        
        # 6. Gráficos
        self._plot_diagnostic_charts(
            self.y_train, y_pred_train, 
            y_test, y_pred_test,
            errors_train, errors_test,
            model_name
        )
        
        print("\n" + "="*80)
        
        # Salvar diagnóstico
        diagnostic_result = {
            'mae_train': mae_train,
            'mae_test': mae_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mae_gap_percent': mae_gap,
            'rmse_gap_percent': ((rmse_test - rmse_train)/rmse_train * 100),
            'suspicious_features': suspicious_features,
            'ks_test_pvalue': ks_pvalue
        }
        
        self.diagnostics_history[model_name] = diagnostic_result
        
        return diagnostic_result
    
    def _plot_diagnostic_charts(self, y_train, y_pred_train, y_test, y_pred_test, 
                               errors_train, errors_test, model_name):
        """Plota gráficos de diagnóstico."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Diagnóstico - {model_name}', fontsize=14, fontweight='bold')
        
        # Real vs Predito (Train)
        axes[0, 0].scatter(y_train, y_pred_train, alpha=0.5, s=10)
        min_val = min(y_train.min(), y_pred_train.min())
        max_val = max(y_train.max(), y_pred_train.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Real')
        axes[0, 0].set_ylabel('Predito')
        axes[0, 0].set_title('Train - Real vs Predito')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Real vs Predito (Test)
        axes[0, 1].scatter(y_test, y_pred_test, alpha=0.5, s=10, color='orange')
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 1].set_xlabel('Real')
        axes[0, 1].set_ylabel('Predito')
        axes[0, 1].set_title('Test - Real vs Predito')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribuição de erros
        axes[1, 0].hist(errors_train, bins=50, alpha=0.6, label='Train', color='blue')
        axes[1, 0].hist(errors_test, bins=50, alpha=0.6, label='Test', color='red')
        axes[1, 0].set_xlabel('Erro Absoluto')
        axes[1, 0].set_ylabel('Frequência')
        axes[1, 0].set_title('Distribuição dos Erros')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Resíduos
        axes[1, 1].plot(errors_train, alpha=0.6, label='Train', linewidth=0.5)
        axes[1, 1].plot(errors_test, alpha=0.6, label='Test', linewidth=0.5)
        axes[1, 1].set_xlabel('Índice')
        axes[1, 1].set_ylabel('Erro Absoluto')
        axes[1, 1].set_title('Erros ao Longo das Amostras')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # ========================================================================
    # 📊 VISUALIZAÇÃO E ANÁLISE
    # ========================================================================
    
    def _print_training_results(self,
                                model_name: str,
                                study: optuna.Study,
                                best_params: Dict,
                                training_time: float):
        """Imprime resultados do treinamento."""
        
        print("\n" + "=" * 70)
        print(f"✅ RESULTADOS - {model_name.upper()}")
        print("=" * 70)
        
        print(f"\n⏱️ Tempo de treinamento: {training_time/60:.2f} minutos")
        print(f"🎯 Trials completados: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"✂️ Trials podados: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        
        print(f"\n📊 Melhor score (neg_mae): {study.best_value:.4f}")
        print(f"📊 MAE equivalente: {-study.best_value:.4f}")
        
        print("\n🎯 MELHORES HIPERPARÂMETROS:")
        print("-" * 70)
        
        # Filtrar parâmetros de configuração
        config_params = ['objective', 'eval_metric', 'random_state', 'tree_method', 
                        'n_jobs', 'verbosity', 'force_col_wise', 'boosting_type',
                        'metric', 'max_iter']
        
        for param, value in sorted(best_params.items()):
            if param not in config_params:
                if isinstance(value, float):
                    if value < 0.01:
                        print(f"  • {param:<25} {value:.2e}")
                    else:
                        print(f"  • {param:<25} {value:.4f}")
                else:
                    print(f"  • {param:<25} {value}")
        
        # Importância dos parâmetros
        print("\n📈 IMPORTÂNCIA DOS HIPERPARÂMETROS (Top 5):")
        print("-" * 70)
        try:
            param_importance = optuna.importance.get_param_importances(study)
            for i, (param, importance) in enumerate(sorted(param_importance.items(), 
                                                          key=lambda x: x[1], 
                                                          reverse=True)[:5], 1):
                print(f"  {i}. {param:<25} {importance:.4f}")
        except:
            print("  (Necessário ≥2 trials completos)")
        
        print("=" * 70 + "\n")
    
    def plot_optimization_results(self,
                                  study: optuna.Study,
                                  model_name: str = "Model",
                                  save_plots: bool = True,
                                  output_dir: str = "optimization_plots"):
        """Plota resultados da otimização."""
        
        if save_plots:
            Path(output_dir).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Otimização {model_name} - Análise Completa', 
                    fontsize=16, fontweight='bold')
        
        # 1. Histórico de otimização
        ax1 = axes[0, 0]
        trials = study.trials
        trial_numbers = [t.number for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        trial_values = [-t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if trial_values:
            ax1.plot(trial_numbers, trial_values, 'o-', alpha=0.6, label='Trial MAE', markersize=4)
            ax1.plot(trial_numbers, 
                    np.minimum.accumulate(trial_values), 
                    'r-', linewidth=2, label='Best MAE')
            ax1.set_xlabel('Trial Number', fontsize=11)
            ax1.set_ylabel('MAE', fontsize=11)
            ax1.set_title('Histórico de Otimização', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Importância dos parâmetros
        ax2 = axes[0, 1]
        try:
            param_importance = optuna.importance.get_param_importances(study)
            params = list(param_importance.keys())[:8]
            importances = [param_importance[p] for p in params]
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
            bars = ax2.barh(params, importances, color=colors)
            ax2.set_xlabel('Importância', fontsize=11)
            ax2.set_title('Importância dos Hiperparâmetros', fontsize=12, fontweight='bold')
            ax2.invert_yaxis()
            
            for bar, imp in zip(bars, importances):
                ax2.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                        f'{imp:.3f}', va='center', ha='left', fontsize=9)
        except:
            ax2.text(0.5, 0.5, 'Dados insuficientes\n(≥2 trials necessários)',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Importância dos Hiperparâmetros', fontsize=12, fontweight='bold')
        
        # 3. Distribuição de scores
        ax3 = axes[1, 0]
        if trial_values:
            n_bins = min(30, max(10, len(trial_values)//3))
            ax3.hist(trial_values, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(min(trial_values), color='red', linestyle='--', 
                       linewidth=2, label=f'Melhor: {min(trial_values):.4f}')
            ax3.axvline(np.median(trial_values), color='orange', linestyle='--',
                       linewidth=2, label=f'Mediana: {np.median(trial_values):.4f}')
            ax3.set_xlabel('MAE', fontsize=11)
            ax3.set_ylabel('Frequência', fontsize=11)
            ax3.set_title('Distribuição dos Scores (MAE)', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Estatísticas resumidas
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calcular valores
        best_mae = min(trial_values) if trial_values else None
        worst_mae = max(trial_values) if trial_values else None
        mean_mae = np.mean(trial_values) if trial_values else None
        median_mae = np.median(trial_values) if trial_values else None
        std_mae = np.std(trial_values) if trial_values else None
        improvement = ((worst_mae - best_mae) / worst_mae * 100) if (trial_values and worst_mae > 0) else None
        top_10_count = len([v for v in trial_values if v <= np.percentile(trial_values, 10)]) if trial_values else 0
        
        # Formatação
        best_mae_str = f"{best_mae:.6f}" if best_mae is not None else "N/A"
        worst_mae_str = f"{worst_mae:.6f}" if worst_mae is not None else "N/A"
        mean_mae_str = f"{mean_mae:.6f}" if mean_mae is not None else "N/A"
        median_mae_str = f"{median_mae:.6f}" if median_mae is not None else "N/A"
        std_mae_str = f"{std_mae:.6f}" if std_mae is not None else "N/A"
        improvement_str = f"{improvement:.2f}%" if improvement is not None else "N/A"
        
        stats_text = f"""
        📊 ESTATÍSTICAS DA OTIMIZAÇÃO
        {'='*40}
        
        Trials Completados:  {len([t for t in trials if t.state == optuna.trial.TrialState.COMPLETE])}
        Trials Podados:      {len([t for t in trials if t.state == optuna.trial.TrialState.PRUNED])}
        
        Melhor MAE:          {best_mae_str}
        Pior MAE:            {worst_mae_str}
        MAE Médio:           {mean_mae_str}
        MAE Mediano:         {median_mae_str}
        Desvio Padrão:       {std_mae_str}
        
        Melhoria:            {improvement_str}
        
        Trials no Top 10%:   {top_10_count}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = Path(output_dir) / f"{model_name.lower()}_optimization.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot salvo em: {plot_path}")
        
        plt.show()
    
    def compare_all_models(self):
        """Compara todos os modelos treinados."""
        
        if not self.training_history:
            print("⚠️ Nenhum modelo treinado ainda")
            return
        
        df = pd.DataFrame(self.training_history)
        df = df.sort_values('best_cv_mae')
        
        print("\n" + "=" * 90)
        print("🏆 COMPARAÇÃO DE TODOS OS MODELOS")
        print("=" * 90)
        
        print(f"\n{'Modelo':<15} {'MAE (CV)':<12} {'Tempo (min)':<12} {'Trials':<10} {'Data/Hora':<20}")
        print("-" * 90)
        
        for idx, row in df.iterrows():
            print(f"{row['model_name']:<15} {row['best_cv_mae']:<12.6f} "
                  f"{row['training_time_minutes']:<12.2f} "
                  f"{row['n_completed_trials']}/{row['n_trials']:<7} "
                  f"{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'):<20}")
        
        print("-" * 90)
        best_model = df.iloc[0]
        print(f"\n🥇 MELHOR MODELO: {best_model['model_name']}")
        print(f"📊 MAE (CV): {best_model['best_cv_mae']:.6f}")
        print(f"⏱️ Tempo de treinamento: {best_model['training_time_minutes']:.2f} minutos")
        
        # Calcular diferença percentual entre melhor e segundo melhor
        if len(df) > 1:
            second_best = df.iloc[1]
            improvement = ((second_best['best_cv_mae'] - best_model['best_cv_mae']) / 
                          second_best['best_cv_mae'] * 100)
            print(f"📈 Melhoria sobre 2º lugar ({second_best['model_name']}): {improvement:.2f}%")
        
        print("=" * 90 + "\n")
        
        # Plotar comparação visual
        self._plot_model_comparison(df)
    
    def _plot_model_comparison(self, df: pd.DataFrame):
        """Plota comparação visual entre modelos."""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. MAE por modelo
        ax1 = axes[0]
        colors = ['green' if i == 0 else 'skyblue' for i in range(len(df))]
        bars = ax1.barh(df['model_name'], df['best_cv_mae'], color=colors)
        ax1.set_xlabel('MAE (Cross-Validation)', fontsize=11)
        ax1.set_title('Comparação de MAE entre Modelos', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        
        # Adicionar valores nas barras
        for bar, mae in zip(bars, df['best_cv_mae']):
            ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f' {mae:.6f}', va='center', fontsize=9)
        
        # 2. Tempo de treinamento
        ax2 = axes[1]
        ax2.barh(df['model_name'], df['training_time_minutes'], color='coral')
        ax2.set_xlabel('Tempo de Treinamento (minutos)', fontsize=11)
        ax2.set_title('Tempo de Treinamento por Modelo', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        
        for i, (idx, row) in enumerate(df.iterrows()):
            ax2.text(row['training_time_minutes'], i,
                    f" {row['training_time_minutes']:.1f} min", 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def get_training_history(self) -> pd.DataFrame:
        """Retorna histórico de treinamentos como DataFrame."""
        if not self.training_history:
            print("⚠️ Nenhum treinamento realizado ainda")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.training_history)
        return df.sort_values('best_cv_mae')
    
    def get_best_model(self) -> Tuple[str, Any, Dict, float]:
        """Retorna informações do melhor modelo treinado."""
        if not self.trained_models:
            print("⚠️ Nenhum modelo treinado ainda")
            return None
        
        best_name = min(self.trained_models.items(), 
                       key=lambda x: x[1]['cv_mae'])[0]
        best_data = self.trained_models[best_name]
        
        return (best_name, 
                best_data['model'], 
                best_data['params'], 
                best_data['cv_mae'])
    
    def print_training_summary(self):
        """Imprime resumo detalhado de todos os treinamentos."""
        
        if not self.training_history:
            print("⚠️ Nenhum treinamento realizado ainda")
            return
        
        df = self.get_training_history()
        
        print("\n" + "=" * 70)
        print("📋 RESUMO DETALHADO DE TREINAMENTOS")
        print("=" * 70)
        
        for idx, row in df.iterrows():
            print(f"\n{'='*70}")
            print(f"🔹 {row['model_name']}")
            print(f"{'='*70}")
            print(f"  📊 MAE (CV):           {row['best_cv_mae']:.6f}")
            print(f"  ⏱️ Tempo:              {row['training_time_minutes']:.2f} minutos")
            print(f"  🎯 Trials:             {row['n_completed_trials']}/{row['n_trials']}")
            print(f"  ✂️ Trials podados:     {row['n_pruned_trials']}")
            print(f"  📅 Data:               {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n" + "=" * 70)
        print(f"🏆 MELHOR MODELO: {df.iloc[0]['model_name']}")
        print(f"📊 MAE: {df.iloc[0]['best_cv_mae']:.6f}")
        print("=" * 70 + "\n")