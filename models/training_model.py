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
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

# Optuna
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

"""
# Forças todos os prints
import builtins
import functools

print = functools.partial(builtins.print, flush=True)"""

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Classe para treinamento automatizado de modelos com otimização Optuna.
    
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
            print("✅ ModelTrainer Inicializado")
            print("=" * 70)
            print(f"📊 Shape X_train: {self.X_train.shape}")
            print(f"📊 Shape y_train: {self.y_train.shape}")
            print(f"🔢 Número de features: {self.X_train.shape[1]}")
            print(f"🔄 Cross-validation folds: {self.n_splits}")
            print(f"🎲 Random state: {self.random_state}")
            print("=" * 70 + "\n")
    
    def _validate_data(self):
        """Validação robusta dos dados."""
        if not isinstance(self.X_train, pd.DataFrame):
            raise TypeError("X_train deve ser um pandas DataFrame")
        
        if not isinstance(self.y_train, (pd.Series, np.ndarray)):
            raise TypeError("y_train deve ser um pandas Series ou numpy array")
        
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X_train e y_train devem ter o mesmo número de amostras")
        
        # Verificação de NaNs
        n_nans = self.X_train.isnull().sum().sum()
        if n_nans > 0:
            nan_cols = self.X_train.columns[self.X_train.isnull().any()].tolist()
            raise ValueError(f"X_train contém {n_nans} valores nulos nas colunas: {nan_cols}")
        
        # Verificar valores infinitos
        numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            n_infs = np.isinf(self.X_train[numeric_cols].values).sum()
            if n_infs > 0:
                raise ValueError(f"X_train contém {n_infs} valores infinitos")
        
        # Verificar y_train
        y_series = pd.Series(self.y_train) if not isinstance(self.y_train, pd.Series) else self.y_train
        if y_series.isnull().any():
            raise ValueError(f"y_train contém {y_series.isnull().sum()} valores nulos")
        
        if np.isinf(y_series.values).any():
            raise ValueError("y_train contém valores infinitos")
        
        # Alertar sobre variância zero
        zero_var_cols = self.X_train.columns[self.X_train.std() == 0].tolist()
        if zero_var_cols:
            warnings.warn(f"⚠️ Features com variância zero (considere remover): {zero_var_cols}")
    
    def _create_optuna_study(self, 
                            study_name: str,
                            n_startup_trials: int = 20) -> optuna.Study:
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
            n_startup_trials=10,
            n_warmup_steps=5
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
            n_jobs=1,
            error_score='raise'
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
    # XGBOOST TRAINING
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
            
            # mean_score é negativo (ex: -3.0 para MAE=3.0)
            # penalty positivo (ex: 0.5) torna score mais negativo (pior)
            penalty = 0.1 * std_score
            
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
    # LIGHTGBM TRAINING
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
            
            # Penalizar alta variância
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
    # RANDOM FOREST TRAINING
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
            
            # Penalizar alta variância
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
    

        #     Normalizar dados
        # self.scaler = StandardScaler()
        # X_train_scaled = pd.DataFrame(
        #     self.scaler.fit_transform(self.X_train),
        #     columns=self.X_train.columns,
        #     index=self.X_train.index
        # )

        #      Criar modelo e avaliar
        # model = ElasticNet(**params)
        # mean_score, std_score = self._calculate_cv_score(model, X=X_train_scaled)

        #    Treinar modelo final
        # final_model = ElasticNet(**best_params)
        # final_model.fit(X_train_scaled, self.y_train)

    # ========================================================================
    # ELASTICNET TRAINING
    # ========================================================================
    
    def train_elasticnet(self,
                        n_trials: int = 80,
                        timeout: Optional[int] = 3600,
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

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', ElasticNet(**params))
            ])
    
            # Agora passamos o X_train ORIGINAL, o pipeline cuida do resto
            mean_score, std_score = self._calculate_cv_score(pipeline)
            
            # Penalizar alta variância
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
            n_startup_trials=20
        )
        
        start_time = time.time()
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=-1,  
            show_progress_bar=self.verbose
        )
        
        training_time = time.time() - start_time
        
        # Extrair melhores parâmetros
        best_params = study.best_trial.params.copy()
        final_model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(**best_params, max_iter=10000, random_state=self.random_state))
        ])

        # Treinamos usando os dados BRUTOS (self.X_train)
        final_model.fit(self.X_train, self.y_train)
        
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
        
        model_inner = final_model.named_steps['model']
        n_features_nonzero = np.sum(model_inner.coef_ != 0)
        # n_features_nonzero = np.sum(final_model.coef_ != 0)
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
            print(f"🔁 Iterações para convergência: {model_inner.n_iter_}")
            # print(f"🔁 Iterações para convergência: {final_model.n_iter_}")
            print()
        
        # Plotar resultados
        if plot_results:
            self.plot_optimization_results(study, model_name="ElasticNet")
            # Passamos o modelo interno para a função de plot de coeficientes
            self._plot_elasticnet_coefficients(model_inner)
        
        return best_params, final_model, study  
        # Retornamos o final_model (que já contém o scaler internamente!)
    
    def _plot_elasticnet_coefficients(self, model: ElasticNet, top_n: int = 20):
        """Plota coeficientes do ElasticNet."""
        
        # 1. Lógica de "Desembrulho": Extrai o modelo se ele estiver dentro de um Pipeline
        if hasattr(model, 'named_steps'):
            model_to_plot = model.named_steps['model']
        else:
            model_to_plot = model

        # 2. Verificação de segurança
        if not hasattr(model_to_plot, 'coef_'):
            print("⚠️ O modelo fornecido não possui coeficientes (não é um modelo linear).")
            return

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
    # TREINAMENTO DE TODOS OS MODELOS
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
        
        """
        if 'elasticnet' in n_trials_dict:
            try:
                params, model, study, scaler = self.train_elasticnet(
                    n_trials=n_trials_dict['elasticnet'],
                    timeout=timeout,
                    plot_results=plot_individual
                )
                results['ElasticNet'] = (params, model, study, scaler)
            except Exception as e:
                print(f"❌ Erro ao treinar ElasticNet: {e}")"""

        # ElasticNet - Agora padronizado com 3 retornos
        if 'elasticnet' in n_trials_dict:
            try:
                # O model aqui já é um Pipeline(Scaler + ElasticNet)
                params, model, study = self.train_elasticnet(
                    n_trials=n_trials_dict['elasticnet'],
                    timeout=timeout,
                    plot_results=plot_individual
                )
                results['ElasticNet'] = (params, model, study)
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
    # MÉTODO DE DIAGNÓSTICO
    # ========================================================================
    """
    def diagnose_model(self, 
                      model: Any,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      model_name: str = "Modelo",
                      scaler: Optional[StandardScaler] = None) -> Dict:
        
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
    
    
    """

    def diagnose_model(self, 
                   model: Any,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   model_name: str = "Modelo") -> Dict:
        """
        Diagnóstico completo de problemas no modelo.
        
        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Target de teste
            model_name: Nome do modelo
            scaler: Scaler (se modelo usar normalização)
            
        Returns:
            Dicionário com métricas e diagnósticos

        Diagnóstico completo adaptado para Pipelines e modelos nativos.
        Não é mais necessário passar o scaler separadamente, pois ele está no Pipeline.
        """
        print("\n" + "="*80)
        print(f"🔍 DIAGNÓSTICO COMPLETO - {model_name}")
        print("="*80)
        
        # 1. Previsões Inteligentes
        # Se for Pipeline (ElasticNet), ele escala internamente. 
        # Se for árvore, ele usa o dado bruto. A lógica é a mesma no .predict()
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(X_test)
        
        # Métricas Base
        mae_train = mean_absolute_error(self.y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_train = r2_score(self.y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        print(f"{'📊 Métrica':<15} {'Train':>15} {'Test':>15} {'Gap':>15}")
        print("-" * 80)
        mae_gap = ((mae_test - mae_train)/mae_train * 100)
        print(f"{'MAE':<15} {mae_train:>15.4f} {mae_test:>15.4f} {mae_gap:>14.2f}%")
        rmse_gap = ((rmse_test - rmse_train)/rmse_train * 100)
        print(f"{'RMSE':<15} {rmse_train:>15.4f} {rmse_test:>15.4f} {rmse_gap:>14.2f}%")
        print(f"{'R²':<15} {r2_train:>15.4f} {r2_test:>15.4f} {'-':>15}")
        print("-" * 80)
        
        # 2. Análise de Erros (Resíduos)
        errors_train = np.abs(self.y_train - y_pred_train)
        errors_test = np.abs(y_test - y_pred_test)
        
        print("\n📊 ANÁLISE DE ERROS")
        print(f"Erro médio Train:   {errors_train.mean():.4f}")
        print(f"Erro médio Test:    {errors_test.mean():.4f}")
        print(f"Erro máximo Train:  {errors_train.max():.4f}")
        print(f"Erro máximo Test:   {errors_test.max():.4f}")
        print(f"% erros > 0.5 (Train): {(errors_train > 0.5).sum() / len(errors_train) * 100:.1f}%")
        print(f"% erros > 0.5 (Test):  {(errors_test > 0.5).sum() / len(errors_test) * 100:.1f}%")
        
        # 3. Teste Kolmogorov-Smirnov (Sanidade dos Dados) e Distribuição do Target
        print("\n📊 DISTRIBUIÇÃO DO TARGET")
        print(f"Train - Mean: {self.y_train.mean():.4f}, Std: {self.y_train.std():.4f}, "
              f"Range: [{self.y_train.min():.2f}, {self.y_train.max():.2f}]")
        print(f"Test  - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}, "
              f"Range: [{y_test.min():.2f}, {y_test.max():.2f}]")
        
        ks_stat, ks_pvalue = ks_2samp(self.y_train, y_test)
        print(f"\n🧪 Teste KS (Treino vs Teste): p-value = {ks_pvalue:.4f}")
        if ks_pvalue < 0.05:
            print("⚠️ ALERTA: Distribuições de treino e teste são diferentes!")
        else:
            print("✅ Distribuições de treino e teste são estatisticamente similares")
        
        # 4. Verificação de Data Leakage (Correlação bruta)
        print("\n🚨 VERIFICAÇÃO DE DATA LEAKAGE")
        suspicious = [(col, self.X_train[col].corr(self.y_train)) 
                    for col in self.X_train.columns 
                    if abs(self.X_train[col].corr(self.y_train)) > 0.95]
        
        if suspicious:
            for col, corr in suspicious:
                print(f"  ⚠️ {col}: correlação extrema = {corr:.4f}")
        else:
            print("  ✅ Nenhuma feature suspeita detectada")
        
        # 5. Feature Importance (Adaptado para Pipeline e Modelos Puros)
        print("\n📊 IMPORTÂNCIA DAS FEATURES")
        
        # Caso seja um Pipeline (ex: ElasticNet), extraímos o modelo de dentro dele
        inner_model = model.named_steps['model'] if hasattr(model, 'named_steps') else model
        
        importances = None
        if hasattr(inner_model, 'feature_importances_'):
            importances = inner_model.feature_importances_
        elif hasattr(inner_model, 'coef_'):
            importances = np.abs(inner_model.coef_)
            
        if importances is not None:
            feat_imp = pd.Series(importances, index=self.X_train.columns).sort_values(ascending=False).head(10)
            for feat, val in feat_imp.items():
                print(f"  {feat:<30} {val:.4f}")
        
        # 6. Gráficos de Diagnóstico (Resíduos vs Predito)
        self._plot_diagnostic_charts(
            self.y_train, y_pred_train, 
            y_test, y_pred_test,
            errors_train, errors_test,
            model_name
        )
        
        # Salvar diagnóstico
        diagnostic_result = {
            'mae_train': mae_train,
            'mae_test': mae_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mae_gap_percent': mae_gap,
            'rmse_gap_percent': rmse_gap,
            'suspicious_features': suspicious,
            'ks_test_pvalue': ks_pvalue
        }
        
        self.diagnostics_history[model_name] = diagnostic_result
        
        return diagnostic_result
    
    def _plot_diagnostic_charts(self, y_train, y_pred_train, y_test, y_pred_test, 
                               errors_train, errors_test, model_name):
        
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
    # VISUALIZAÇÃO E ANÁLISE
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
        """
        ✅ VERSÃO MELHORADA: Compara todos os modelos usando diagnósticos salvos.
        
        Melhorias:
        - Tratamento robusto de dados faltantes
        - Validação de estrutura do diagnostics_history
        - Métricas adicionais de análise
        - Formatação aprimorada
        """
        if not self.diagnostics_history:
            print("⚠️ Nenhum modelo diagnosticado ainda. Execute diagnose_model() primeiro.")
            return
        
        # Criar DataFrame a partir do histórico de diagnósticos
        data_for_df = []
        
        for model_name, diagnostic in self.diagnostics_history.items():
            row = {'model_name': model_name}
            row.update(diagnostic)
            data_for_df.append(row)
        
        df = pd.DataFrame(data_for_df)
        
        # Adicionar informações de treinamento se disponíveis
        if self.training_history:
            training_df = pd.DataFrame(self.training_history)
            df = df.merge(
                training_df[['model_name', 'training_time_minutes', 'n_trials', 'best_cv_mae']],
                on='model_name',
                how='left'
            )
        else:
            # Criar colunas vazias se não houver histórico de treinamento
            df['training_time_minutes'] = np.nan
            df['n_trials'] = np.nan
            df['best_cv_mae'] = np.nan
        
        # Ordenar por MAE de Teste
        df = df.sort_values('mae_test', ascending=True).reset_index(drop=True)
        
        # ===== CABEÇALHO =====
        print("\n" + "=" * 120)
        print(f"{'🏆 RANKING FINAL: PERFORMANCE NO CONJUNTO DE TESTE':^120}")
        print("=" * 120)
        
        # ===== TABELA PRINCIPAL =====
        print(f"\n{'Pos':<5} {'Modelo':<15} {'MAE Test':<12} {'Gap %':<10} "
            f"{'RMSE Test':<12} {'R² Test':<10} {'KS p-val':<10} {'Status':<20}")
        print("-" * 120)
        
        for idx, row in df.iterrows():
            # ✅ Classificação de status baseada em múltiplos critérios
            status = []
            status_emoji = "✅"
            
            # Critério 1: Overfitting (Gap de MAE)
            if row['mae_gap_percent'] > 50:
                status.append("Overfit Severo")
                status_emoji = "🚨"
            elif row['mae_gap_percent'] > 30:
                status.append("Overfit Alto")
                status_emoji = "❌"
            elif row['mae_gap_percent'] > 15:
                status.append("Overfit Moderado")
                status_emoji = "⚠️"
            else:
                status.append("OK")
            
            # Critério 2: Data Drift (KS Test)
            if row['ks_test_pvalue'] < 0.01:
                status.append("Drift Severo")
                status_emoji = "🚨"
            elif row['ks_test_pvalue'] < 0.05:
                status.append("Drift Detectado")
                if status_emoji == "✅":
                    status_emoji = "⚠️"
            
            # Critério 3: Features suspeitas
            n_suspicious = len(row['suspicious_features']) if isinstance(row['suspicious_features'], list) else 0
            if n_suspicious > 0:
                status.append(f"{n_suspicious} Feature(s) Suspeita(s)")
                if status_emoji == "✅":
                    status_emoji = "⚠️"
            
            # Critério 4: Performance (R²)
            if row['r2_test'] < 0.3:
                status.append("R² Baixo")
                if status_emoji == "✅":
                    status_emoji = "⚠️"
            
            status_str = " | ".join(status) if len(status) > 1 else status[0]
            
            print(f"{idx+1:<5} {row['model_name']:<15} {row['mae_test']:<12.6f} "
                f"{row['mae_gap_percent']:>8.2f}%  "
                f"{row['rmse_test']:<12.6f} {row['r2_test']:<10.4f} "
                f"{row['ks_test_pvalue']:<10.4f} {status_emoji} {status_str:<18}")
        
        print("-" * 120)
        
        # ===== RESUMO DO MELHOR MODELO =====
        best = df.iloc[0]
        print(f"\n{'🥇 MELHOR MODELO':^120}")
        print("-" * 120)
        print(f"Modelo:              {best['model_name']}")
        print(f"MAE (Test):          {best['mae_test']:.6f}")
        print(f"RMSE (Test):         {best['rmse_test']:.6f}")
        print(f"R² (Test):           {best['r2_test']:.4f}")
        print(f"Gap MAE:             {best['mae_gap_percent']:.2f}%")
        print(f"Gap RMSE:            {best['rmse_gap_percent']:.2f}%")
        print(f"KS Test p-value:     {best['ks_test_pvalue']:.4f}")
        
        if 'training_time_minutes' in best and pd.notna(best['training_time_minutes']):
            print(f"Tempo de treino:     {best['training_time_minutes']:.2f} minutos")
        if 'best_cv_mae' in best and pd.notna(best['best_cv_mae']):
            print(f"MAE (CV):            {best['best_cv_mae']:.6f}")
        
        # ===== ANÁLISE DE PROBLEMAS =====
        print(f"\n{'⚠️ ANÁLISE DE PROBLEMAS':^120}")
        print("-" * 120)
        
        # Overfitting
        overfit_models = df[df['mae_gap_percent'] > 15]
        if not overfit_models.empty:
            print(f"\n🔴 Modelos com Overfitting (Gap > 15%):")
            for _, row in overfit_models.iterrows():
                print(f"   • {row['model_name']:<15} Gap: {row['mae_gap_percent']:>6.2f}%")
        else:
            print("\n✅ Nenhum modelo com overfitting significativo")
        
        # Data Drift
        drift_models = df[df['ks_test_pvalue'] < 0.05]
        if not drift_models.empty:
            print(f"\n🔴 Modelos com Data Drift (p-value < 0.05):")
            for _, row in drift_models.iterrows():
                print(f"   • {row['model_name']:<15} p-value: {row['ks_test_pvalue']:.4f}")
        else:
            print("\n✅ Nenhum modelo com data drift detectado")
        
        # Features Suspeitas
        models_with_leakage = df[df['suspicious_features'].apply(lambda x: len(x) if isinstance(x, list) else 0) > 0]
        if not models_with_leakage.empty:
            print(f"\n🔴 Modelos com Features Suspeitas de Leakage:")
            for _, row in models_with_leakage.iterrows():
                n_suspicious = len(row['suspicious_features'])
                print(f"   • {row['model_name']:<15} {n_suspicious} feature(s) com correlação > 0.95")
                if isinstance(row['suspicious_features'], list):
                    for feat, corr in row['suspicious_features'][:3]:  # Mostrar top 3
                        print(f"       - {feat}: {corr:.4f}")
        else:
            print("\n✅ Nenhuma feature suspeita de leakage detectada")
        
        print("=" * 120 + "\n")
        
        # ===== GERAR VISUALIZAÇÕES =====
        self._plot_model_comparison(df)

    def _plot_model_comparison(self, df: pd.DataFrame):
        """
        ✅ VERSÃO MELHORADA: Visualização avançada e completa.
        
        Melhorias:
        - 8 gráficos em vez de 6
        - Cores contextuais (verde/amarelo/vermelho)
        - Anotações de valores
        - Grid para facilitar leitura
        - Tratamento de valores NaN
        """
        
        # Criar figura com 8 subplots (2 linhas x 4 colunas)
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('🏆 Análise Comparativa Completa de Modelos', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        models = df['model_name'].values
        x_pos = np.arange(len(models))
        
        # ===== 1. MAE: Train vs Test =====
        ax1 = fig.add_subplot(gs[0, 0])
        width = 0.35
        bars1 = ax1.bar(x_pos - width/2, df['mae_train'], width, 
                        label='Train', color='#3498db', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, df['mae_test'], width, 
                        label='Test', color='#e74c3c', alpha=0.8)
        
        ax1.set_xlabel('Modelo', fontsize=10)
        ax1.set_ylabel('MAE', fontsize=10)
        ax1.set_title('MAE: Treino vs Teste', fontsize=11, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # ===== 2. Gap de Overfitting (%) =====
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Cores baseadas em thresholds
        colors = []
        for gap in df['mae_gap_percent']:
            if gap < 10:
                colors.append('#2ecc71')  # Verde
            elif gap < 30:
                colors.append('#f39c12')  # Laranja
            else:
                colors.append('#e74c3c')  # Vermelho
        
        bars = ax2.bar(models, df['mae_gap_percent'], color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(10, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Ótimo (10%)')
        ax2.axhline(30, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Crítico (30%)')
        
        ax2.set_xlabel('Modelo', fontsize=10)
        ax2.set_ylabel('Gap (%)', fontsize=10)
        ax2.set_title('Gap de MAE - Indicador de Overfitting', fontsize=11, fontweight='bold')
        ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores
        for bar, gap in zip(bars, df['mae_gap_percent']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{gap:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # ===== 3. R² Score: Train vs Test =====
        ax3 = fig.add_subplot(gs[0, 2])
        
        ax3.plot(models, df['r2_train'], marker='o', linewidth=2, 
                markersize=8, label='Train', color='#3498db')
        ax3.plot(models, df['r2_test'], marker='s', linewidth=2, 
                markersize=8, label='Test', color='#e74c3c')
        
        ax3.set_xlabel('Modelo', fontsize=10)
        ax3.set_ylabel('R²', fontsize=10)
        ax3.set_title('R² Score (Train vs Test)', fontsize=11, fontweight='bold')
        ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax3.set_ylim([max(0, df['r2_test'].min() - 0.1), 
                    min(1.0, df['r2_test'].max() + 0.05)])
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # ===== 4. RMSE: Train vs Test =====
        ax4 = fig.add_subplot(gs[0, 3])
        
        bars1 = ax4.bar(x_pos - width/2, df['rmse_train'], width, 
                        label='Train', color='#9b59b6', alpha=0.8)
        bars2 = ax4.bar(x_pos + width/2, df['rmse_test'], width, 
                        label='Test', color='#e67e22', alpha=0.8)
        
        ax4.set_xlabel('Modelo', fontsize=10)
        ax4.set_ylabel('RMSE', fontsize=10)
        ax4.set_title('RMSE: Treino vs Teste', fontsize=11, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # ===== 5. Tempo de Treinamento =====
        ax5 = fig.add_subplot(gs[1, 0])
        
        if 'training_time_minutes' in df.columns and not df['training_time_minutes'].isna().all():
            # Cores baseadas no tempo
            times = df['training_time_minutes'].fillna(0)
            colors_time = plt.cm.YlOrRd(times / times.max())
            
            bars = ax5.barh(models, times, color=colors_time, edgecolor='black')
            ax5.set_xlabel('Minutos', fontsize=10)
            ax5.set_title('Tempo de Treinamento', fontsize=11, fontweight='bold')
            ax5.invert_yaxis()
            
            # Adicionar valores
            for bar, time in zip(bars, times):
                width = bar.get_width()
                if time > 0:
                    ax5.text(width, bar.get_y() + bar.get_height()/2.,
                            f' {time:.1f}m', va='center', fontsize=8)
        else:
            ax5.text(0.5, 0.5, 'Dados de tempo\nnão disponíveis', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=10)
            ax5.set_title('Tempo de Treinamento', fontsize=11, fontweight='bold')
        
        # ===== 6. KS Test p-value (Estabilidade) =====
        ax6 = fig.add_subplot(gs[1, 1])
        
        # Cores baseadas em significância
        colors_ks = []
        for p_val in df['ks_test_pvalue']:
            if p_val >= 0.05:
                colors_ks.append('#2ecc71')  # Verde - distribuições similares
            elif p_val >= 0.01:
                colors_ks.append('#f39c12')  # Laranja - diferença moderada
            else:
                colors_ks.append('#e74c3c')  # Vermelho - diferença significativa
        
        bars = ax6.bar(models, df['ks_test_pvalue'], color=colors_ks, 
                    alpha=0.7, edgecolor='black')
        ax6.axhline(0.05, color='red', linestyle='--', alpha=0.7, 
                    linewidth=2, label='α = 0.05')
        ax6.axhline(0.01, color='darkred', linestyle='--', alpha=0.7, 
                    linewidth=2, label='α = 0.01')
        
        ax6.set_xlabel('Modelo', fontsize=10)
        ax6.set_ylabel('p-value', fontsize=10)
        ax6.set_title('KS Test - Similaridade Train vs Test', fontsize=11, fontweight='bold')
        ax6.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Adicionar anotação
        ax6.text(0.98, 0.98, 'p > 0.05: Distribuições similares ✅\np < 0.05: Data drift ⚠️',
                transform=ax6.transAxes, fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ===== 7. Ranking por MAE (Test) =====
        ax7 = fig.add_subplot(gs[1, 2])
        
        df_sorted = df.sort_values('mae_test', ascending=False)
        
        # Cores para ranking (ouro, prata, bronze, outros)
        colors_rank = []
        for i in range(len(df_sorted)):
            if i == len(df_sorted) - 1:  # Primeiro lugar (invertido porque está de baixo pra cima)
                colors_rank.append('#FFD700')  # Ouro
            elif i == len(df_sorted) - 2:
                colors_rank.append('#C0C0C0')  # Prata
            elif i == len(df_sorted) - 3:
                colors_rank.append('#CD7F32')  # Bronze
            else:
                colors_rank.append('#95a5a6')  # Cinza
        
        bars = ax7.barh(df_sorted['model_name'], df_sorted['mae_test'], 
                        color=colors_rank, edgecolor='black', alpha=0.8)
        ax7.set_xlabel('MAE (Test)', fontsize=10)
        ax7.set_title('🏆 Ranking Final por MAE', fontsize=11, fontweight='bold')
        ax7.invert_yaxis()
        
        # Adicionar medalhas e valores
        for i, (bar, mae) in enumerate(zip(bars, df_sorted['mae_test'])):
            width = bar.get_width()
            rank = len(df_sorted) - i
            
            # Medalha
            if rank == 1:
                medal = "🥇"
            elif rank == 2:
                medal = "🥈"
            elif rank == 3:
                medal = "🥉"
            else:
                medal = f"{rank}º"
            
            ax7.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {mae:.4f} {medal}', va='center', fontsize=9, fontweight='bold')
        
        # ===== 8. Mapa de Calor - Métricas Normalizadas =====
        ax8 = fig.add_subplot(gs[1, 3])
        
        # Selecionar métricas para o heatmap
        metrics = ['mae_test', 'rmse_test', 'mae_gap_percent', 'r2_test']
        metrics_available = [m for m in metrics if m in df.columns]
        
        if metrics_available:
            # Normalizar métricas (0-1)
            df_norm = df[['model_name'] + metrics_available].copy()
            
            for col in metrics_available:
                if col == 'r2_test':
                    # R² maior é melhor - não inverter
                    df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                else:
                    # Outras métricas: menor é melhor - inverter
                    df_norm[col] = 1 - ((df[col] - df[col].min()) / (df[col].max() - df[col].min()))
            
            # Criar matriz para heatmap
            heatmap_data = df_norm[metrics_available].values.T
            
            im = ax8.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            # Configurar eixos
            ax8.set_xticks(np.arange(len(models)))
            ax8.set_yticks(np.arange(len(metrics_available)))
            ax8.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
            
            # Labels mais descritivos
            metric_labels = {
                'mae_test': 'MAE Test',
                'rmse_test': 'RMSE Test',
                'mae_gap_percent': 'Gap MAE %',
                'r2_test': 'R² Test'
            }
            ax8.set_yticklabels([metric_labels.get(m, m) for m in metrics_available], fontsize=9)
            
            ax8.set_title('Mapa de Performance\n(Verde=Melhor, Vermelho=Pior)', 
                        fontsize=11, fontweight='bold')
            
            # Adicionar valores nas células
            for i in range(len(metrics_available)):
                for j in range(len(models)):
                    value = heatmap_data[i, j]
                    color = 'white' if value < 0.5 else 'black'
                    ax8.text(j, i, f'{value:.2f}',
                            ha="center", va="center", color=color, fontsize=8)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax8, fraction=0.046, pad=0.04)
            cbar.set_label('Score Normalizado', fontsize=9)
        else:
            ax8.text(0.5, 0.5, 'Dados insuficientes\npara heatmap', 
                    ha='center', va='center', transform=ax8.transAxes, fontsize=10)
            ax8.set_title('Mapa de Performance', fontsize=11, fontweight='bold')
        
        # ===== RODAPÉ COM INFORMAÇÕES =====
        footer_text = f"""
        📊 Total de modelos: {len(df)} | 🥇 Melhor: {df.iloc[0]['model_name']} (MAE: {df.iloc[0]['mae_test']:.6f})
        ⚠️ Legenda Gap: Verde (<10%) = Ótimo | Laranja (10-30%) = Moderado | Vermelho (>30%) = Crítico
        """
        
        fig.text(0.5, 0.01, footer_text, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        plt.show()

    def _plot_model_comparison_v2(self, df: pd.DataFrame):
        """Visualização avançada baseada nos resultados do diagnóstico."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle('Análise Comparativa de Modelos (Baseado em Diagnósticos)', fontsize=16, fontweight='bold')
        
        models = df['model_name']
        x = range(len(models))
        
        # 1. MAE Train vs Test
        axes[0, 0].bar(x, df['mae_train'], width=0.4, label='Train', alpha=0.7)
        axes[0, 0].bar([p + 0.4 for p in x], df['mae_test'], width=0.4, label='Test', alpha=0.7)
        axes[0, 0].set_title('MAE: Treino vs Teste')
        axes[0, 0].set_xticks([p + 0.2 for p in x])
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()

        # 2. Gap de Overfitting (%)
        colors = ['green' if g < 15 else 'red' for g in df['mae_gap_percent']]
        axes[0, 1].bar(models, df['mae_gap_percent'], color=colors, alpha=0.6)
        axes[0, 1].axhline(15, color='black', linestyle='--', alpha=0.3)
        axes[0, 1].set_title('Gap de MAE % (Overfitting)')
        axes[0, 1].set_ylabel('Percentual (%)')

        # 3. R² Score Test
        axes[0, 2].plot(models, df['r2_test'], marker='o', linewidth=2, color='blue')
        axes[0, 2].set_title('R² Score (Teste)')
        axes[0, 2].set_ylim(df['r2_test'].min() - 0.05, 1.02)
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Tempo de Treinamento
        axes[1, 0].barh(models, df['training_time_minutes'], color='coral')
        axes[1, 0].set_title('Tempo de Treino (Minutos)')

        # 5. KS Test p-value (Estabilidade dos Dados)
        axes[1, 1].bar(models, df['ks_test_pvalue'], color='purple', alpha=0.5)
        axes[1, 1].axhline(0.05, color='red', linestyle='--', label='Alpha 0.05')
        axes[1, 1].set_title('P-Value KS Test (Distr. Treino vs Teste)')
        axes[1, 1].legend()

        # 6. Ranking MAE Test (Horizontal)
        df_sorted = df.sort_values('mae_test', ascending=False)
        axes[1, 2].barh(df_sorted['model_name'], df_sorted['mae_test'], color='gold')
        axes[1, 2].set_title('Ranking Final (Menor MAE)')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    """
    def compare_all_models(self):
        
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
        plt.show()"""
    
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