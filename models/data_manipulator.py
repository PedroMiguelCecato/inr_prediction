import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class INRDataManipulation:
    def __init__(self, path=None, sheet_name="TTR"):
        self.path = path
        self.sheet_name = sheet_name

        self.data_original = None
        self.data_weekly = None
        self.data_final = None
        self.features_dose = None
        self.features_inr = None

        if path is not None:
            self.read_data()
            self.fit_data()
        else:
            print("Caminho do arquivo não definido. Use set_path() para definir o caminho.")

    # =========================
    # Infraestrutura básica
    # =========================
    def check_path_and_data(self, require_final=True):
        if self.path is None:
            raise ValueError("Caminho do arquivo não definido. Use set_path() para definir o caminho.")
        
        if self.data_original is None:
            self.read_data()

        if require_final and self.data_final is None:
            self.fit_data()

    def set_path(self, path=None, sheet_name="TTR"):
        self.path = path
        self.sheet_name = sheet_name

        self.data_original = None
        self.data_weekly = None
        self.data_final = None
        self.feature_columns = None

        self.check_path_and_data()

    def get_data_original(self):
        self.check_path_and_data(require_final=False)
        return self.data_original

    def get_data_weekly(self):
        self.check_path_and_data()
        return self.data_weekly

    def get_data_final(self):
        self.check_path_and_data()
        return self.data_final

    def get_features_dose(self):
        self.check_path_and_data()
        return self.features_dose

    def get_features_inr(self):
        self.check_path_and_data()
        return self.features_inr
    
    # =========================
    # Leitura e execução
    # =========================
    def read_data(self):
        # tenta excel primeiro
        try:
            df = pd.read_excel(self.path, sheet_name=self.sheet_name)
        except Exception as e_excel:
            # tenta CSV como fallback
            try:
                df = pd.read_csv(self.path)
            except Exception as e_csv:
                raise ValueError(f"Erro ao ler arquivo: Excel-> {e_excel}; CSV-> {e_csv}")

        self.data_original = df.copy()

    def fit_data(self):
        self.check_path_and_data(require_final=False)
        data = self.data_original.copy()

        # tentar extrair low_range/high_range de 'Unnamed: 15'
        low_val = None
        high_val = None
        if 'Unnamed: 15' in data.columns:
            low_val = data.loc[0, 'Unnamed: 15'] if 0 in data.index else None
            high_val = data.loc[1, 'Unnamed: 15'] if 1 in data.index else None

        # criar colunas low_range/high_range se ausentes
        if low_val is not None:
            data['low_range'] = low_val
        else:
            data['low_range'] = np.nan

        if high_val is not None:
            data['high_range'] = high_val
        else:
            data['high_range'] = np.nan

        # filtrar colunas necessárias
        needed = []
        for c in ['Test Date', 'DOSE SEMANAL', 'INR', 'INR Diff']:
            if c in data.columns:
                needed.append(c)
            else:
                raise ValueError(f"Coluna esperada '{c}' não encontrada no arquivo.")

        data_filtred = data[needed + ['low_range', 'high_range']].copy()
        data_filtred = data_filtred.rename(columns={'DOSE SEMANAL': 'dose_semanal',
                                                    'INR Diff': 'inr_diff',
                                                    'INR': 'inr',
                                                    'Test Date': 'test_date'})

        # garantir tipos adequados
        data_filtred["test_date"] = pd.to_datetime(data_filtred["test_date"], errors="coerce")
        data_filtred["dose_semanal"] = pd.to_numeric(data_filtred.get("dose_semanal"), errors="coerce")
        data_filtred["inr"] = pd.to_numeric(data_filtred.get("inr"), errors="coerce")
        data_filtred["inr_diff"] = pd.to_numeric(data_filtred.get("inr_diff"), errors="coerce")
        data_filtred["low_range"] = pd.to_numeric(data_filtred.get("low_range"), errors="coerce")
        data_filtred["high_range"] = pd.to_numeric(data_filtred.get("high_range"), errors="coerce")

        # ordenar
        data_filtred = data_filtred.sort_values('test_date').reset_index(drop=True)

        # preencher inr NaN com média
        media_inr = data_filtred["inr"].mean(skipna=True)
        data_filtred["inr"] = data_filtred["inr"].fillna(media_inr)

        # recalcular inr_diff
        data_filtred['inr_diff'] = data_filtred['inr'].diff().round(3)
        if len(data_filtred) > 0:
            data_filtred.loc[0, 'inr_diff'] = 0.0

        # gerar dataset semanal, com novas features e salvar em self.data_final
        self.data_weekly = self.weekly(data=data_filtred)
        self.data_final = self.create_time_features(data=self.data_weekly)
        return self.data_final

    # =========================
    # Manipulação de dados
    # =========================
    def weekly(self, data, freq_days=7, tolerance_days=3):
        # cópia defensiva e garantias
        df = data.copy()
        orig = df[['test_date', 'inr', 'dose_semanal', 'low_range', 'high_range']].copy()
        orig = orig.dropna(subset=['test_date']).reset_index(drop=True)

        start = orig['test_date'].min()
        end = orig['test_date'].max()
        weekly_dates = pd.date_range(start=start, end=end, freq=f'{int(freq_days)}D')

        rows = []
        tol = pd.Timedelta(days=int(tolerance_days))

        for d in weekly_dates:
            # procurar exame original dentro da janela +/- tol (prioriza exames reais)
            diffs = (orig['test_date'] - d).abs()
            close_mask = diffs <= tol

            if close_mask.any():
                # escolher o exame mais próximo (se empate, o mais recente)
                candidates = orig[close_mask].copy()
                candidates['abs_diff'] = (candidates['test_date'] - d).abs()
                candidates = candidates.sort_values(['abs_diff', 'test_date'], ascending=[1, 0])
                chosen = candidates.iloc[0].to_dict()
                chosen['test_date'] = d
                chosen['generated'] = 0
                rows.append(chosen)
                continue

            # se não há exame próximo, buscar prev e next na tabela original
            prev_mask = orig[orig['test_date'] < d]
            next_mask = orig[orig['test_date'] > d]

            prev_row = prev_mask.iloc[-1] if len(prev_mask) > 0 else None
            next_row = next_mask.iloc[0] if len(next_mask) > 0 else None

            # caso sem prev e next (não deveria acontecer dentro do intervalo), pular
            if prev_row is None and next_row is None:
                continue

            # bordas: replicar valor existente
            if prev_row is None:
                inr_val = float(next_row['inr'])
                dose_val = next_row['dose_semanal']
                low_val = next_row['low_range']
                high_val = next_row['high_range']
            elif next_row is None:
                inr_val = float(prev_row['inr'])
                dose_val = prev_row['dose_semanal']
                low_val = prev_row['low_range']
                high_val = prev_row['high_range']
            else:
                # INTERPOLAÇÃO LINEAR: calcula fração entre prev_date e next_date
                prev_date = pd.to_datetime(prev_row['test_date'])
                next_date = pd.to_datetime(next_row['test_date'])
                total_delta = (next_date - prev_date).days

                if total_delta == 0:
                    # proteção contra divisão por zero (mesma data nos dados originais)
                    fraction = 0.5
                else:
                    fraction = (d - prev_date).days / total_delta
                    # garantir limites [0,1]
                    fraction = max(0.0, min(1.0, fraction))

                prev_inr = float(prev_row['inr'])
                next_inr = float(next_row['inr'])
                inr_val = prev_inr + (next_inr - prev_inr) * fraction

                # repetir dose/low/high a partir do prev (escolha sua preferida)
                dose_val = prev_row['dose_semanal']
                low_val = prev_row['low_range']
                high_val = prev_row['high_range']

            rows.append({'test_date': d,
                         'inr': float(inr_val),
                         'dose_semanal': dose_val,
                         'low_range': low_val,
                         'high_range': high_val,
                         'generated': 1})

        # montar DF resultante
        weekly_df = pd.DataFrame(rows)
        weekly_df = weekly_df.sort_values('test_date').reset_index(drop=True)

        # recalcular inr_diff = inr_current - inr_previous
        if not weekly_df.empty:
            weekly_df['inr_diff'] = weekly_df['inr'].diff().round(3)
            weekly_df.loc[0, 'inr_diff'] = 0.0
        else:
            weekly_df['inr_diff'] = pd.Series(dtype=float)

        weekly_df['inr'] = weekly_df['inr'].round(3)
        weekly_df['inr_diff'] = weekly_df['inr_diff'].fillna(0.0).round(3)

        cols_order = ['test_date', 'dose_semanal', 'inr', 'inr_diff', 'low_range', 'high_range', 'generated']
        existing_cols = [c for c in cols_order if c in weekly_df.columns]
        weekly_df = weekly_df[existing_cols]

        return weekly_df

    def create_time_features(self, data, date_col="test_date", target_col="inr",
                             lags=[1,2,3,4], roll_windows=[2,4]):
        df = data.copy()

        # Features temporais 
        df['weekofyear'] = df[date_col].dt.isocalendar().week.astype(int)
        df['month'] = df[date_col].dt.month.astype(int)
        df['year'] = df[date_col].dt.year.astype(int)

        # Lags do INR (DEPENDÊNCIA TEMPORAL DO ALVO)
        for lag in lags:
            df[f'inr_lag_{lag}'] = df[target_col].shift(lag)

        # Médias móveis (SUAVIZAÇÃO DA DINÂMICA DO INR)
        for w in roll_windows:
            # média dos últimos w valores, deslocada para evitar vazamento de informação
            df[f'inr_roll_mean_{w}'] = (df[target_col].shift(1).rolling(window=w, min_periods=1).mean())

        # Remoção das primeiras linhas sem lags suficientes 
        min_lag = max(lags) if len(lags) > 0 else 0
        features_df = df.iloc[min_lag:].reset_index(drop=True)

        # Seleção das features
        base_cols_f = ['dose_semanal', 'generated', 'weekofyear', 'month']
        base_cols_i = ['inr', 'generated', 'weekofyear', 'month']

        lag_cols = [f'inr_lag_{lag}' for lag in lags]
        roll_cols = [f'inr_roll_mean_{w}' for w in roll_windows]

        feature_cols_f = base_cols_f + lag_cols + roll_cols
        feature_cols_i = base_cols_i + lag_cols + roll_cols
        self.features_dose = [c for c in feature_cols_f if c in features_df.columns]
        self.features_inr = [c for c in feature_cols_i if c in features_df.columns]
        
        return features_df

    # =========================
    # Plot principal
    # =========================
    def plot_inr(self, low=2.5, high=3.5):
        self.check_path_and_data(require_final=True)

        low = self.data_final['low_range'].iloc[0] if 'low_range' in self.data_final.columns and not self.data_final['low_range'].isna().all() else low
        high = self.data_final['high_range'].iloc[0] if 'high_range' in self.data_final.columns and not self.data_final['high_range'].isna().all() else high

        plt.figure(figsize=(12,6))
        plt.plot(self.data_final['test_date'], self.data_final['inr'], marker='o', linestyle='-')
        plt.axhspan(low, high, alpha=0.2, color='green', label=f'Faixa Alvo ({low}-{high})')
        plt.axhline(y=low, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.axhline(y=high, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.xlabel("Data do Teste")
        plt.ylabel("Valor de INR")
        plt.title("Série Temporal do INR ao longo do tempo")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
