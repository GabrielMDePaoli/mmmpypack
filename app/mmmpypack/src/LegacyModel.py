import pandas as pd
import numpy as np
from datetime import timedelta
# Métricas
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.inspection import permutation_importance
# Modelos
from sklearn.linear_model import LinearRegression

# O que fazer com avisos?
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
pd.options.mode.chained_assignment = None

class LegacyModel:
    '''
    ### Definição

    A classe Model é a classe responsável por conter os dados sobre o modelo que está dentro dela.
    '''
    # Quando eu já tenho o modelo e desejo importá-lo
    def __init__(self,
                model:LinearRegression,
                version:str,
                product:str,
                holidays:pd.DataFrame=None,
                secondary_model:LinearRegression|None=None) -> None:
        '''
        ### Definição

        Método para importar um modelo >=v2.2 já existente para a classe. Ao utilizar esse método a classe ficara travada e não passara mais pelo processo de dataprep.

        Este método vai adaptar os transformadores com o objetivo de otimizar os processos de transformação, diminuindo o tempo de execução para realziar predições.

        ### Parametros

        •	model (obrigatório | modelo sklearn treinado) - Qualquer modelo sklearn treinado.

        •	version (str | obrigatório) - String que define a versão do modelo que esta sendo salvo.

        •	product (str | obrigatório) - String que contenha o nome do produto do modelo que será treinado. Essencial para todo o processo de transformação e saída do resultado. O nome do produto deve seguir a regra das nomenclatura: {frente}_{produto} (ex. NET_BAL).
        
        •	holidays (pd.DataFrame | obrigatório) - Dataframe que contem diversas datas de feriados para anos anteriores e futuros.

        •	secondary_model (opcional | modelo sklearn treinado) - Qualquer modelo sklearn treinado que servirá para somar com as previsões do modelo original.
        '''
        # Informações sobre o modelo
        self.model = model
        self.secondary_model = secondary_model
        self.root_features = ['_'.join(feature.split('_')[:6]) for feature in model.feature_names_in_]
        if secondary_model != None:
            self.root_features.extend(['_'.join(feature.split('_')[:6]) for feature in secondary_model.feature_names_in_])
        self.root_features = list(set(self.root_features))
        self.version = version
        self.product = product
        # Base de feriados
        try:
            if isinstance(holidays, pd.DataFrame):
                # Salvar nova base de feriados
                try:
                    holidays["date"] = pd.to_datetime(holidays["date"])
                    holidays.set_index("date", inplace=True)
                except:
                    pass
                self.holidays = holidays.copy()
            elif isinstance(self.holidays, pd.DataFrame) and isinstance(holidays, None):
                pass # Manter holidays já existente
            elif isinstance(self.holidays, None) and isinstance(holidays, None):
                raise ValueError('holidays ainda não foi definido e precisa ser definido.')
        except:
            raise ValueError('holidays ainda não foi definido e precisa ser definido.')
        # Condigurações inciais
        # Salvar todas as features que serão utilizadas (sem agrupamento e com agrupamento separadamente)
        all_features = list(model.feature_names_in_)
        if secondary_model != None:
            all_features.extend(list(secondary_model.feature_names_in_))
            # self.all_features = list(set(all_features))
        ## Variável com o de para de cada variável e como ela deverá ser transformada
        feature_dict = list()
        ## Definindo quais features precisam de quais transformações
        for feature in all_features:
            original = '_'.join(feature.split('_')[:6])
            transformation = '_'.join(feature.split('_')[6:])
            feature_dict.append([original, transformation])
        self.feature_dict = feature_dict.copy()
        return

    # Processos de predição
    def modelpred_transform_inv(self, X:pd.DataFrame) -> pd.DataFrame:
        '''
        ### Definição

        Controla as transformações dos investimentos após já terem passado pelo processo de MinMaxScaler.

        ### Parametros

        •	X (obrigatório | pd.DataFrame) - Dataframe contendo os investimentos necessários para a predição.
        '''
        # Variáveis de uso local
        feature_dict = self.feature_dict.copy()

        with np.errstate(divide = 'ignore'):
            for feature in feature_dict:
                transformation = feature[1]
                feature = feature[0]
                # Se for tipo data, pular
                if feature.startswith('DTA'): continue
                # Diluição
                if 'ADSTOCK' in transformation: dilution_type = 'ADSTOCK'
                elif 'MM2' in transformation: dilution_type = 'MM2'
                elif 'MM30' in transformation: dilution_type = 'MM30'
                else: dilution_type = ''
                # Transformação
                if 'LOG' in transformation:  transformation_calc = 'LOG'
                if 'LOG2' in transformation:  transformation_calc = 'LOG2'
                elif 'SQRT' in transformation:  transformation_calc = 'SQRT'
                elif 'ARCTAN' in transformation:  transformation_calc = 'ARCTAN'
                # Lista com valores de investimento
                inv_list = X[feature].to_list()
                # Investimentos transformados
                transformed_inv = self.modelpred_apply_transformation(
                    inv_list,
                    dilution_type,
                    transformation_calc
                )
                # Salvando a lista de investimentos transformados
                X[f'{feature}_{transformation}'] = transformed_inv

        # Removendo valores negativos (não é possível gerar vendas negativas ou investimentos negativos), nem NaNs pois eles são 0
        X[(X < 0) | (np.isinf(X)) | (np.isnan(X))] = 0
        return X

    def modelpred_apply_transformation(self,
                                       inv_list:list,
                                       dilution_type:str,
                                       transformation_calc:str) -> list:
        '''
        ### Definição

        Aplica as transformações de acordo com parametros.

        ### Parametros

        •	inv_list (obrigatório | list) - Lista com todos os investimentos da coluna.

        •	col_name (obrigatório | str) - nome da coluna sendo modificada.

        •	lag_ammount (obrigatório | int) - Quantidade de dias de lag.

        •	dilution_type (obrigatório | str) - Tipo de diluição ('MMY', 'ADSTOCKN', 'ORIG').

        •	dilution_intensity (obrigatório | int|float) - Intensidade da diluição.

        •	transformation_calc (obrigatório | str) - Calculo a ser realziado ('LAG1', 'LAG3', 'YJ').
        '''
        # Aplicando diluição
        if dilution_type == 'MM2':
            mm2, vl_mm2 = [], []
            for value in inv_list:
                vl_mm2 = [value] + vl_mm2[:2 - 1]
                mm2.append(sum(vl_mm2) / len(vl_mm2))
            inv_list = mm2.copy()
        elif dilution_type == 'MM30':
            mm30, vl_mm30 = [], []
            for value in inv_list:
                vl_mm30 = [value] + vl_mm30[:30 - 1]
                mm30.append(sum(vl_mm30) / len(vl_mm30))
            inv_list = mm30.copy()
        elif dilution_type == 'ADSTOCK':
            adstock, vl_a = [], []
            for value in inv_list:
                vl_a = [value] + vl_a[:5]
                adstock.append(sum([v/2**i for i, v in enumerate(vl_a)]))
            inv_list = adstock.copy()
            # inv_list = self.adstocking(1/2, media_inv=inv_list)
        elif dilution_type == '': # Mantem os investimentos originais
            pass

        # Aplicando transformações
        if transformation_calc == 'LOG':
            inv_list = np.log(inv_list)
        elif transformation_calc == 'LOG2':
            inv_list = np.log(inv_list) ** 2
        elif transformation_calc == 'SQRT':
            inv_list = np.sqrt(inv_list)
        elif transformation_calc == 'ARCTAN':
            inv_list = np.arctan(inv_list)
        return inv_list

    def modelpred_prepare_X(self, X:pd.DataFrame) -> pd.DataFrame:
        '''
        ### Descrição

        Método que realiza a transformação do X para os modelos.

        ### Parametros

        •	X (obrigatório | pd.DataFrame) - Dataframe contendo os investimentos necessários para a predição.
        '''
        # Variáveis de uso local
        root_features = self.root_features.copy()

        # Remover variáveis que o modelo não utiliza
        root_features = list(set(self.root_features))
        try:
            X = X[root_features]
        except:
            X = self.mousse_sazonal_v3(X.copy())
            X = X[root_features]

        # Aplicando transformações
        X = self.modelpred_transform_inv(X.copy())
        self.last_X = X.copy()

        return X

    def modelpred_pipeline(self, X:pd.DataFrame):
        '''
        ### Descrição

        Método que realiza a predição com base em um X fornecido pelo usuário.

        ### Parametros

        •	X (obrigatório | pd.DataFrame) - Dataframe contendo os investimentos necessários para a predição.
        '''
        # Variáveis de uso local
        model = self.model
        secondary_model = self.secondary_model
        
        # Preparar X para os modelos
        X = self.modelpred_prepare_X(X.copy())

        # Realizar predição
        y_pred = model.predict(X[model.feature_names_in_])
        if secondary_model != None:
            y_pred += secondary_model.predict(X[secondary_model.feature_names_in_])

        return y_pred

    # Processos de monitoramento
    def monitor_get_model_metrics(self,
                                  X:pd.DataFrame,
                                  y:pd.Series,
                                  period:list[str],) -> dict:
        '''
        ### Definição

        Gera um relatório de performance do modelo, num determinado periodo e base que for enviada.

        ### Parametros

        •	X (obrigatório | pd.DataFrame) - Dataframe contendo os investimentos necessários para a predição.

        •	y (obrigatório | pd.Series) - Série com o resultado real de vendas que o modelo deve atingir.

        •	period (obrigatório | list) - Lista com a data incial e final da analise.
        '''
        # Realizar predição
        y_pred = self.modelpred_pipeline(X)

        # Cortando os resultados de acordo com period
        pred_df = pd.Series(data=y_pred, index=y.index.copy(), name=f'PRED_{self.product}')
        pred = pred_df.loc[period[0]:period[1]]
        real = y.loc[period[0]:period[1]]

        # Salvando resultados
        monit_dict = {
            'y_pred': pred.loc[period[0]:period[1]],
            'y_real': real.loc[period[0]:period[1]],
            'removed_days': pd.DataFrame(),
            'MAPE': mean_absolute_percentage_error(real, pred),
            'R2': r2_score(real, pred),
            'sMAPE': self.smape(real, pred),
            'DIF1': ((sum(real)) / sum(pred)) - 1,
            'DIF2': ((sum(pred)) / sum(real)) - 1,
            'ORCAMENTO USADO': self.calculate_inv_usage(X)
        }

        return monit_dict

    def smape(self, real, pred):
        real = np.array(real)
        pred = np.array(pred)
        return (100/len(real) * np.sum(2 * np.abs(pred - real) / (np.abs(real) + np.abs(pred)))) / 100

    # Funções de analise do modelo
    def get_vars_and_coefs(self) -> pd.DataFrame:
        '''
        ### Descrição

        Gera uma tabela com o coeficiente do intercepto e das demais features.
        '''
        # Variáveis locais
        model = self.model
        secondary_model = self.secondary_model
        # Código
        intercept = model.intercept_
        features_list = list( model.feature_names_in_)
        coef_list = list(model.coef_)
        if self.secondary_model != None:
            intercept += secondary_model.intercept_
            for coef_idx, feature in enumerate(list(secondary_model.feature_names_in_)):
                if feature in features_list:
                    feature_list_index = features_list.index(feature)
                    coef_list[feature_list_index] += secondary_model.coef_[coef_idx]
                else:
                    features_list.append(feature)
                    coef_list.append(secondary_model.coef_[coef_idx])
        intercept_df = pd.DataFrame(data={'FEATURE':['INTERCEPT'],'COEF':[intercept]})
        var_coef_df = pd.DataFrame({'FEATURE':features_list,'COEF':coef_list})
        var_coef_df = pd.concat([intercept_df,var_coef_df], axis=0)
        return var_coef_df

    def calculate_feature_importance(self,
                                     X:pd.DataFrame,
                                     y:pd.Series,
                                     period:list[str],) -> pd.DataFrame:
        '''
        ### Definição

        Gera um relatório de importancia das features do modelo, num determinado periodo e base que for enviada.

        ### Parametros

        •	X (obrigatório | pd.DataFrame) - Dataframe contendo os investimentos necessários para a predição.

        •	y (obrigatório | pd.Series) - Série com o resultado real de vendas que o modelo deve atingir.

        •	period (obrigatório | list) - Lista com a data incial e final da analise.
        '''
        model = self.model
        secondary_model = self.secondary_model

        # Preparar X dos modelos
        X = self.modelpred_prepare_X(X.copy())

        # Coletando apenas o periodo desejado
        X = X.loc[period[0]:period[1]]
        y = y.loc[period[0]:period[1]]

        # Calcular importancia
        result = permutation_importance(model, X[model.feature_names_in_], y, n_repeats=33, random_state=333)
        df = pd.DataFrame({'Features':model.feature_names_in_, 'Importancia':result['importances_mean']})
        if secondary_model != None:
            result2 = permutation_importance(secondary_model, X[secondary_model.feature_names_in_], y, n_repeats=33, random_state=333)
            df = pd.concat([df, pd.DataFrame({'Features':secondary_model.feature_names_in_, 'Importancia':result2['importances_mean']})])

        return df

    def calculate_saturation(self,
                             inv_df:pd.DataFrame,
                             periods:list,
                             sat_interval:list,
                             step_size:int,
                             sat_type:str) -> pd.DataFrame:
        '''
        ### Definição

        Gera um relatório de saturação de cada produto / veículo / mídia existente no modelo.

        ### Parametros

        •	inv_df (obrigatório | pd.DataFrame) - Dataframe contendo os investimentos necessários para a predição.

        •	periods (obrigatório | list[datetime]) - Lista com o inicio e final do periodo quando a saturação será realizada.

        •	sat_interval (obrigatório | int) - Lista com o inicio e final do intervalo que a saturação será realizada (ex. [0,300]).

        •	step_size (obrigatório | int) - Inteiro com o pulo de cada saturação (ex. 25).

        •	sat_type (obrigatório | str) - String com o que será saturado, pode ser um destes valores: "Produtos", "Veículos", "Mídia".
        '''
        # Filtrar alguns dias antes do periodo definido #
        initial_date_postpone = 15
        periods[0] = periods[0] - timedelta(days=initial_date_postpone)
        inv_df = inv_df.loc[periods[0]:periods[1]]
        # --------------------------------------------- #

        # Configurar quais serão os itens saturados #
        to_saturate = []
        if sat_type in ['Produtos', 'Veículos', 'Mídia']:
            saturable_features = [feature for feature in self.root_features if not feature.startswith('DTA')]
            if sat_type == 'Produtos':
                for feature in saturable_features:
                    product_regex = '_'.join(feature.split('_')[1:3])
                    to_saturate.append(product_regex)
            if sat_type == 'Veículos':
                ginfo = self.group_info.copy()
                for feature in saturable_features:
                    midia = feature.split('_')[0]
                    vehicle = feature.split('_')[3]
                    temp_ginfo = ginfo[(ginfo['SIGLA VEICULO'] == vehicle) & (ginfo['COMBINAR'] == True)]
                    if len(temp_ginfo) > 0:
                        # Salvar veiculos origem do grupo
                        for idx, row in temp_ginfo.iterrows():
                            vehicle_regex = f'{midia}\\w+{row["SIGLA"]}'
                            to_saturate.append(vehicle_regex)
                    else:
                        vehicle_regex = f'{midia}\\w+{vehicle}'
                        to_saturate.append(vehicle_regex)
            if sat_type == 'Mídia':
                for feature in saturable_features:
                    midia_regex = feature.split("_")[0]
                    to_saturate.append(midia_regex)
        else:
            raise ValueError('sat_type precisa ser um destes valores: ["Produtos", "Veículos", "Mídia"]')
        to_saturate = list(set(to_saturate))
        to_saturate = ['FIN_INV$'] + to_saturate
        # ----------------------------------------- #

        # LOOP de saturação #
        sat_data = {'AFETADO':[],'MULTIPLICADOR':[],'VENDAS':[],'INVESTIMENTO':[],'CUSTO':[],'INVESTIMENTO INCREMENTAL':[],'VENDA INCREMENTAL':[],'CUSTO INCREMENTAL':[]}
        for _, saturate_item in enumerate(to_saturate):
            item_change = True
            for sat_mult in range(sat_interval[0], sat_interval[1]+1, step_size):
                # Multiplicar as colunas desejadas
                temp_inv_df = inv_df.copy()
                temp_item_df = temp_inv_df.filter(regex=saturate_item)
                temp_item_df = temp_item_df.multiply(sat_mult / 100)
                temp_inv_df[temp_item_df.columns] = temp_item_df
                # Predição
                sales_pred = self.modelpred_pipeline(temp_inv_df)
                # Salvando valores para plot
                sat_data['AFETADO'].append(saturate_item) # Parte da base afetada
                sat_data['MULTIPLICADOR'].append(sat_mult) # Multiplicador utilizado
                sat_data['VENDAS'].append(sales_pred[initial_date_postpone:].sum()) # Soma de todas as vendas
                sat_data['INVESTIMENTO'].append(temp_item_df.sum().sum()) # Soma dos investimentos
                sat_data['CUSTO'].append(sat_data['INVESTIMENTO'][-1] / sat_data['VENDAS'][-1])
                if not item_change:
                    sat_data['INVESTIMENTO INCREMENTAL'].append(sat_data['INVESTIMENTO'][-1] - sat_data['INVESTIMENTO'][-2])
                    sat_data['VENDA INCREMENTAL'].append(sat_data['VENDAS'][-1] - sat_data['VENDAS'][-2])
                    sat_data['CUSTO INCREMENTAL'].append(sat_data['INVESTIMENTO INCREMENTAL'][-1] / sat_data['VENDA INCREMENTAL'][-1])
                elif item_change:
                    item_change = False
                    sat_data['INVESTIMENTO INCREMENTAL'].append(None)
                    sat_data['VENDA INCREMENTAL'].append(None)
                    sat_data['CUSTO INCREMENTAL'].append(None)
        # ----------------- #

        sat_df = pd.DataFrame(sat_data)
        return sat_df

    def calculate_inv_usage(self, X):
        root_features = list(set(self.root_features))
        inv_root_features = [feature for feature in root_features if 'FIN_INV' in feature]

        X = X.filter(regex='FIN_INV$')
        total_X_inv = X.sum().sum()
        model_X_inv = X[inv_root_features].sum().sum()

        return model_X_inv / total_X_inv

    # Função de transformações em geral
    def adstocking(self,
                   adstock_rate:float,
                   media_inv:list) -> list:
        '''
        ### Descrição

        Método que aplica o efeito de diluição ADSTOCK

        ### Parametros

        •	adstock_rate (float | obrigatório) - Taxa de adstock que será aplicada.

        •	media_inv (list | obrigatório) - Lista contendo os investimentos que terão o efeito de adstock.
        '''
        adstocked_media = []
        for i in range(len(media_inv)):
            if i == 0:
                adstocked_media.append(media_inv[i])
            else:
                adstocked_media.append(media_inv[i] + adstock_rate * adstocked_media[i-1])
        return adstocked_media

    def mousse_sazonal_v3(self,
                          df:pd.DataFrame) -> pd.DataFrame:
        '''
        ### Descrição

        Método que aplica a sazonalidade a base de investimentos e vendas.
        '''
        # Adicionando valores binários ao dataset
        holidays = self.holidays.copy()

        ## Dia da semana
        df["weekday"] = [value.weekday() for value in df.index]
        weekdays = ['SEGUNDA', 'TERCA', 'QUARTA', 'QUINTA', 'SEXTA', 'SABADO', 'DOMINGO']
        for weekday in weekdays:
            temp = [1 if value == weekdays.index(weekday) else 0 for value in df["weekday"].values]
            df[f'DTA_DIA_{weekday}'] = temp

        ## Número do mês
        df["month"] = [value.month for value in df.index]
        months = ['JANEIRO', 'FEVEREIRO', 'MARÇO', 'ABRIL', 'MAIO', 'JUNHO', 'JULHO', 'AGOSTO', 'SETEMBRO', 'OUTUBRO', 'NOVEMBRO', 'DEZEMBRO']
        for month in months:
            temp = [1 if value == months.index(month)+1 else 0 for value in df["month"].values]
            df[f'DTA_MES_{month}'] = temp

        ## Black friday (dia da black friday, não é o priodo)
        def check_blackfriday(row):
            if row['DTA_MES_NOVEMBRO'] == 1 and row['DTA_DIA_SEXTA'] and row.name.day > 24:
                return 1
            else:
                return 0
        df['DTA_FER_BLACKFRIDAY'] = df.apply(check_blackfriday, axis = 1)

        ## Adicionando 1 em caso de feriado
        if holidays is not None:
            temp = [1 if idx in holidays.index else 0 for idx in df.index]
            df['DTA_FER_FERIADO'] = temp

        # Deletando colunas não utilizadas
        df.drop(['weekday','month'], axis=1, inplace=True)

        return df


