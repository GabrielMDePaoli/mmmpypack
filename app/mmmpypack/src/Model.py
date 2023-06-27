import pandas as pd
import numpy as np
from datetime import timedelta
# Transformadores
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import yeojohnson
# Métricas
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.inspection import permutation_importance
# Modelos
from sklearn.linear_model import BayesianRidge

# O que fazer com avisos?
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
pd.options.mode.chained_assignment = None


class Model:
    '''
    ### Definição

    A classe Model é a classe responsável por conter os dados sobre o modelo que está dentro dela.
    '''
    # Quando eu já tenho o modelo e desejo importá-lo
    def __init__(self,
                model:BayesianRidge,
                version:str,
                product:str,
                group_info:pd.DataFrame=None,
                inv_outliers_config:pd.DataFrame=None,
                MMS:MinMaxScaler=None,
                whopper_config:dict=None,
                whopper_list:dict=None,
                johnofig:dict=None,
                monit_outliers:list=None,
                inv_df_name:str='',
                train_method_description:str='',
                holidays:pd.DataFrame=None) -> None:
        '''
        ### Definição

        Método para importar um modelo >=v2.2 já existente para a classe. Ao utilizar esse método a classe ficara travada e não passara mais pelo processo de dataprep.

        Este método vai adaptar os transformadores com o objetivo de otimizar os processos de transformação, diminuindo o tempo de execução para realziar predições.

        ### Parametros

        •	model (obrigatório | modelo sklearn treinado) - Qualquer modelo sklearn treinado.

        •	version (str | obrigatório) - String que define a versão do modelo que esta sendo salvo.

        •	product (str | obrigatório) - String que contenha o nome do produto do modelo que será treinado. Essencial para todo o processo de transformação e saída do resultado. O nome do produto deve seguir a regra das nomenclatura: {frente}_{produto} (ex. NET_BAL).

        •	group_info (pd.DataFrame | obrigatório) - Dataframe que contem as informações dos agrupamentos e fúnis dos veículo.

        •	inv_outliers_config (obrigatório | pd.DataFrame) - Base contendo todas as definições máximas de investimentos.

        •	MMS (obrigatório | MinMaxScaler) - Objeto MinMaxScaler configurado.

        •	whopper_config (obrigatório | dict) - Dicionário contendo as chaves 'Fundo', 'Meio', 'Topo', 'Não classificado' e 'group_method', as 3 primeiras possuem 3 chaves cada 'LAG', 'MMY', 'ADSTOCKN' que configuram a intensidades desses transformadores para cada um desses níveis de fúnil (caso seja < 0, a intensidade será dinâmica). A última das 4 primeiras chaves define qual é o agrupamento a ser realizado (None para não realizar agrupamento).

        •	whopper_list (obrigatório | dict) - Dicionário contendo, para cada coluna, as chaves 'LAG', 'MMY', 'ADSTOCKN' que configuram a intensidades desses transformadores.

        •	johnofig (obrigatório | dict) - Dicionário contendo, para cada coluna, as chaves 'LAG', 'MMY', 'ADSTOCKN' que configuram a transformação Johnsons.

        •	monit_outliers (obrigatório | list) - Lista que define os outliers de vendas para quando o modelo for passado pelo processo de monitoramento.

        •	inv_df_name (obrigatório | dict) - String contendo o nome da base de investimentos que foi utilizada, é uma informação importante para facilitar o controle da base, facilitando a busca por algum erro ou motivo de diferença.

        •	train_method_description (obrigatório | dict) - Descrição detalhando o método de treinamento utilizado, verificar o dicionario de abordagens para entender melhor qual é a descrição correta de se definir aqui.
        
        •	holidays (pd.DataFrame | obrigatório) - Dataframe que contem diversas datas de feriados para anos anteriores e futuros.
        '''
        # Informações sobre o modelo
        self.model = model
        self.root_features = ['_'.join(feature.split('_')[:6]) for feature in model.feature_names_in_]
        self.version = version
        self.product = product
        # Informações sobre o treino
        self.inv_df_name = inv_df_name
        self.train_method_description = train_method_description
        # Salvando transformadores ou configurações
        try:
            if isinstance(group_info, pd.DataFrame):
                self.group_info = group_info
            elif isinstance(self.group_info, None) and isinstance(group_info, None):
                raise ValueError('group_info precisa ser definido')
        except:
            raise ValueError('group_info precisa ser definido')
        try:
            if isinstance(inv_outliers_config, pd.DataFrame):
                self.inv_outliers_config = inv_outliers_config
            elif isinstance(self.inv_outliers_config, None) and isinstance(inv_outliers_config, None):
                raise ValueError('inv_outliers_config precisa ser definido')
        except:
            raise ValueError('inv_outliers_config precisa ser definido')
        try:
            if isinstance(MMS, MinMaxScaler):
                self.MMS = MMS
            elif isinstance(self.MMS, None) and isinstance(MMS, None):
                raise ValueError('MMS precisa ser definido')
        except:
            raise ValueError('MMS precisa ser definido')
        try:
            if isinstance(whopper_config, dict):
                self.whopper_config = whopper_config
            elif isinstance(self.whopper_config, None) and isinstance(whopper_config, None):
                raise ValueError('whopper_config precisa ser definido')
        except:
            raise ValueError('whopper_config precisa ser definido')
        try:
            if isinstance(whopper_list, dict):
                self.whopper_list = whopper_list
            elif isinstance(self.whopper_list, None) and isinstance(whopper_list, None):
                raise ValueError('whopper_list precisa ser definido')
        except:
            raise ValueError('whopper_list precisa ser definido')
        try:
            if isinstance(johnofig, dict):
                self.johnofig = johnofig
            elif isinstance(self.johnofig, None) and isinstance(johnofig, None):
                raise ValueError('whopper_list precisa ser definido')
        except:
            raise ValueError('johnofig precisa ser definido')
        try:
            if isinstance(monit_outliers, list):
                self.monit_outliers = monit_outliers
            elif isinstance(self.monit_outliers, None) and isinstance(monit_outliers, None):
                raise ValueError('whopper_list precisa ser definido')
        except:
            raise ValueError('monit_outliers precisa ser definido')
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
        # Otimizar transformadores e configurações
        self.model_import_optmize_configs()
        return

    def model_import_optmize_configs(self):
        '''
        ### Descrição

        Método que realiza a otimização de transformadores com o objetivo de torná-los mais rápidos para o modelo importado
        '''
        # Definir vari;aveis de uso local
        model = self.model
        MMS = self.MMS
        inv_outliers_config = self.inv_outliers_config.copy()
        whopper_config = self.whopper_config.copy()
        group_info = self.group_info.copy()

        # Salvar todas as features que serão utilizadas (sem agrupamento e com agrupamento separadamente)
        all_features = model.feature_names_in_
        ## Variável com o de para de cada variável e como ela deverá ser transformada
        feature_dict = dict()
        ## Definindo quais features precisam de quais transformações
        for feature in all_features:
            original = '_'.join(feature.split('_')[:6])
            transformation = '_'.join(feature.split('_')[6:])
            feature_dict[original] = transformation

        # MinMaxScaler
        ## Removendo features que não serão utilizadas
        temp_mms_config = {'data_min_': [], 'data_max_': [], 'feature_names_in_': [], 'scale_': [], 'min_': [], 'data_range_': [], }
        for idx, feature_MMS in enumerate(MMS.feature_names_in_):
            if feature_MMS in feature_dict:
                temp_mms_config['data_min_'].append(MMS.data_min_[idx])
                temp_mms_config['data_max_'].append(MMS.data_max_[idx])
                temp_mms_config['feature_names_in_'].append(MMS.feature_names_in_[idx])
                temp_mms_config['scale_'].append(MMS.scale_[idx])
                temp_mms_config['min_'].append(MMS.min_[idx])
                temp_mms_config['data_range_'].append(MMS.data_range_[idx])
        MMS.data_min_ = temp_mms_config['data_min_']
        MMS.data_max_ = temp_mms_config['data_max_']
        MMS.feature_names_in_ = temp_mms_config['feature_names_in_']
        MMS.scale_ = temp_mms_config['scale_']
        MMS.min_ = temp_mms_config['min_']
        MMS.data_range_ = temp_mms_config['data_range_']
        MMS.n_features_in_ = len(MMS.feature_names_in_)

        # Tabela de limite de investimentos
        ## Removendo linhas que não são necessárias
        to_remove = []
        for idx, feature in enumerate(inv_outliers_config['COL'].to_list()):
            if feature not in feature_dict:
                to_remove.append(idx)
        inv_outliers_config.drop(index=to_remove, inplace=True)

        # Tabela de agrupamentos
        ## Removendo linhas que não são necessárias
        group_info.filter(regex=f'SIGLA|MIDIA|VEICULO|CLASSIFICACAO|FUNIL|SIGLA VEICULO|{whopper_config["group_method"]}')
        all_media_vehicles = []
        for feature in feature_dict:
            feature = feature.split('_')
            try: all_media_vehicles.append(f'{feature[0]}_{feature[3]}')
            except: pass
        to_remove = []
        for idx, row in group_info.iterrows():
            if f'{row["MIDIA"]}_{row["SIGLA"]}' not in all_media_vehicles and \
                f'{row["MIDIA"]}_{row["SIGLA VEICULO"]}' not in all_media_vehicles:
                to_remove.append(idx)
        group_info.drop(index=to_remove, inplace=True)

        # Salvar atributos da classe
        self.MMS = MMS
        self.inv_outliers_config = inv_outliers_config.copy()
        self.group_info = group_info.copy()
        self.feature_dict = feature_dict.copy()
        return

    # Processos de predição
    def modelpred_group_vars(self, X:pd.DataFrame) -> pd.DataFrame:
        '''
        ### Definição

        Agrupa as variáveis conforme a base de agrupamento e o tipo de agrupamento.

        ### Parametros

        •	X (obrigatório | pd.DataFrame) - Dataframe contendo os investimentos necessários para a predição.
        '''
        # Definindo variáveis internas
        group_info = self.group_info.copy()
        group_method = self.whopper_config['group_method']
        to_group_list = list(group_info['CLASSIFICACAO'].value_counts().index)

        for focus in to_group_list:
            # Combinar o que for do grupo e se o método combina aquela variável
            class_group_info = group_info[(group_info['CLASSIFICACAO'] == focus) & group_info[group_method] == True]
            for whopper_type in class_group_info['FUNIL'].unique():
                whopper_group_info = class_group_info[class_group_info['FUNIL'] == whopper_type]
                # Nome do veículo desagrupado
                vehicles = [f'{row["MIDIA"]}\\w+{row["SIGLA"]}' for idx, row in whopper_group_info.iterrows()]
                # Nome do veículo agrupado
                new_group_var = whopper_group_info['SIGLA VEICULO'].values[0]
                # Aplicando agrupamentos
                X = self.mousse_group_vars_v3(X.copy(), vehicles, [new_group_var, focus], False, False)

        return X

    def modelpred_limit_inv_outliers(self, X:pd.DataFrame) -> pd.DataFrame:
        '''
        ### Definição

        Limita os outliers de investimento.

        ### Parametros

        •	X (obrigatório | pd.DataFrame) - Dataframe contendo os investimentos necessários para a predição.
        '''
        # Variável de uso local
        inv_outliers_config = self.inv_outliers_config

        # Limitando investimentos
        for _, row in inv_outliers_config.iterrows():
            upper_limit = row['UPPER']
            col = row['COL']
            X[col][X[col] > upper_limit] = upper_limit

        return X

    def modelpred_transform_inv(self, X:pd.DataFrame) -> pd.DataFrame:
        '''
        ### Definição

        Controla as transformações dos investimentos após já terem passado pelo processo de MinMaxScaler.

        ### Parametros

        •	X (obrigatório | pd.DataFrame) - Dataframe contendo os investimentos necessários para a predição.
        '''
        # Variáveis de uso local
        feature_dict = self.feature_dict.copy()
        whopper_list = self.whopper_list

        with np.errstate(divide = 'ignore'):
            for feature in feature_dict:
                if feature.startswith('DTA'): continue
                # Sigla das transformações
                transformation = feature_dict[feature]
                # Lag
                lag_ammount = whopper_list[feature]['LAG']
                # Diluição
                dilution_type = transformation.split('_')[1]
                try: dilution_intensity = whopper_list[feature][dilution_type]
                except: dilution_intensity = -3 # Sem intensidade é original
                # Transformação
                transformation_calc = transformation.split('_')[2]
                # Lista com valores de investimento
                inv_list = X[feature].to_list()
                # Investimentos transformados
                transformed_inv = self.modelpred_apply_transformation(
                    inv_list,
                    feature,
                    lag_ammount,
                    dilution_type, dilution_intensity,
                    transformation_calc
                )
                # Salvando a lista de investimentos transformados
                X[f'{feature}_{transformation}'] = transformed_inv

        # Removendo valores negativos (não é possível gerar vendas negativas ou investimentos negativos), nem NaNs pois eles são 0
        X[(X < 0) | (np.isinf(X)) | (np.isnan(X))] = 0
        return X

    def modelpred_apply_transformation(self,
                                       inv_list:list,
                                       col_name:str,
                                       lag_ammount:int,
                                       dilution_type:str,
                                       dilution_intensity:int|float,
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
        if dilution_type == 'MMY':
            mmY, vl_mmY = [], []
            for value in inv_list:
                vl_mmY = [value] + vl_mmY[:dilution_intensity - 1]
                mmY.append(sum(vl_mmY) / len(vl_mmY))
            inv_list = mmY.copy()
        elif dilution_type == 'ADSTOCKN':
            inv_list = self.adstocking(1/dilution_intensity, media_inv=inv_list)

        # Aplicando lag
        if lag_ammount > 0:
            lag = [0] * lag_ammount
            inv_list = lag + inv_list[:-lag_ammount]

        # Aplicando transformações
        if transformation_calc == 'LOG1':
            inv_list = np.log10(inv_list)
        elif transformation_calc == 'LOG3':
            inv_list = np.log10(inv_list) ** 3
        elif transformation_calc == 'YJ':
            inv_list = yeojohnson(inv_list, self.johnofig[col_name][dilution_type])

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

        # Agrupamento das variáveis
        X = self.modelpred_group_vars(X.copy())

        # Remover variáveis que o modelo não utiliza
        try:
            X = X[root_features]
        except:
            X = self.mousse_sazonal_v3(X.copy())
            X = X[root_features]

        # Removendo outliers de investimento
        X = self.modelpred_limit_inv_outliers(X.copy())

        # MinMaxScaler pré coinfigurado
        inv_X = X.filter(regex='FIN_INV$')
        X[inv_X.columns] = self.MMS.transform(X[inv_X.columns])

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
        
        # Gerar X para o modelo
        X = self.modelpred_prepare_X(X.copy())

        # Realizar predição
        y_pred = model.predict(X[model.feature_names_in_])

        return y_pred

    # Processos de monitoramento
    def monitor_get_model_metrics(self,
                                  X:pd.DataFrame,
                                  y:pd.Series,
                                  period:list[str],
                                  remove_out:bool=True,) -> dict:
        '''
        ### Definição

        Gera um relatório de performance do modelo, num determinado periodo e base que for enviada.

        ### Parametros

        •	X (obrigatório | pd.DataFrame) - Dataframe contendo os investimentos necessários para a predição.

        •	y (obrigatório | pd.Series) - Série com o resultado real de vendas que o modelo deve atingir.

        •	period (obrigatório | list) - Lista com a data incial e final da analise.

        •	remove_out (obrigatório | bool) - Se deve remover os dias considerados outliers ou mantê-los no monitoramento.
        '''
        # Realizar predição
        y_pred = self.modelpred_pipeline(X)

        # Cortando os resultados de acordo com period
        pred_df = pd.Series(data=y_pred, index=y.index.copy(), name=f'PRED_{self.product}')
        pred = pred_df.loc[period[0]:period[1]]
        real = y.loc[period[0]:period[1]]

        # Removendo outliers
        if remove_out:
            # Condição de remoção
            row_conditions = (real >= 0) & (real >= self.monit_outliers[0]) & (real <= self.monit_outliers[1])
            # Salvando dias que serão removidos
            removed_days = real[row_conditions == False]
            # Remoção dos outliers na base real
            real = real[row_conditions]
            # Removendo dias em que as vendas preditas foram outliers nas vendas reais
            pred = pred[row_conditions]
        elif not remove_out:
            removed_days = pd.Series()

        # Salvando resultados
        monit_dict = {
            'y_pred': pred.loc[period[0]:period[1]],
            'y_real': real.loc[period[0]:period[1]],
            'removed_days': removed_days,
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
        intercept_df = pd.DataFrame(data={'FEATURE':['INTERCEPT'],'COEF':[self.model.intercept_]})
        var_coef_df = pd.DataFrame({'FEATURE':self.model.feature_names_in_,'COEF':self.model.coef_})
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

        # Preparar X dos modelos
        X = self.modelpred_prepare_X(X.copy())

        # Coletando apenas o periodo desejado
        X = X.loc[period[0]:period[1]]
        y = y.loc[period[0]:period[1]]

        # Calcular importancia
        result = permutation_importance(model, X[model.feature_names_in_], y, n_repeats=33, random_state=333)
        df = pd.DataFrame({'Features':model.feature_names_in_, 'Importancia':result['importances_mean']})

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
        X = self.modelpred_group_vars(X.copy())
        total_X_inv = X.sum().sum()
        model_X_inv = X[inv_root_features].sum().sum()

        return model_X_inv / total_X_inv

    # Função de transformações em geral
    def mousse_group_vars_v3(self,
                             df:pd.DataFrame,
                             vars_to_group:list,
                             new_var_name:list,
                             verbose:bool=False,
                             keep:bool=True) -> pd.DataFrame:
        '''
        ### Descrição

        Método utilizada para agrupar variáveis de acordo com a lista passada e retornar um dataframe com essas variáveis agrupadas realizando a soma de seus valores a cada dia.

        #### Parâmetros

        •	df (obrigatório | pandas DataFrame): DataFrame completo de investimentos.
        
        •	vars_to_group (obrigatório | list): Lista das siglas das variáveis que se deseja agrupar.
        
        •	new_var_name (obrigatório | list): Lista em que o primeiro valor é a sigla da nova variável e o segundo valor é o nome completo da variável.
        
        •	verbose (obrigatório | bool): Se as informações sobre o processo de agrupamento deverão ser exibidas.
        
        •	keep (obrigatório | bool): Se o novo DataFrame deverá manter as variáveis agrupadas.
        '''

        vehicles = vars_to_group
        all_vehicles_regex = "|".join(vehicles)
        new_group_var = new_var_name[0]

        temp_df = df.copy()
        temp_df = temp_df.filter(regex=f'{all_vehicles_regex}')

        if verbose:
            print(f'{temp_df.shape[1]} colunas são dessa caregoria')
        affected_var = {'vehicle': [],'var_name': [],'var_filter': [],'new_var_name': [],'midias': [],'products': []}
        for col in temp_df.columns:
            m, f, p, v, _, _ = col.split('_')
            affected_var['vehicle'].append(v)
            affected_var['var_name'].append(f"{m}_{f}_{p}_{v}_FIN_INV")
            affected_var['var_filter'].append(f"{m}_{f}_{p}_\\w{'{3}'}_FIN_INV")
            affected_var['new_var_name'].append(f"{m}_{f}_{p}_{new_group_var}_FIN_INV")
            affected_var['midias'].append(m)
            affected_var['products'].append(p)
        affected_var = pd.DataFrame(affected_var)

        if verbose:
            print('Contagem de mídias dentro da classificação')
            for row in affected_var[['midias']].value_counts().items():
                print(' -', row[0][0], row[1])
            print('Contagem agrupamentos realizados')

        for row in affected_var[['var_filter','new_var_name']].value_counts().items():
            if verbose:
                print(' -', row[0][1], row[1])
            temp_df[row[0][1]] = temp_df.filter(regex=f'{row[0][0]}').sum(1)

        if keep:
            df[temp_df.columns] = temp_df
        elif not keep:
            temp_df = temp_df.filter(regex=f'{new_group_var}')
            df[temp_df.columns] = temp_df
            df.drop(affected_var['var_name'].to_list(), axis=1, inplace=True)
        return df

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


