import pandas as pd
import numpy as np
# Transformadores
from feature_engine.selection import DropConstantFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import yeojohnson
import statsmodels.api as sm
# Modelos
from sklearn.linear_model import BayesianRidge
# Carregadores de objetos e dados
from pickle import  load as pkl_load, dump as pkl_dump
from json import    load as json_load, dump as json_dump
from joblib import  load as joblib_load, dump as joblib_dump

# O que fazer com avisos?
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
pd.options.mode.chained_assignment = None

class Transform4Test:
    '''
    ### Definição

    A classe Model é a classe responsável por conter os dados sobre o modelo que está dentro dela, sendo eles, os investimentos de treino, transformadores, configurações, entre outros arquivos necessários para se executar o modelo corretamente.
    '''
    # Processos de transformação de dados para treino do modelo
    def __init__(self,
                 inv_df:pd.DataFrame,
                 inv_df_name:str,
                 product:str,
                 valid_period:list,
                 period_splitter:dict,
                 remove_period:list,
                 whopper_config:dict,
                 group_info:pd.DataFrame,
                 holidays:pd.DataFrame,
                 max_intensity:dict=None,
                 sales_limit:list=[None,None]) -> None:
        '''
        ### Definição

        A classe Model é a classe responsável por conter os dados sobre o modelo que está dentro dela, sendo eles, os investimentos de treino, transformadores, configurações, entre outros arquivos necessários para se executar o modelo corretamente.

        ### Parâmetros:

        •	inv_df (pd.DataFrame | obrigatório) - O dataframe enviado deve conter todos os dados de investimento e vendas dentro do alcance definido no parametro valid_period.

        •	inv_df_name (str | obrigatório) - String contendo o nome da base de investimentos que foi utilizada, é uma informação importante para facilitar o controle da base, facilitando a busca por algum erro ou motivo de diferença.

        •	product (str | obrigatório) - String que contenha o nome do produto do modelo que será treinado. Essencial para todo o processo de transformação e saída do resultado. O nome do produto deve seguir a regra das nomenclatura: {frente}_{produto} (ex. NET_BAL).

        •	valid_period (list | obrigatório) - Lista com 2 datas (ano-mes-dia) em forma de string. Define qual será o periodo válido para uso dos dados no processo de treino do modelo.

        •	period_splitter (dict | obrigatório) - Dicionário controlador da forma na qual os dados serão separados para treino, validação e teste. A sequência de dados é a seguinte: inicio do periodo (str data), fim do periodo (str data), realizar shuffle (bool) [Se sim -> coletar X% da base para treino, nome da base secundária (ex.: 'val'), seed]. As chaves compativeis são: 'train', 'val', 'test', sendo que em caso positivo de shuffle, 'val' vem dentro do 'train'.

        •	remove_period (list | obrigatório) - Lista contendo N periodos que se deseja remover.

        •	whopper_config (dict | obrigatório) - Dicionário contendo as chaves 'Fundo', 'Meio', 'Topo', 'Não classificado' e 'group_method', as 3 primeiras possuem 3 chaves cada 'LAG', 'MMY', 'ADSTOCKN' que configuram a intensidades desses transformadores para cada um desses níveis de fúnil (caso seja < 0, a intensidade será dinâmica). A última das 4 primeiras chaves define qual é o agrupamento a ser realizado (None para não realizar agrupamento).

        •	group_info (pd.DataFrame | obrigatório) - Dataframe que contem as informações dos agrupamentos e fúnis dos veículo.

        •	max_intensity (dict | obrigatório) - Dicionário contendo as chaves 'LAG', 'MMY' e 'ADSTOCKN', todas definem o máximo de intensidade que uma transformação pode receber quando não for pré definida no whopper_config.

        •	holidays (pd.DataFrame | obrigatório) - Dataframe que contem diversas datas de feriados para anos anteriores e futuros.

        •	sales_limit (list | obrigatório) - Lista com o valor mínimo e máximo de quantidade de vendas para o produto. Definir como None para o caso de a defenição ser feita com base nos outliers por q1 e q3. 

        '''
        try:
            if not isinstance(self.model, None):
                raise Exception('Não é possível utilizar este objeto, pois ele já foi configurado com um modelo! Crie um objeto do zero para que essa funcionalidade funcione.')
        except: pass

        self.inv_df = inv_df.copy()
        self.inv_df_name = inv_df_name
        self.product = product
        # Definição da colunas de vendas na base
        self.prod_col = f'DEP_{product}_RCP' if product != 'CLR_PRE' else f'DEP_{product}'
        self.valid_period = valid_period.copy()
        self.period_splitter = period_splitter.copy()
        self.remove_period = remove_period.copy()
        
        if isinstance(whopper_config['group_method'], str):
            self.combine_var = True
        elif isinstance(whopper_config['group_method'], None):
            self.combine_var = False
        self.whopper_config = whopper_config.copy()
        self.group_info = group_info.copy()
        if type(max_intensity) == type(None):
            self.max_intensity = {
                'LAG': 15,
                'MMY': 12,
                'ADSTOCKN': 3.0
            }
        elif type(max_intensity) == type(dict()):
            self.max_intensity = max_intensity
        # Base de feriados
        if isinstance(holidays, pd.DataFrame):
            if holidays.index.names != ['date']:
                # Salvar nova base de feriados
                holidays["date"] = pd.to_datetime(holidays["date"])
                holidays.set_index("date", inplace=True)
            self.holidays = holidays.copy()
        # Limites de venda
        self.sales_limit = sales_limit
        # Executar pipeline de geração dados de modelagem
        # e criação dos transformadores
        self.dataprep_pipeline_model_train()
        return

    def dataprep_df_import(self) -> None:
        '''
        ### Definição

        Define o index da base caso já não esteja definido, remove periodos inválidos da base e remove o veículo Trade caso ainda exista.
        '''
        inv_df = self.inv_df.copy()
        try:
            inv_df["DATA_DIA"] = pd.to_datetime(inv_df["DATA_DIA"])
            inv_df.set_index("DATA_DIA", inplace=True)
        except:
            pass
        # Mantendo apenas periodo válido
        inv_df = inv_df.loc[self.valid_period[0]:self.valid_period[1]]
        # Removendo Trade
        inv_df = inv_df.filter(regex='^((?!TRD).)*$')
        # Salvando base de investimentos modificada
        self.inv_df = inv_df.copy()
        return

    def dataprep_splitter(self) -> None:
        '''
        ### Definição

        Separa os dados de acordo com os periodos definidos no period_splitter.
        '''
        for key, period in self.period_splitter.items():
            temp_df:pd.DataFrame = self.inv_df.loc[period[0]:period[1]]
            if period[2]:
                if key == 'train':
                    train_df, val_df = train_test_split(temp_df, train_size=period[3], random_state=self.period_splitter['train'][5])
            else:
                if key == 'train':
                    train_df = temp_df.copy()
                elif key == 'val':
                    val_df = temp_df.copy()
                elif key == 'test':
                    test_df = temp_df.copy()

        # Salvando atributos da classe
        self.train_df = {
            'inv': train_df.filter(regex=f'FIN_INV$'),
            'sales': train_df.filter(regex=f'{self.prod_col}'),
        }
        self.val_df = {
            'inv': val_df.filter(regex=f'FIN_INV$'),
            'sales': val_df.filter(regex=f'{self.prod_col}'),
        }
        self.test_df = {
            'inv': test_df.filter(regex=f'FIN_INV$'),
            'sales': test_df.filter(regex=f'{self.prod_col}'),
        }
        return
    
    def dataprep_group_vars(self) -> None:
        '''
        ### Definição

        Agrupa as variáveis conforme a base de agrupamento e o tipo de agrupamento. Além de configurar os níveis de transformação definidas no whopper_config.
        '''
        # Definindo variáveis internas
        group_info = self.group_info.copy()
        combine_var = self.combine_var
        whopper_config = self.whopper_config.copy()
        group_method = whopper_config['group_method']
        to_group_list = list(group_info['CLASSIFICACAO'].value_counts().index)
        whopper_list = {}
        train_inv = self.train_df['inv'].copy()
        val_inv = self.val_df['inv'].copy()
        test_inv = self.test_df['inv'].copy()

        if combine_var == False:
            group_info[group_method] = False

        for focus in to_group_list:
            class_group_info = group_info[(group_info['CLASSIFICACAO'] == focus) & group_info[group_method] == True]
            for whopper_type in class_group_info['FUNIL'].unique():
                whopper_group_info = class_group_info[class_group_info['FUNIL'] == whopper_type]
                vehicles = [f'{row["MIDIA"]}\\w+{row["SIGLA"]}' for idx, row in whopper_group_info.iterrows()]
                all_vehicles_regex = "|".join(vehicles)
                new_group_var = whopper_group_info['SIGLA VEICULO'].values[0]
                # Salvar tipo de funil para o agrupamento
                for midia in whopper_group_info['MIDIA'].unique():
                    whopper_list[midia + '_' + new_group_var] = {
                        'LAG': whopper_config[whopper_type]['LAG'],
                        'MMY': whopper_config[whopper_type]['MMY'],
                        'ADSTOCKN': whopper_config[whopper_type]['ADSTOCKN'],
                    }

                train_inv = self.mousse_group_vars_v3(train_inv.copy(), vehicles, [new_group_var, focus], False, False)
                val_inv = self.mousse_group_vars_v3(val_inv.copy(), vehicles, [new_group_var, focus], False, False)
                test_inv = self.mousse_group_vars_v3(test_inv.copy(), vehicles, [new_group_var, focus], False, False)

        for idx, row in group_info[group_info[group_method] == False].iterrows():
            whopper_list[row['MIDIA'] + '_' + row['SIGLA']] = {
                'LAG': whopper_config[row['FUNIL']]['LAG'],
                'MMY': whopper_config[row['FUNIL']]['MMY'],
                'ADSTOCKN': whopper_config[row['FUNIL']]['ADSTOCKN'],
            }

        # Salvando atributos da classe
        self.train_df['inv'] = train_inv.copy()
        self.val_df['inv'] = val_inv.copy()
        self.test_df['inv'] = test_inv.copy()
        self.whopper_list = whopper_list
        return

    def dataprep_remove_constant(self) -> None:
        '''
        ### Definição

        Remove features que são constantes demais (>= 90%).
        '''
        # Definindo variáveis locais
        train_inv = self.train_df['inv'].copy()
        val_inv = self.test_df['inv'].copy()
        test_inv = self.val_df['inv'].copy()

        # Variáveis quase constantes (90%)
        min_var = 0.9
        DCF = DropConstantFeatures(tol=min_var)
        # Verificando apenas periodo de treino e teste
        train_inv = DCF.fit_transform(train_inv)
        # Mantendo colunas
        val_inv = val_inv[train_inv.columns]
        test_inv = test_inv[train_inv.columns]

        # Salvando atributos da classe
        self.train_df['inv'] = train_inv.copy()
        self.test_df['inv'] = val_inv.copy()
        self.val_df['inv'] = test_inv.copy()
        self.DCF = DCF
        return

    def dataprep_find_lags(self) -> None:
        '''
        ### Definição

        Alterando lag para o que possui maior correlação com o periodo de Treino

        Apenas os lags menores que 0 serão afetados e terão os tetos de cada intensidade de acordo com o valor definido nos parametros do find_my_lag
        '''
        # Salvando atributos da classe
        self.whopper_list = self.find_my_lag()
        return

    def dataprep_limit_inv_outliers(self) -> None:
        '''
        ### Definição

        Limita os investimentos de cada variável
        '''
        # Definindo variáveis internas
        train_inv = self.train_df['inv'].copy()
        val_inv = self.val_df['inv'].copy()
        test_inv = self.test_df['inv'].copy()

        inv_outliers_config_dict = {'COL': [], 'UPPER':[]}
        for col in train_inv.columns:
            col_data = train_inv[col].to_list()
            not0_col_data = [value for value in col_data if value > 0]
            q1 = np.quantile(not0_col_data, 0.25)
            q3 = np.quantile(not0_col_data, 0.75)
            iqr = q3 - q1
            upper_limit = q3 + 1.5 * iqr
            # Salvando limites para salvar em um arquivo
            inv_outliers_config_dict['COL'].append(col)
            inv_outliers_config_dict['UPPER'].append(upper_limit)
            train_inv[col][train_inv[col] > upper_limit] = upper_limit
            val_inv[col][val_inv[col] > upper_limit] = upper_limit
            test_inv[col][test_inv[col] > upper_limit] = upper_limit

        # Salvando atributos da classe
        self.inv_outliers_config = pd.DataFrame(inv_outliers_config_dict)
        self.train_df['inv'] = train_inv.copy()
        self.val_df['inv'] = val_inv.copy()
        self.test_df['inv'] = test_inv.copy()
        return

    def dataprep_minmaxscaler(self) -> None:
        '''
        ### Definição

        Transforma toda a coluna para um alcance de 0 a 100 e salva o objeto MMS ao final.
        '''
        # Definindo variáveis internas
        train_inv = self.train_df['inv'].copy()
        val_inv = self.val_df['inv'].copy()
        test_inv = self.test_df['inv'].copy()

        # Aplicando MinMaxScaler
        MMS = MinMaxScaler((0,100))
        # Configurando MMS apenas para base de treino
        MMS.fit(train_inv[train_inv.columns])
        # Aplicando MMS
        train_inv[train_inv.columns] = MMS.transform(train_inv[train_inv.columns])
        val_inv[val_inv.columns] = MMS.transform(val_inv[val_inv.columns])
        test_inv[test_inv.columns] = MMS.transform(test_inv[test_inv.columns])

        # Salvando atributos da classe
        self.train_df['inv'] = train_inv.copy()
        self.val_df['inv'] = val_inv.copy()
        self.test_df['inv'] = test_inv.copy()
        self.MMS = MMS

        return

    def dataprep_calc_tra_saz(self) -> None:
        '''
        ### Definição

        Aplicação de transformações e sazonalidade na base de treino, validação e teste.
        '''
        # Definindo variáveis internas
        train_df = self.train_df.copy()
        val_df = self.val_df.copy()
        test_df = self.test_df.copy()
        period_splitter = self.period_splitter.copy()

        # Combinando datasets
        train_df = pd.concat([train_df['sales'], train_df['inv']], axis=1)
        val_df = pd.concat([val_df['sales'], val_df['inv']], axis=1)
        test_df = pd.concat([test_df['sales'], test_df['inv']], axis=1)

        # Juntando toda a base para criar transformações de continuidade corretamente
        df = pd.concat([train_df, test_df, val_df], axis=0)
        df.sort_index(inplace=True)

        # Gerando dataset com transformações
        # df = mousse_transform_df(df.copy())                    # Transformações para modelos v2
        # df = mousse_transform_df_v3(df.copy())                 # Novas transformações para modelos >=v2.1 e v3
        df = self.mousse_who_transform_df_v3(df.copy())  # Novas transformações para modelos considerando funil >=v2.1 e v3

        # Adicionando variáveis de sazonalidade
        df = self.mousse_sazonal_v3(df.copy())

        # Separando dados de treino, teste e validação
        for key, period in period_splitter.items():
            temp_df:pd.DataFrame = df.loc[period[0]:period[1]]
            if period[2]:
                if key == 'train':
                    train_df, test_df = train_test_split(temp_df, train_size=period[3], random_state=period_splitter['train'][5])
            else:
                if key == 'train':
                    train_df = temp_df.copy()
                elif key == 'val':
                    val_df = temp_df.copy()
                elif key == 'test':
                    test_df = temp_df.copy()

        # Salvando atributos da classe
        self.train_df = train_df.copy()
        self.val_df = val_df.copy()
        self.test_df = test_df.copy()

        return

    def dataprep_sales_anomalie_control(self) -> None:
        '''
        ### Definição

        Método que remove os dias de venda altas ou baixas demais. Ou muito diferentes definidos em remove_periods.
        '''
        # Definindo variáveis internas
        train_df = self.train_df.copy()
        val_df = self.val_df.copy()
        test_df = self.test_df.copy()
        prod_col = self.prod_col
        valid_period = self.valid_period
        remove_period = self.remove_period

        # Coletando os limites de acordo com base de treino
        q1 = train_df[prod_col].quantile(0.25)
        q3 = train_df[prod_col].quantile(0.75)
        iqr = q3 - q1

        if self.sales_limit[0] == None:
            lower_limit = q1 - 1.5*iqr # Vendas muito baixas
        elif isinstance(self.sales_limit[0], int):
            lower_limit = self.sales_limit[0] # Vendas muito baixas
        else:
            raise TypeError('lower_limit inválido')

        if self.sales_limit[1] == None:
            upper_limit = q3 + 1.5*iqr # Vendas altas demais
        elif isinstance(self.sales_limit[1], int):
            upper_limit = self.sales_limit[1] # Vendas altas demais
        else:
            raise TypeError('upper_limit inválido')

        # Removendo outliers e 0's
        train_df = train_df[
            (train_df[prod_col] > 0) &
            (train_df[prod_col] > lower_limit) &
            (train_df[prod_col] < upper_limit)]
        val_df = val_df[
            (val_df[prod_col] > 0) &
            (val_df[prod_col] > lower_limit) &
            (val_df[prod_col] < upper_limit)]
        test_df = test_df[
            (test_df[prod_col] > 0) &
            (test_df[prod_col] > lower_limit) &
            (test_df[prod_col] < upper_limit)]
        
        # Removendo pontos e periodos muito discrepantes
        for period in remove_period:
            train_df = train_df[
                (train_df.index >= valid_period[0]) & (train_df.index <= period[0]) |
                (train_df.index >= period[1]) & (train_df.index <= valid_period[1])
            ]
            test_df = test_df[
                (test_df.index >= valid_period[0]) & (test_df.index <= period[0]) |
                (test_df.index >= period[1]) & (test_df.index <= valid_period[1])
            ]
            val_df = val_df[
                (val_df.index >= valid_period[0]) & (val_df.index <= period[0]) |
                (val_df.index >= period[1]) & (val_df.index <= valid_period[1])
            ]

        # Salvando atributos da classe
        self.train_df = train_df.copy()
        self.val_df = val_df.copy()
        self.test_df = test_df.copy()
        self.sales_limit = [lower_limit, upper_limit]
        return

    def dataprep_sales_series_decompose(self) -> None:
        # Decompose Train
        target = self.train_df[self.prod_col].groupby(pd.Grouper(freq='d')).sum()
        dec = sm.tsa.seasonal_decompose(target)
        seasonal = dec.seasonal
        trend = dec.trend
        resid = dec.resid
        seasonal[np.isnan(seasonal)] = 0
        trend[np.isnan(trend)] = 0
        resid[np.isnan(dec.resid)] = 0
        self.train_df[f'{self.prod_col}_SAZ'] = seasonal
        self.train_df[f'{self.prod_col}_INV'] = trend + resid
        # Decompose Val
        target = self.val_df[self.prod_col].groupby(pd.Grouper(freq='d')).sum()
        dec = sm.tsa.seasonal_decompose(target)
        seasonal = dec.seasonal
        trend = dec.trend
        resid = dec.resid
        seasonal[np.isnan(seasonal)] = 0
        trend[np.isnan(trend)] = 0
        resid[np.isnan(dec.resid)] = 0
        self.val_df[f'{self.prod_col}_SAZ'] = seasonal
        self.val_df[f'{self.prod_col}_INV'] = trend + resid
        # Decompose Test
        target = self.test_df[self.prod_col].groupby(pd.Grouper(freq='d')).sum()
        dec = sm.tsa.seasonal_decompose(target)
        seasonal = dec.seasonal
        trend = dec.trend
        resid = dec.resid
        seasonal[np.isnan(seasonal)] = 0
        trend[np.isnan(trend)] = 0
        resid[np.isnan(dec.resid)] = 0
        self.test_df[f'{self.prod_col}_SAZ'] = seasonal
        self.test_df[f'{self.prod_col}_INV'] = trend + resid
        return

    def dataprep_pipeline_model_train(self) -> None:
        '''
        ### DEFINIÇÃO

        Função que realiza todo passo a passo para gerar os dados de treino, validação e teste para o modelo focado no produto que foi definido na inicialização da classe.
        '''
        self.dataprep_df_import()
        self.dataprep_splitter()
        self.dataprep_group_vars()
        self.dataprep_remove_constant()
        self.dataprep_find_lags()
        self.dataprep_limit_inv_outliers()
        self.dataprep_minmaxscaler()
        self.dataprep_calc_tra_saz()
        self.dataprep_sales_anomalie_control()
        self.train_df.index.names = ['DATA_DIA']
        self.test_df.index.names = ['DATA_DIA']
        self.val_df.index.names = ['DATA_DIA']
        self.dataprep_sales_series_decompose()
        return

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

    def find_my_lag(self) -> list:
        '''
        ### DEFINIÇÃO

        Método que encontra o melhor lag e outras variáveis para cada variável.
        '''
        # Definindo variáveis locais
        inv = self.train_df['inv'].copy()
        target = self.train_df['sales'][self.prod_col].copy()
        whopper_list = self.whopper_list
        lag_range = self.max_intensity['LAG']
        mm_range = self.max_intensity['MMY']
        adstock_range = self.max_intensity['ADSTOCKN']

        # Procurando pelas melhores intensidades
        for col in inv.columns:
            inv_series = inv[col].to_list()
            current_vehicle = col.split('_')[0] + '_' + col.split('_')[3] # Veículo para acessar whopper_list
            # SELEÇÃO MELHOR LAG
            # Criando variáveis de controle
            best_lag_corr = -1
            best_lag = -1
            lagged_inv_final = None
            if whopper_list[current_vehicle]['LAG'] < 0:
                # Parar redefinição caso o veículo já possua um lag definido
                # Testando diversos lags
                for lag_ammount in range(lag_range[0], lag_range[1]):
                    lagged_inv = [0]*lag_ammount + inv_series[lag_ammount:]
                    corr = np.corrcoef(lagged_inv, target)[0, 1]
                    if corr > best_lag_corr:
                        best_lag_corr = corr
                        best_lag = lag_ammount
                        lagged_inv_final = lagged_inv.copy()
            # Preparando lag pré definido caso o usuário deseje
            if lagged_inv_final == None:
                lagged_inv_final = [0]*whopper_list[current_vehicle]['LAG'] + inv_series[whopper_list[current_vehicle]['LAG']:]
            # SELEÇÃO MELHOR MÉDIA MÓVEL
            # Criando variáveis de controle
            best_mm_corr = -1
            best_mm = -1
            if whopper_list[current_vehicle]['MMY'] < 0:
                # Parar redefinição caso o veículo já possua um mm definido
                # Testando diversos lags
                for mm_ammount in range(mm_range[0], mm_range[1]):
                    lagged_inv = lagged_inv_final.copy()
                    # Gerando lista de valores ADSTOCKN, MMX, MMY
                    mm_inv, vl_mmY = [], []
                    for value in lagged_inv:
                        vl_mmY = [value] + vl_mmY[:mm_ammount - 1]
                        mm_inv.append(sum(vl_mmY) / len(vl_mmY))
                    del(vl_mmY, lagged_inv)
                    corr = np.corrcoef(mm_inv, target)[0, 1]
                    if corr > best_mm_corr:
                        best_mm_corr = corr
                        best_mm = mm_ammount
            # SELEÇÃO MELHOR ADSTOCK
            # Criando variáveis de controle
            best_adstock_corr = -1
            best_adstock = -1
            if whopper_list[current_vehicle]['ADSTOCKN'] < 0:
                # Parar redefinição caso o veículo já possua um adstock definido
                # Testando diversos lags
                for adstock_ammount in np.arange(adstock_range[0], adstock_range[1], 0.3):
                    lagged_inv = lagged_inv_final.copy()
                    adstock_inv = self.adstocking(1/adstock_ammount, media_inv=lagged_inv)
                    corr = np.corrcoef(adstock_inv, target)[0, 1]
                    if corr > best_adstock_corr:
                        best_adstock_corr = corr
                        best_adstock = adstock_ammount
            # Salvando configuração da variável
            whopper_list[col] = {
                'LAG': best_lag if best_lag != -1 else whopper_list[current_vehicle]['LAG'], # Salvando o melhor lag
                'MMY': best_mm if best_mm != -1 else whopper_list[current_vehicle]['MMY'],
                'ADSTOCKN': best_adstock if best_adstock != -1 else whopper_list[current_vehicle]['ADSTOCKN'],
            }

        # Retorna as melhores intensidades
        return whopper_list

    def mousse_who_transform_df_v3(self,
                                   df:pd.DataFrame) -> pd.DataFrame:
        '''
        ### Descrição

        Função utilizada para transformar todas as colunas do dataset que é enviado a ela.

        As variáveis de investimento (finalizadas com FIN_INV) serão transformadas e salvas em outras colunas (variações da coluna), qualquer outra não receberá variação.

        E nessa variação da função de transformação, apenas o LAG configurado no whopper_list será salvo

        #### Transformações de atraso

        •	LAG0N - N dias de atraso para a variável.

        #### Transformações de continuidade

        •	ORIG - Variaveil sem aplicação de continuidade.

        •	ADSTOCKN - Adstock com resiudos divididos por N partes a cada dia.

        •	MMY - Média móvel de Y dias (considerando 0's).

        #### Transformações de continuidade

        •	LOG1 - Quando se aplica Log aos valores.

        •	LOG3 - Quando se aplica Log elevado ao cubo aos valores.

        •	YJ - Quando se aplica yeojohnson.

        ### Parametros

        •	df (pd.DataFrame | obrigatório) - é o dataset (em um dataframe do pandas) com os dados de investimento.
        '''
        whopper_list = self.whopper_list.copy()
        johnofig = dict() # Configuração do Jhonsons

        with np.errstate(divide = 'ignore'):
            inv_cols = [c for c in df.columns if c.endswith('FIN_INV')]

            for inv in inv_cols:
                johnofig[inv] = {}
                var_name = inv # Nome da variável
                current_vehicle = inv.split('_')[0] + '_' + inv.split('_')[3] # Veículo
                try:
                    # Quando for feito o processo find_my_lag
                    lag_ammount = whopper_list[var_name]['LAG']
                    mmy_ammount = whopper_list[var_name]['MMY']
                    adstockn_ammount = whopper_list[var_name]['ADSTOCKN']
                except:
                    # Com diluições já pré definido
                    lag_ammount = whopper_list[current_vehicle]['LAG']
                    mmy_ammount = whopper_list[current_vehicle]['MMY']
                    adstockn_ammount = whopper_list[current_vehicle]['ADSTOCKN']
                all_transformations = [
                    # Sem efeitos futuros
                    'LAGN_ORIG_LOG1','LAGN_ORIG_LOG3','LAGN_ORIG_YJ',
                    # Adstock
                    'LAGN_ADSTOCKN_LOG1','LAGN_ADSTOCKN_LOG3','LAGN_ADSTOCKN_YJ',
                    # Média móvel de y dias
                    'LAGN_MMY_LOG1','LAGN_MMY_LOG3','LAGN_MMY_YJ',
                ]
                aux_list = {}
                for key in all_transformations:
                    aux_list[key] = []

                col_values = df[inv].to_list()

                # Gerando lista de valores ADSTOCKN, MMX, MMY
                mmY, vl_mmY = [], []
                for value in col_values:
                    vl_mmY = [value] + vl_mmY[:mmy_ammount - 1]
                    mmY.append(sum(vl_mmY) / len(vl_mmY))
                del(vl_mmY)
                # ADSTOCKN
                adstockN = self.adstocking(1/adstockn_ammount, media_inv=col_values)

                if lag_ammount > 0:
                    lag = [0] * lag_ammount
                    # Configurando atraso de N dias
                    col_values = lag + col_values[:-lag_ammount]
                    adstockN = lag + adstockN[:-lag_ammount]
                    mmY = lag + mmY[:-lag_ammount]

                # LOG
                aux_list[f'LAGN_ORIG_LOG1'] = np.log10(col_values)
                # LOG3
                aux_list[f'LAGN_ORIG_LOG3'] = np.log10(col_values)**3
                # YJ
                aux_list[f'LAGN_ORIG_YJ'], johnofig[inv]['ORIG'] = yeojohnson(col_values)

                # ADSTOCK N dias
                # ADSTOCK LOG
                aux_list[f'LAGN_ADSTOCKN_LOG1'] = np.log10(adstockN)
                # ADSTOCK LOG3
                aux_list[f'LAGN_ADSTOCKN_LOG3'] = np.log10(adstockN)**3
                # ADSTOCK YJ
                aux_list[f'LAGN_ADSTOCKN_YJ'], johnofig[inv]['ADSTOCKN'] = yeojohnson(adstockN)

                # Média móvel y dias
                # Média móvel LOG (y dias)
                aux_list[f'LAGN_MMY_LOG1'] = np.log10(mmY)
                # Média móvel LOG3 (y dias)
                aux_list[f'LAGN_MMY_LOG3'] = np.log10(mmY)**3
                # Média móvel YJ (y dias)
                aux_list[f'LAGN_MMY_YJ'], johnofig[inv]['MMY'] = yeojohnson(mmY)

                for key in aux_list.keys():
                    df[f'{inv}_{key}'] = aux_list[key]

            # Removendo valores negativos (não é possível gerar vendas negativas ou investimentos negativos), nem NaNs pois eles são 0
            df[(df < 0) | (np.isinf(df)) | (np.isnan(df))] = 0

        # Salvando configuração do YJ
        self.johnofig = johnofig
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
