from sklearn.feature_selection import RFE
from termcolor import colored
import pandas as pd
import numpy as np
from time import sleep
from random import choice, choices, randint, random, sample
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from joblib import dump as joblib_dump

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn


class MMMFeatureSelection:
    def __init__(self,
                 product:str,
                 train:pd.DataFrame,
                 val:pd.DataFrame,
                 test:pd.DataFrame|None,
                 preselection_config:dict,
                 variations:dict,
                 defined_model:any,
                 verbose:bool=True) -> None:
        '''
        ### Definição

        ### Parametros

        •	xxx (yyy | obrigatório) - abc.
        '''
        self.verbose = verbose
        # Variável que verifica se é necesssário executar a configuração da otimização (GA)
        self.optimization_configured = False
        # Definindo produto e base de investimentos
        self.product = product # Nome do produto que será realizada a predição
        self.product_column = f'DEP_{product}_RCP' if product not in ['CLR_PRE'] else f'DEP_{product}'
        self.train = train.copy() # Base de treino para o modelo aprender os pesos
        try:
            self.train["DATA_DIA"] = pd.to_datetime(self.train["DATA_DIA"])
            self.train.set_index("DATA_DIA", inplace=True)
        except: pass
        self.val = val.copy() # Base de validação para o modelo otimizar a seleção de features
        try:
            self.val["DATA_DIA"] = pd.to_datetime(self.val["DATA_DIA"])
            self.val.set_index("DATA_DIA", inplace=True)
        except: pass
        if isinstance(test, pd.DataFrame):
            self.test = test.copy() # Base de teste para que o modelo consiga ser testado (opcional)
            try:
                self.test["DATA_DIA"] = pd.to_datetime(self.test["DATA_DIA"])
                self.test.set_index("DATA_DIA", inplace=True)
            except: pass
        elif isinstance(test, None):
            self.test = None

        # Definindo configurações de pré seleção
        if 'week_var' in preselection_config:
            self.week_var = preselection_config['week_var']
        else:
            self.week_var = False
        if 'month_var' in preselection_config:
            self.month_var = preselection_config['month_var']
        else:
            self.month_var = False
        if 'holiday_var' in preselection_config:
            self.holiday_var = preselection_config['holiday_var']
        else:
            self.holiday_var = False
        if 'initial_rfe' in preselection_config:
            self.initial_rfe = preselection_config['initial_rfe']
        else:
            self.initial_rfe = 40
        if 'random_count' in preselection_config:
            self.random_count = preselection_config['random_count']
        else:
            self.random_count = 0
        if 'population_count' in preselection_config:
            self.population_count = preselection_config['population_count']
        else:
            self.population_count = 12
        if 'keep' in preselection_config:
            self.keep = preselection_config['keep']
        else:
            self.keep = 3

        # Configurações das variações
        self.variations = variations.copy()
        self.fin_inv_transformations = [k for k in variations if k >= 0]
        self.dta_transformations = [k for k in variations if k < 0]

        # Configurando modelo a ser utilizado
        self.defined_model = defined_model

        return
    
    def save_pod(self,
                 pod_config:dict|None) -> None:
        '''
        ### Definição

        ### Parametros

        •	pod_config (dict, None | opcional) - Parametros de Otimização Dinamicos. Dicinionário que configura os parametros dinâmicos para a otimização.
        '''
        # Verifica se aquele atributo já foi definido
        defined_params = False
        defined_learn_treshold_params = False
        defined_rest_treshold_params = False
        # Caso o POD tenha sido definido
        if pod_config != None:
            if 'params' in pod_config:
                self.params = pod_config['params']
                defined_params = True
            if 'learn_treshold_params' in pod_config:
                self.learn_treshold_params = pod_config['learn_treshold_params']
                defined_learn_treshold_params = True
            if 'rest_treshold_params' in pod_config:
                self.rest_treshold_params = pod_config['rest_treshold_params']
                defined_rest_treshold_params = True
        # Definindo POD não definido pela chamada da função
        else:
            if not defined_params:
                self.params = {
                    # Modificáveis
                    'NUM_REMOVE': 6,
                    'NUM_HIGH_REMOVE': 1,
                    'NUM_ADD': 6,
                    'NEGATIVE_WEIGHT': 0,
                    # Definidos antes de iniciar a seleção
                    'CHANGE_CHANCE': 0.6,
                    'POP_SIZE': self.population_count,
                    'RANDOM_SIZE': self.random_count,
                    'KEEP': self.keep,
                    'TRANSFER_VAR': 90,
                    'MAX_COEF': 1000,
                }
            if not defined_learn_treshold_params:
                self.learn_treshold_params = {
                    'param_config': {
                        'NUM_REMOVE': {
                            'limits': [2, 15],
                            'intervals': 3
                        },
                        'NUM_HIGH_REMOVE': {
                            'limits': [1, 2],
                            'intervals': 1
                        },
                        'NUM_ADD': {
                            'limits': [6, 13],
                            'intervals': 3
                        },
                        'NEGATIVE_WEIGHT': {
                            'limits': [0.008, 0.025],
                            'intervals': 3
                        }
                    },
                    'increasable': self.increasable['learn'],
                    'patience': self.patience,
                    'max_gen_type': 200,
                    'increase': self.increase_params_mult,
                }
            if not defined_rest_treshold_params:
                self.rest_treshold_params = {
                    'param_config': {
                        'NUM_REMOVE': {
                            'limits': [6, 14],
                            'intervals': 3
                        },
                        'NUM_HIGH_REMOVE': {
                            'limits': [1, 2],
                            'intervals': 1
                        },
                        'NUM_ADD': {
                            'limits': [8, 16],
                            'intervals': 3
                        },
                        'NEGATIVE_WEIGHT': {
                            'limits': [0, 1],
                            'intervals': 1
                        }
                    },
                    'increasable': self.increasable['rest'],
                    'patience': self.patience,
                    'max_gen_type': 30,
                    'increase': self.increase_params_mult,
                }
        # Para auxiliar na escolha da melhor transformação inicial
        self.negative_weight = self.params['NEGATIVE_WEIGHT']
        return

    # Processo de otimização
    def config_feature_selection(self,
                                 config:dict,
                                 preselected:list):
        self.patience = {
            'overall': 0,
            'type': 0,
            'params': 0
        }
        self.increasable = {
            'learn': [
                'NEGATIVE_WEIGHT',
            ],
            'rest': [
                'NUM_REMOVE',
                'NUM_ADD',
            ],
        }
        print(config)
        if 'increase_params_mult' in config:
            self.increase_params_mult = config['increase_params_mult']
        else:
            self.increase_params_mult = 0.1
        if 'num_cycles' in config:
            self.num_cycles = config['num_cycles']
        else:
            self.num_cycles = 12
        if 'target_col_suffix' in config:
            self.target_column = f'{self.product_column}{config["target_col_suffix"]}'
        else:
            self.target_column = f'{self.product_column}'
        if 'ignore_neg_coef' in config:
            self.ignore_neg_coef = config['ignore_neg_coef']
        else:
            self.ignore_neg_coef = False
        if 'train_metric' in config:
            self.train_metric = config['train_metric']
        else:
            self.train_metric = {'dif2':0.65,'mape':0.35}
        # Definindo variáveis préselecionadas (opcional)
        self.preselected = preselected
        return

    def set_best_preselected(self) -> None:
        '''
        ### Definição

        Método que realiza a seleção da melhor transformação inicial para a otimização (caso não haja variáveis na lista preselected). Também realiza a conversão da lista de variáveis iniciais para um genoma do algoritmo genético.

        ### Parametros

        '''
        # Variáveis de uso local
        preselected = self.preselected.copy()
        train, val, test = self.train.copy(), self.val.copy(), self.test.copy()
        variations = self.variations.copy()
        product_column = self.target_column
        initial_rfe = self.initial_rfe
        week_var, month_var, holiday_var = self.week_var, self.month_var, self.holiday_var

        # Caso não exista preseleção, escolher a melhor transformação para prever as vendas
        if len(preselected) == 0:
            best_transformation = {
                'transformation': '', 'fit': np.inf, 'mape': np.inf, 'dif1': np.inf, 'dif2': np.inf, 'intercept': 0, 'features': []
            }
            for t_key in variations:
                if t_key <= 0: continue
                # Transformação selecionada como inicial
                train_f = train.filter(regex=f'^DTA|^DEP|{variations[t_key]}$')
                val_f = val.filter(regex=f'^DTA|^DEP|{variations[t_key]}$')
                test_f = test.filter(regex=f'^DTA|^DEP|{variations[t_key]}$')
                # Preparando Regex para o filtro de colunas
                regex_text = 'FIN_INV'
                if week_var:
                    regex_text += '|^DTA_DIA'
                if month_var:
                    regex_text += '|^DTA_MES'
                if holiday_var:
                    regex_text += '|^DTA_FER'
                # Definição de base de features e labels de treino/teste
                X_train, y_train = train_f.filter(regex=f'{regex_text}'), train_f[product_column]
                X_val, y_val = val_f.filter(regex=f'{regex_text}'), val_f[product_column]
                X_test, y_test = test_f.filter(regex=f'{regex_text}'), test_f[product_column]
                # Executando RFE
                rfe_model = RFE(self.defined_model(), step=0.05, n_features_to_select=initial_rfe).fit(X_train, y_train)
                X_train = X_train[X_train.columns[rfe_model.support_]]
                X_val = X_val[X_val.columns[rfe_model.support_]]
                X_test = X_test[X_test.columns[rfe_model.support_]]
                # Treinando o modelo com as variáveis selecionadas
                model = self.defined_model().fit(X_train, y_train)
                used_fetures = list(model.feature_names_in_)
                # Resultado sem otimizar as variáveis selecionadas
                model_stats, val_stats, _ = self.get_model_metrics(X_val, y_val, model, [X_test, y_test])
                if best_transformation['fit'] > val_stats['fit_value_no_penalty']:
                    best_transformation['transformation'] = variations[t_key]
                    best_transformation['fit'] = val_stats['fit_value_no_penalty']
                    best_transformation['mape'] = val_stats['mape']
                    best_transformation['dif1'] = val_stats['dif1']
                    best_transformation['dif2'] = val_stats['dif2']
                    best_transformation['intercept'] = model_stats['intercept']
                    best_transformation['features'] = list(model.feature_names_in_)
            # Preparando colunas para a conversão
            if self.verbose:
                print(colored("ATENÇÃO", 'red'), f'- A transormação {best_transformation["transformation"]} esta sendo utilizada')
            selected_columns = list(best_transformation['features']) # Utiliza a melhor transformação na preseleção
            # selected_columns = [var for var in list(best_transformation['features']) if var.startswith('DTA')] # Linha para remover preseleção de alguma transformação
            X_train, y_train = train[selected_columns], train[product_column]
            X_val, y_val = val[selected_columns], val[product_column]
        elif len(preselected) > 0: # Processo executado caso as features já forem pré selecionadas
            if self.verbose:
                print(colored("ATENÇÃO", 'red'), '- Variáveis pré selecionadas estão sendo utilizadas!')
            X_train, y_train = train[preselected], train[product_column]
            X_val, y_val = val[preselected], val[product_column]
            used_fetures = preselected
        # Salvando lista de variações para ser o ponto de partida do algoritmo genético
        variations_set = list()
        var_dict = dict()
        for list_id, c in enumerate(list(train.filter(regex='^DTA|FIN_INV$').columns)):
            found_features = [idx if c in f else -1 for idx, f in enumerate(used_fetures)]
            if max(found_features) >= 0:
                if 'DTA' not in c:
                    sufix = '_'.join(used_fetures[max(found_features)].split('_')[4:])
                elif 'DTA' in c:
                    sufix = ''
                var = list(variations.keys())[list(variations.values()).index(sufix)]
                variations_set.append(var)
            else:
                if 'DTA' not in c:
                    variations_set.append(0)
                elif 'DTA' in c:
                    variations_set.append(-2)
            var_dict[list_id] = c

        # Salvar variáveis da classe
        self.var_dict = var_dict
        self.variations_set = variations_set
        return

    def reset_best_results(self) -> None:
        '''
        ### Definição

        Reinicia os atributos que armazenam os melhores reusltados da otimização.
        '''
        self.best_model_fit = None # Modelo que respeita as regras com o maior fit
        self.best_model_metrics = [0, 0] # MAPE, DIF
        self.models_hist = [] # Lista de modelos que respeitam as regras
        self.best_fit = np.inf # Melhor fit encontrado entre os modelos
        
        return
    
    def reset_population(self) -> None:
        '''
        ### Definição

        Deleta a população atual para que outra seja criada e utilizada.
        '''
        del(self.population) # Reinicia a população

        return

    def create_treshold(self, param_config:dict, increasable:list, patience:dict, max_gen_type:int, increase:float=0.25):
        '''
        ### Definição

        Método que cria os estágios de cada etapa da otimização. Um dos processos necessários para o POD.

        ### Parametros

        
        '''
        tresholds = {}
        change_points = [] # Lista com todos os pontos (patience params) necessários para realizar alguma troca, utilizado para reduzir o número de chamadas da função de mudança de parametros

        limits_increase = 1 + (increase * patience['overall']) # A cada incremento do PACIENCE_OVERALL os limites aumentarão de x em x porcento
        for param in param_config:
            current_config = param_config[param]
            if param in increasable:
                if param != 'NEGATIVE_WEIGHT':
                    current_config['limits'] = [int(i * limits_increase) for i in current_config['limits']] # Aumentar limite do parametro permitido
                if param == 'NEGATIVE_W EIGHT':
                    current_config['limits'] = [i * limits_increase for i in current_config['limits']] # Aumentar limite do parametro permitido
            
            par_limit = current_config['limits'] # Limites do parametro
            n_intervals = current_config['intervals'] # Total de tresholds
            try:
                n_val = 1 / n_intervals # Gera o multiplicador para gerar o melhor step para gerar n_intervals
            except:
                n_val = 1 / n_intervals # Gera o multiplicador para gerar o melhor step para gerar n_intervals

            if param != 'NEGATIVE_WEIGHT':
                step = int((par_limit[1] - par_limit[0]) * n_val) # Calcula o step necessário para dividir os dados em n_intervals
                steps_param = list(range(par_limit[0], par_limit[1], step if step > 0 else 1)) # Lista com os valores que serão definidos em cada treshold
            elif param == 'NEGATIVE_WEIGHT':
                step = (par_limit[1] - par_limit[0]) * n_val # Calcula o step necessário para dividir os dados em n_intervals
                steps_param = np.round(list(np.arange(par_limit[0], par_limit[1], step if step > 0 else 1)), 4) # Lista com os valores que serão definidos em cada treshold
            
            gen_limit = [0, max_gen_type] # Limites das gerações
            n_intervals = 1 / len(steps_param) # Total de tresholds baseado no número de valores da etapa de treshold dos parametros
            step = int((gen_limit[1] - gen_limit[0]) * n_val) # Calcula o step necessário para dividir os dados em n_intervals 
            steps_param_ = list(np.arange(gen_limit[0], gen_limit[1], step if step > 0 else 1)) # Lista com os tresholds

            change_points += list(steps_param_)
            tresholds[param] = dict(zip(steps_param_, steps_param)) # Montagem do dicionário para poder utilizar os tresholds

        return tresholds, list(np.unique(change_points))

    def change_params(self, params:dict, tresholds:dict, current_point:dict):
        '''
        ### Definição

        Método que define qual é o conjunto de parametros da etapa na qual a otimização se encontra. Um dos processos necessários para o POD.

        ### Parametros

        
        '''
        for param in tresholds:
            try:
                params[param] = tresholds[param][current_point]
            except:
                continue
        return params


    # Funções extras
    def remove_one_negative_var(self):
        # Remover feature com menor efeito
        X_train, y_train = self.train.filter(regex='^DTA|FIN_INV'), self.train[self.target_column]
        X_val, y_val = self.val.filter(regex='^DTA|FIN_INV'), self.val[self.target_column]

        # Exibindo as variáveis com coeficientes negativos
        population_temp = list(self.population[0]) # Definir o melhor individuo como único
        coef_df = pd.DataFrame({'coef': self.model.coef_}, index=self.model.feature_names_in_).sort_values('coef')
        coef_df = coef_df[coef_df['coef'] < 0].filter(regex='FIN_INV', axis=0)
        old_neg_coef_count = coef_df.shape[0]

        # Removendo uma variável por vez para coletar a que menos afeta o modelo
        main_removable_var = {'fit':np.inf}
        for not_wanted_var in list(coef_df.index):
            selected_features = list(self.model.feature_names_in_)
            selected_features.remove(not_wanted_var)
            test_model = self.defined_model().fit(X_train[selected_features], y_train)
            # Coletar métricas do novo modelo
            new_model_stats, new_val_stats, _ = self.get_model_metrics(X_val[selected_features], y_val, test_model)
            if new_val_stats['fit_value_no_penalty'] < main_removable_var['fit']:
                main_removable_var = {
                    'column':not_wanted_var,
                    'fit':new_val_stats['fit_value_no_penalty'],
                    'mape':new_val_stats['mape'],
                    'dif1':new_val_stats['dif1'],
                    'dif2':new_val_stats['dif2'],
                    'intercept':new_model_stats['intercept'],
                    'neg_qtt':new_model_stats['neg_qtt'],
                    'features':list(test_model.feature_names_in_)
                }

        for list_id, c in enumerate(list(self.train.filter(regex='^DTA|FIN_INV$').columns)):
            if c == main_removable_var['column']:
                population_temp[list_id] = 0

        orig_c_name = '_'.join(main_removable_var['column'].split('_')[:6])
        for key in self.var_dict:
            if self.var_dict[key] == orig_c_name:
                population_temp[key] = 0
        if self.verbose:
            # Destacando variável removida
            print(f'Variável removida:', colored(f'{main_removable_var["column"]}', 'red'))
            # Modelo antes de remover
            print(colored('|', 'red'), 'Modelo antes de remover')
            print(colored('|', 'red'), 'Fit antigo:', colored(f"{round(self.test_stats['fit_value'], 4):,}", 'magenta'))
            print(colored('|', 'red'), 'Intercept antigo:', colored(f"{round(self.model_stats['intercept'], 2):,}", 'blue'))
            print(colored('|', 'red'), 'Negativas:', colored(f"{old_neg_coef_count:,}", 'red'))
            # Modelo depois de remover
            print(colored('|', 'blue'), 'Modelo depois de remover')
            print(colored('|', 'blue'), 'Fit novo:', colored(f"{round(main_removable_var['fit'], 4):,}", 'magenta'))
            print(colored('|', 'blue'), 'Intercept novo:', colored(f"{round(main_removable_var['intercept'], 2):,}", 'blue'))
            print(colored('|', 'blue'), 'Negativas:', colored(f"{main_removable_var['neg_qtt']:,}", 'red'))

        sleep(1.33)
        if input('Salvar resultado? (S/N)').upper() == 'S':
            # Atualizando o fit, intercept e modelo
            self.test_stats['mape'] = main_removable_var['mape']
            self.test_stats['dif1'] = main_removable_var['dif1']
            self.test_stats['dif2'] = main_removable_var['dif2']
            self.model_stats['intercept'] = main_removable_var['intercept']
            self.model = test_model
            self.population = [population_temp]
            if self.verbose:
                print(colored('As alterações foram salvas!', 'green'))
        else:
            if self.verbose:
                print(colored('As alterações não foram salvas!', 'red'))
        return

    def count_models(self):
        '''
        ### Definição

        Retorna o número de modelos gerados que respeitam as regars definidas.
        '''
        return len(self.models_hist)

    def get_best_models(self):
        '''
        ### Definição

        Retorna 2 listas:
        
        - A primeira contem o ID do melhor modelo de acordo com a diferença no periodo de validação.
        - O segundo contem o ID do melhor modelo de acordo com a diferença no periodo de teste.
        '''
        best_test = np.inf
        best_test_id = 0
        best_val = np.inf
        best_val_id = 0
        for idx, model_hist_ in enumerate(self.models_hist):
            if np.abs(model_hist_['val_stats']['dif2']) < best_val:
                best_val = np.abs(model_hist_['val_stats']['dif2'])
                best_val_id = idx
            if np.abs(model_hist_['test_stats']['dif2']) < best_test:
                best_test = np.abs(model_hist_['test_stats']['dif2'])
                best_test_id = idx

        return [best_val_id, best_val], [best_test_id, best_test]

    def save_model(self,
                   model_id:int,
                   version:str):
        '''
        ### Definição

        Salva o modelo em um arquivo .joblib.
        '''
        joblib_dump(self.models_hist[model_id]['model'], f'{self.product}_{version}.joblib')
        return

    def get_one_model_metrics(self,
                          model_id):
        '''
        ### Definição

        Retorna as métricas do modelo selecionado.
        '''
        if self.count_models() > 0:
            try:
                return self.models_hist[model_id]
            except:
                raise IndexError(f'O ID escolhido não existe, apenas até o ID {self.count_models() - 1}')
        else:
            return None


    # Funções algoritmo genético
    def run_selection(self,
                      starting_step:str='learn',
                      test_n_gen:int|None=None):
        if not self.optimization_configured:
            raise Exception('A otimização não foi configurada, favor executar o método pipeline_config_feature_selection()')
        # Variáveis de uso local
        BREAK = False
        best_fitness = np.inf

        try:
            self.population = [self.population[0]] * self.params['POP_SIZE'] # Utilizar o melhor individuo salvo na memória (caso exista)
        except:
            self.population = [self.variations_set] * self.params['POP_SIZE'] # Gerar novos individuos

        for cycle in range(self.num_cycles):
            if self.verbose:
                print(
                    colored(f"INICIANDO CICLO {cycle} de {self.num_cycles}", "yellow")
                )

            current_generation = 0
            # Inicar com tipo de seleção definida em starting_step
            if cycle == 0:
                self.learn_treshold_params['patience'] = self.patience
                if starting_step == 'learn': first_treshold_params = self.learn_treshold_params
                elif starting_step == 'rest': first_treshold_params = self.rest_treshold_params
                current_step = starting_step # Passo atual será o passo definido como inicial
                tresholds, change_points = self.create_treshold(**first_treshold_params)
                max_gen_type = first_treshold_params['max_gen_type']*2
                del(first_treshold_params)
            # Alterar tipo de seleção
            elif current_step == 'learn': # Tornar seleção como descanso
                current_step = 'rest'
                self.rest_treshold_params['patience'] = self.patience
                tresholds, change_points = self.create_treshold(**self.rest_treshold_params)
                max_gen_type = self.rest_treshold_params['max_gen_type']*2
            # Alterar tipo de seleção
            elif current_step == 'rest': # Tornar seleção como aprendizado
                current_step = 'learn'
                self.learn_treshold_params['patience'] = self.patience
                tresholds, change_points = self.create_treshold(**self.learn_treshold_params)
                max_gen_type = self.learn_treshold_params['max_gen_type']*2

            # Resetar os contadores
            self.patience['overall'] += 1 # Para aumentar os limites
            self.patience['params'] = 0 # Para zerar e reinicar a seleção
            self.patience['type'] = 0 # Para zerar e reinicar a seleção

            # Reiniciar ciclo de seleção
            while self.patience['type'] <= max_gen_type:
                if current_step == 'learn':
                    need_best_reset = True
                self.params = self.change_params(self.params, tresholds, self.patience['params']) # Definição dos parametros modificaveis
                change_points.remove(self.patience['params']) # Sempre remover ponto de mudança ao utilizá-lo (para que os parametros não se repitam)
                if self.verbose:
                    print(
                        colored(f"Ponto de alteração {self.patience['params']} atingido, falta(m) {len(change_points)} ponto(s)", "yellow")
                    )

                self.num_remove = self.params['NUM_REMOVE']
                self.num_high_remove = self.params['NUM_HIGH_REMOVE']
                self.num_add = self.params['NUM_ADD']
                self.pop_size = self.params['POP_SIZE']
                KEEP = self.params['KEEP']
                self.transfer_var = self.params['TRANSFER_VAR']
                self.negative_weight = self.params['NEGATIVE_WEIGHT']
                self.change_chance = self.params['CHANGE_CHANCE']
                self.max_coef = self.params['MAX_COEF']

                while self.patience['params'] not in change_points and self.patience['type'] <= max_gen_type:
                    # Incrementar os contadores
                    self.patience['params'] += 1 # Para zerar e reinicar a seleção
                    self.patience['type'] += 1 # Para zerar e reinicar a seleção

                    # Gerar população aleatória caso seja necessário
                    for _ in range(self.random_count):
                        self.population = self.population + self.generate_random_genome()

                    # Calcular melhor fitness e checar o motivo para ter parado o processo de evolução
                    self.population = sorted(
                        self.population,
                        key = self.fitness,
                        reverse=False
                    )
                    model_stats, val_stats, test_stats, model = self.fitness_and_metrics(self.population[0])
                    self.model_stats = model_stats.copy()
                    self.val_stats = val_stats.copy()
                    self.test_stats = test_stats.copy()
                    self.model = model
                    # Em caso de teste esse try será acionado e quando atingir a geração limite, a seleção vai ser parada
                    try:
                        if current_generation >= test_n_gen:
                            BREAK = True
                            break
                    except: pass
                    if val_stats['fit_value'] < best_fitness or need_best_reset:
                        need_best_reset = False
                        # Resetar contadores em caso de melhoria
                        self.patience['params'] = 0 # Para zerar e reinicar a seleção
                        # patience['type'] = 0 # Para zerar e reinicar a seleção

                        # Salvar melhor genoma e melhor fitness
                        best_fitness = val_stats['fit_value']
                        if current_step == 'rest':
                            if self.verbose:
                                print(
                                    colored(f"{cycle}|{current_step} - {current_generation}:", "yellow"),
                                    "fit", colored(f"{round(val_stats['fit_value'],3):,}", "blue"),
                                    "r2", colored(f"{round(val_stats['r2'],3):,}", "magenta"),
                                    "mape", colored(f"{round(val_stats['mape'],3):,}", "magenta"),
                                    "dif", colored(f"{round(val_stats['dif2'],3):,}", "magenta"),
                                    "intercept", colored(f"{round(model_stats['intercept'],2):,}", "red"),
                                    "neg_qtt", colored(f"{model_stats['neg_qtt']:,}", "red"),
                                )
                        elif current_step == 'learn':
                            if val_stats['fit_value_no_penalty'] == val_stats['fit_value']:
                                if self.verbose:
                                    print(
                                        colored(f"{cycle}|{current_step} - {current_generation}:", "magenta"),
                                        "fit", colored(f"{round(val_stats['fit_value'],3):,}", "blue"),
                                        "r2", colored(f"{round(val_stats['r2'],3):,}", "magenta"),
                                        "mape", colored(f"{round(val_stats['mape'],3):,}", "magenta"),
                                        "dif", colored(f"{round(val_stats['dif2'],3):,}", "magenta"),
                                        "intercept", colored(f"{round(model_stats['intercept'],2):,}", "red"),
                                        "neg_qtt", colored(f"{model_stats['neg_qtt']:,}", "red"),
                                    )
                                self.models_hist.append({ # Adicionando qualquer modelo que respeite as regras
                                    'model': model,
                                    'val_stats': val_stats.copy(),
                                    'test_stats': test_stats.copy(),
                                    'model_stats': model_stats.copy(),
                                })
                                if val_stats['fit_value'] < self.best_fit:
                                    self.best_model_fit = model # Definindo o modelo que respeita as regras com o maior fit
                                    self.best_fit = val_stats['fit_value']
                                    self.best_model_metrics = [val_stats['mape'], val_stats['dif2']]
                                    if self.verbose:
                                        print(
                                            colored(f"Novo melhor modelo com base no fit! fit({round(self.best_fit,3):,}) intercept({round(model_stats['intercept'],2):,})", 'green')
                                        )
                                if val_stats['fit_value'] == 0:
                                    BREAK = True
                                    break
                            else:
                                if self.verbose:
                                    print(
                                        colored(f"{cycle}|{current_step} - {current_generation}:", "blue"),
                                        "fit", colored(f"{round(best_fitness,3):,}", "blue"),
                                        "r2", colored(f"{round(val_stats['r2'],3):,}", "magenta"),
                                        "mape", colored(f"{round(val_stats['mape'],3):,}", "magenta"),
                                        "dif", colored(f"{round(val_stats['dif2'],3):,}", "magenta"),
                                        "intercept", colored(f"{round(model_stats['intercept'],2):,}", "red"),
                                        "neg_qtt", colored(f"{model_stats['neg_qtt']:,}", "red"),
                                    )

                    # Continue evolution process to next generation
                    ## Keep some genomes
                    next_generation = self.population[:KEEP]
                    ## Crossover and mutation
                    for _ in range(round(self.params['POP_SIZE'] / 2) - 1):
                        '''
                        Combine two random genomes of the old population and create new genomes.
                        '''
                        parents = self.pair_selection(self.population[0:round(self.params['POP_SIZE'] / 2)])
                        child_a, child_b = self.crossover(parents[0], parents[1], 1)
                        child_a = self.mutation(child_a, model)
                        child_b = self.mutation(child_b, model)
                        next_generation = next_generation + [child_a, child_b]

                    self.population = next_generation
                    current_generation += 1
                    continue
                if BREAK: break
            if BREAK: break
        return

    def pair_selection(self, population:list):
        '''
        ### Definição

        Selecionar um par aleatório entre a população
        '''
        return choices(population, k=2)

    def crossover(self, a:list[int], b:list[int], type:int=0):
        '''
        ### Definição

        Realiza o crossing over, combinação do genome entre dois individuos
        '''
        if type == 0:
            genome_lenght = len(a)
            if genome_lenght < 2:
                return a, b
            
            cut = randint(1, genome_lenght - 1)
            return a[:cut] + b[cut:], b[:cut] + a[cut:]
        elif type == 1:
            selected_a = choices(list(range(len(a))), k=self.transfer_var)
            selected_b = choices(list(range(len(b))), k=self.transfer_var)
            a_c = a.copy()
            b_c = b.copy()
            a = [b_c[idx] if idx in selected_b else value for idx, value in enumerate(a_c)]
            b = [a_c[idx] if idx in selected_a else value for idx, value in enumerate(b_c)]
            return a, b

    def mutation(self, variations_set:list[int], model:pd.DataFrame) -> list[int]:
        '''
        ### Definição

        Realiza a mutação do individuo.
        '''
        for _ in range(self.num_add):
            if random() > 1 - (self.change_chance*1.0):
                choosen_element = choice(list(self.var_dict.keys()))
                variations_set[choosen_element] = choice(self.fin_inv_transformations) if 'FIN_INV' in self.var_dict[choosen_element] else choice(self.dta_transformations)

        try:
            coef_df = pd.DataFrame({'coef': model.coef_}, index=model.feature_names_in_)
        except:
            pass

        # Removing negative coefs #
        try:
            neg_coef_df = coef_df.filter(regex='FIN_INV', axis=0)
            neg_coef_df = neg_coef_df[neg_coef_df['coef'] < 0]
            neg_coef_df.sort_values('coef', inplace=True)
            if neg_coef_df.shape[0] == 0:
                raise ValueError('No negative coefs')
            # remove_this = sample(list(neg_coef_df.index), k=self.num_remove if len(list(neg_coef_df.index)) > self.num_remove else len(list(neg_coef_df.index)))
            remove_this = list(neg_coef_df[-self.num_remove:].index)
            for rem in remove_this:
                if random() > 1 - (self.change_chance*1.0):
                    variations_set[list(self.var_dict.keys())[list(self.var_dict.values()).index(rem[:23])]] = 0
        except:
            pass
        # ----------------------- #

        # Removing positive coefs #
        try:
            pos_coef_df = coef_df.filter(regex='FIN_INV', axis=0)
            pos_coef_df = pos_coef_df[pos_coef_df['coef'] > self.max_coef]
            pos_coef_df.sort_values('coef', ascending=False, inplace=True)
            if pos_coef_df.shape[0] == 0:
                raise ValueError(f'No above {self.max_coef} coef')
            # remove_this = sample(list(pos_coef_df.index), k=self.num_high_remove if len(list(pos_coef_df.index)) > self.num_high_remove else len(list(pos_coef_df.index)))
            remove_this = list(pos_coef_df[:self.num_high_remove].index)
            for rem in remove_this:
                if random() > 1 - (self.change_chance*1.0):
                    variations_set[list(self.var_dict.keys())[list(self.var_dict.values()).index(rem[:23])]] = 0
        except:
            pass
        # ----------------------- #

        return variations_set

    def generate_random_genome(self):
        fin_inv_transformations = self.fin_inv_transformations
        dta_transformations = self.dta_transformations

        features_to_set = sample(list(range(len(self.variations_set))), k=self.initial_rfe)
        # Gerar genoma #
        random_genome = []
        for idx, gene in enumerate(self.variations_set):
            if idx in features_to_set: # Adicionar feature
                if gene < 0:
                    random_genome.append(choice(dta_transformations))
                elif gene >= 0:
                    random_genome.append(choice(fin_inv_transformations))
            else: # Não adicionar feature
                if gene < 0:
                    random_genome.append(-2)
                elif gene >= 0:
                    random_genome.append(0)
        return [random_genome]

    # Funções de fitness e métricas
    def fitness(self, variations_set:list):
        '''
        ### Definição

        Função para calcular apenas o fitness do individuo. É mais rápida que a função que calcula as métricas junto, serve para ser executada durante o ordenamento dos individuos.

        ### Parametros

        •	variations_set (list | obrigatório) - Lista com os dados do individuo.
        '''
        # Variáveis de uso local
        train = self.train.copy()
        val = self.val.copy()
        variations = self.variations.copy()
        product_column = self.target_column
        var_dict = self.var_dict.copy()
        defined_model = self.defined_model

        columns = [f"{var_dict[i].replace('FIN_INV', '')}{variations[variations_set[i]]}" for i in range(len(variations_set)) if f"{var_dict[i].replace('FIN_INV', '')}{variations[variations_set[i]]}" in train.columns]

        X_train, y_train = train[columns], train[product_column]
        X_val, y_val = val[columns], val[product_column]

        model = defined_model().fit(X_train, y_train)
        fit_value = self.get_model_fitness(X_val, y_val, model)

        return fit_value

    def calc_dif(self, pred:float, real:float, method:int=0):
        '''
        ### DEFINIÇÃO

        Calcula apenas a diferença (métrica). Para centralizar o cálculo em uma única funcão.

        ### PARAMETROS

        •	pred (obrigatório | float) - Total predito.

        •	real (obrigatório | float) - Total realizado.

        •	method (obrigatório | int) - Método que será utilizado para calcular a diferença.
        '''
        if method == 0:
            train_metric = self.train_metric
            if 'dif1' in train_metric:
                dif = 1 - (pred / real) # Método 1 - Penaliza os que preveêm menos
            if 'dif2' in train_metric:
                dif = 1 - (real / pred) # Método 2 - Equilibrado
        elif method == 1:
            dif = 1 - (pred / real) # Método 1 - Penaliza os que preveêm menos
        elif method == 2:
            dif = 1 - (real / pred) # Método 2 - Equilibrado
        return dif

    def fitness_and_metrics(self, variations_set:list):
        '''
        ### Definição

        Função para calcular o fitness do individuo e algumas metricas

        ### Parametros

        •	variations_set (list | obrigatório) - Lista com os dados do individuo.
        '''
        # Variáveis de uso local
        train = self.train.copy()
        val = self.val.copy()
        test = self.test.copy()
        variations = self.variations.copy()
        product_column = self.target_column
        var_dict = self.var_dict.copy()
        defined_model = self.defined_model

        # columns = [var_dict[i].replace('FIN_INV', variations[variations_set[i]]) for i in range(len(variations_set)) if var_dict[i].replace('FIN_INV', variations[variations_set[i]]) in train.columns]
        columns = [f"{var_dict[i].replace('FIN_INV', '')}{variations[variations_set[i]]}" for i in range(len(variations_set)) if f"{var_dict[i].replace('FIN_INV', '')}{variations[variations_set[i]]}" in train.columns]

        X_train, y_train = train[columns], train[product_column]
        X_val, y_val = val[columns], val[product_column]
        X_test, y_test = test[columns], test[product_column]
        test_X_y = [X_test, y_test]

        model = defined_model().fit(X_train, y_train)
        model_stats, val_stats, test_stats = self.get_model_metrics(X_val, y_val, model, test_X_y)

        # Fitness, modelo, métricas, intercepto + quantidade de negativos
        return model_stats, val_stats, test_stats, model

    def get_model_fitness(self, X:pd.DataFrame, y:any, model:any, return_no_neg=False):
        '''
        ### DEFINIÇÃO

        Calcula apenas o fitness do modelo. É uma variação da função get_model_metrics para coletar apenas o fitness e executar a evolução mais rápido.

        ### PARAMETROS

        •	X (obrigatório | pd.DataFrame) - Dataframe com os dados de investimento.

        •	y (obrigatório | list like) - mesmo tamanho do X, é o target do modelo.

        •	model (obrigatório | modelo) - modelo a ser estudado.

        •	return_no_neg (opcional | bool) - Se ao final será retornado o fit sem considerar as variáveis negativas.
        '''
        # Variáveis de uso local
        train_metric = self.train_metric
        negative_weight = self.negative_weight
        ignore_neg_coef = self.ignore_neg_coef

        # Contagem negativas #
        if not ignore_neg_coef:
            coef_df = pd.DataFrame({'coef': model.coef_}, index=model.feature_names_in_).filter(regex='FIN_INV', axis=0)
            negative_coefs = len(coef_df[coef_df['coef'] < 0])
        elif ignore_neg_coef:
            negative_coefs = 0
        # ------------------ #
        
        # Predição #
        y_pred = model.predict(X)
        # -------- #

        # Calculo das métricas #
        if 'mape' in train_metric:
            mape = mean_absolute_percentage_error(y, y_pred) * train_metric['mape']
        else:
            mape = 0
        if 'dif1' in train_metric:
            dif = np.abs(self.calc_dif(sum(y_pred), y.sum())) * train_metric['dif1']
        elif 'dif2' in train_metric:
            dif = np.abs(self.calc_dif(sum(y_pred), y.sum())) * train_metric['dif2']
        else:
            dif = 0
        if 'r2' in train_metric:
            r2 = r2_score(y, y_pred)
            if r2 >= 0:
                r2 = (1 - r2) * train_metric['r2']
            elif r2 < 0:
                r2 = (np.abs(r2) + 1) * train_metric['r2']
        else:
            r2 = 0
        # -------------------- #

        # Calculo do fitness do individuo #
        fit_value_no_penalty = mape + dif + r2
        try:
            fit_value = fit_value_no_penalty + (negative_coefs * negative_weight)
        except:
            fit_value = fit_value_no_penalty
        # ------------------------------- #

        if not return_no_neg:
            return fit_value
        elif return_no_neg:
            return fit_value, fit_value_no_penalty

    def get_model_metrics(self, X_val:pd.DataFrame, y_val:any, model:any, test_X_y:list=[None, None]):
        '''
        ### DEFINIÇÃO

        Função que retorna as métricas e fitness do individuo com base no parametros enviados.

        ### PARAMETROS

        •	X_val (obrigatório | pd.DataFrame) - Dataframe com os dados de investimento.

        •	y_val (obrigatório | list like) - mesmo tamanho do X, é o target do modelo.

        •	model (obrigatório | modelo) - modelo a ser estudado.

        •	test_X_y (opcional | list) - base de validação em forma de lista, contendo o X e y de validação nessa ordem caso seja necessario.
        '''
        # Variáveis locais
        ignore_neg_coef = self.ignore_neg_coef

        # 1. Salvando dados do modelo #
        # Contagem negativas
        if not ignore_neg_coef:
            coef_df = pd.DataFrame({'coef': model.coef_}, index=model.feature_names_in_).filter(regex='FIN_INV', axis=0)
            negative_coefs = len(coef_df[coef_df['coef'] < 0])
        elif ignore_neg_coef:
            negative_coefs = 0
        try:
            current_intercept = model.intercept_
        except:
            current_intercept = 0
        model_stats = {
            'neg_qtt': negative_coefs,
            'intercept': current_intercept,
        }
        # 1. ######################## #

        # 2. Salvando métricas de validação #
        # Predição
        y_pred = model.predict(X_val)
        # Calculo das métricas
        mape = mean_absolute_percentage_error(y_val, y_pred)
        dif1 = self.calc_dif(sum(y_pred), y_val.sum(), 1)
        dif2 = self.calc_dif(sum(y_pred), y_val.sum(), 2)
        r2 = r2_score(y_val, y_pred)
        fit_value, fit_value_no_penalty = self.get_model_fitness(X_val, y_val, model, True)
        # Salvando dados sobre a base de test
        val_stats = {
            'fit_value': fit_value,
            'fit_value_no_penalty': fit_value_no_penalty,
            'mape': mape,
            'dif1': dif1,
            'dif2': dif2,
            'r2': r2,
        }
        # 2. ########################## #

        # 3. Salvando métricas de teste #
        if type(test_X_y[0]) != type(None) and type(test_X_y[1]) != type(None):
            # Predição
            y_pred = model.predict(test_X_y[0])
            # Calculo das métricas
            mape = mean_absolute_percentage_error(test_X_y[1], y_pred)
            dif1 = self.calc_dif(sum(y_pred), test_X_y[1].sum(), 1)
            dif2 = self.calc_dif(sum(y_pred), test_X_y[1].sum(), 2)
            r2 = r2_score(test_X_y[1], y_pred)
            fit_value, fit_value_no_penalty = self.get_model_fitness(test_X_y[0], test_X_y[1], model, True)
            test_stats = {
                'fit_value': fit_value,
                'fit_value_no_penalty': fit_value_no_penalty,
                'mape': mape,
                'dif1': dif1,
                'dif2': dif2,
                'r2': r2,
            }
        else: test_stats = None
        # 3. ############################## #

        return model_stats, val_stats, test_stats

    # Funções de pipeline prontas
    def pipeline_config_feature_selection(self,
                                          config:dict={},
                                          preselected:list=[],
                                          pod_config:dict|None=None):
        '''
        ### Definição

        ### Parametros

        •	pod_config (dict, None | opcional) - Parametros de Otimização Dinamicos. Dicinionário que configura os parametros dinâmicos para a otimização.
        '''
        # Configuração inicial do processo
        self.config_feature_selection(config, preselected)
        # Definindo POD
        self.save_pod(pod_config)
        # Seleionando variáveis iniciais
        self.set_best_preselected()
        # Resetando (ou definindo) os resultados de otimização
        self.reset_best_results()
        # Salvando que a configuração foi executada
        self.optimization_configured = True
        return

    def pipeline_run_selection(self,
                               starting_step:str='learn',
                               test_n_gen:int|None=None,
                               config:dict={},
                               preselected:list=[],
                               pod_config:dict|None=None):
        '''
        ### DEFINIÇÃO

        Método que executa a pipeline de teste de alguma abordagem desejada.

        ### PARAMETROS

        •	xxx (obrigatório | pd.DataFrame) - abc.
        '''
        # Configurar otimização caso não esteja configurada
        self.pipeline_config_feature_selection(config, preselected, pod_config)

        # Executar teste N vezes
        self.run_selection(starting_step, test_n_gen)

        # Retornar dataframe com informação dos testes
        return


