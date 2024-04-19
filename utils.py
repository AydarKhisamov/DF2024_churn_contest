import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from feature_extraction import *


class Dataset(object):
    """Класс датасета."""

    def __init__(self, standardize: bool = True):
        """
        Инициализация объекта класса.

        Parameters:
            standardize:
                Флаг стандартизации данных.
        """

        self.standardize = standardize


    def build(self,
              transactions: pd.DataFrame,
              client_info: pd.DataFrame,
              period_info: pd.DataFrame,
              targets: pd.DataFrame = None,
              fit_params: bool = False,
              sample: str = 'train'):
        """
        Создаёт датасет в подходящем для обучения модели виде.

        Parameters:
            transactions:
                Датафрейм с логами транзакций.

            client_info:
                Датафрейм с соц.-дем. признаками клиентов.

            period_info:
                Датафрейм, где каждому пользователю соответствует список дат в
                формате ММ.ГГГГ, в которые проходил сбор информации.

            targets:
                Датафрейм с целевым признаком.

            fit_params:
                Флаг сохранения параметров извлечения признаков.

            sample:
                Выборка, по которой создаётся датасет.
        """

        # инициализация датасета
        dset = pd.DataFrame({'user_id': transactions['user_id'].unique()})

        # добавление инфо о периодах сбора данных в датасет
        dset = dset.merge(period_info, on='user_id')
        dset = dset.explode('transaction_mnth', ignore_index=True)

        if fit_params:
            # кол-во транзакций каждого клиента за каждый месяц наблюдения
            # по каждому mcc-коду
            df = transactions.groupby(
                by=['user_id', 'transaction_mnth', 'mcc_code'],
                ).agg(t_cnt=('currency_rk', 'count'))
            df = df.unstack(level='mcc_code').fillna(value=0)
            df = df.droplevel(level=0, axis=1)

            # инициализация и подгонка параметров PCA
            pca = PCA(n_components=12, random_state=42)
            pca.fit(df.values)

            # расчёт факторных нагрузок mcc-кодов
            f_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

            # создание словаря {mcc-код: группа mcc-кодов}
            mcc_group_pairs = zip(df.columns, np.argmax(f_loadings, axis=1))
            self.mcc_dict = {k: v for k, v in mcc_group_pairs}

        # датафреймы с рассчитанными признаками по транзакциям
        df1 = get_t_count(transactions)
        df2 = get_t_count_in_foreign_currency(transactions)
        df3 = get_unique_mcc_count(transactions)
        df4 = get_daydiff_median(transactions, period_info)
        df5 = get_cyclic_t_count(transactions, cycle_base='day')
        df6 = get_cyclic_t_count(transactions, cycle_base='weekday & weeknum')
        df7 = get_mcc_intersect_ratio(transactions, period_info)
        df8 = get_daydiff_max(transactions, period_info)
        df9 = get_grouped_t_count(transactions, self.mcc_dict)

        # соединение датафреймов с признаками
        df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9], axis=1)
        stat_features = df.columns.tolist() # статистические признаки
        df.reset_index(inplace=True)

        # присоединение признаков к датасету
        dset = dset.merge(df, on=['user_id', 'transaction_mnth'], how='left')
        dset = dset.fillna(value=0)

        # создание признака "день сбора статистики"
        dset.rename(columns={'transaction_mnth': 'stat_date'}, inplace=True)
        dset['stat_date'] = dset['stat_date'].dt.to_timestamp('D', 'E').dt.date

        # добавление соц.-дем. признаков пользователей
        dset = dset.merge(
            client_info.drop(columns=['report', 'report_dt']),
            on='user_id',
            )

        # стандартизация данных
        if self.standardize:
            if fit_params:
                self.std_scaler = StandardScaler()
                dset[stat_features] = self.std_scaler.fit_transform(dset[stat_features])
            else:
                dset[stat_features] = self.std_scaler.transform(dset[stat_features])

        if targets is not None:

            # последняя дата периода сбора транзакций по пользователям
            last_date = dset.groupby(by='user_id', as_index=False).agg(
                last_date=('stat_date', 'max')
                )

            # замена значения дней после завершения сбора транзакций на дату
            # последней транзакции
            targets = targets.merge(last_date, on='user_id')
            targets['time'] = targets.apply(
                lambda x: x['last_date'] + pd.Timedelta(x['time'], unit='d'),
                axis=1,
                )
            dset = dset.merge(targets.drop(columns='last_date'), on='user_id')
            dset['time'] = dset['time'] - dset['stat_date']
            dset['time'] = pd.to_timedelta(dset['time'], unit='d').dt.days
            dset.drop(columns='stat_date', inplace=True)

            # возвращение отдельно списков с массивами признаков,
            # таргетов событий, времени до события
            dset.set_index('user_id', inplace=True)
            x = dset.drop(['time', 'target'], axis=1).groupby(level=0).apply(
                lambda x: x.values
                )
            ids = x.index.values
            x = x.values

            if sample == 'valid':
                # возвращается время до события и флаг события с даты
                # последнего сбора информации
                t = dset['time'].groupby(level=0).tail(1).values
                e = dset['target'].groupby(level=0).tail(1).values

            elif sample == 'train':
                # возвращается время до события и флаг события с каждой из дат
                # сбора информации
                t = dset['time'].groupby(level=0).apply(lambda x: x.values).values
                e = dset['target'].groupby(level=0).apply(lambda x: x.values).values

            return ids, x, t, e

        else:

            # возвращение списка с массивами признаков
            dset.drop(columns='stat_date', inplace=True)
            dset.set_index('user_id', inplace=True)
            dset = dset.groupby(level=0).apply(lambda x: x.values)
            ids = dset.index.values
            x = dset.values

            return ids, x