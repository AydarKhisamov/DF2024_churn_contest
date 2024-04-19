import numpy as np
import pandas as pd


def get_t_count(transactions: pd.DataFrame):
    """
    Возвращает кол-во транзакций за месяц.

    Parameters:
        transactions:
            Датафрейм с логами транзакций.
    """

    df = transactions.copy()

    df = df.groupby(by=['user_id', 'transaction_mnth']).agg(
        t_cnt=('mcc_code', 'count'),
        )

    return df


def get_t_count_in_foreign_currency(transactions: pd.DataFrame):
    """
    Возвращает кол-во транзакций в зарубежной валюте за месяц.

    Parameters:
        transactions:
            Датафрейм с логами транзакций.
    """

    df = transactions[transactions['currency_rk'] != 1].copy()

    df = df.groupby(by=['user_id', 'transaction_mnth']).agg(
        t_cnt_in_foreign_currency=('mcc_code', 'count'),
        )

    return df


def get_unique_mcc_count(transactions: pd.DataFrame):
    """
    Возвращает кол-во уникальных mcc-кодов транзакций за месяц.

    Parameters:
        transactions:
            Датафрейм с логами транзакций.
    """

    df = transactions.copy()

    df = df.groupby(by=['user_id', 'transaction_mnth']).agg(
        unique_mcc_cnt=('mcc_code', 'nunique'),
        )

    return df


def get_daydiff_median(transactions: pd.DataFrame, period_info: pd.DataFrame):
    """
    Возвращает максимальную дистанцию в днях между транзакциями за месяц.

    Parameters:
        transactions:
            Датафрейм с логами транзакций.

        period_info:
            Датафрейм, где каждому пользователю соответствует список дат в
            формате ММ.ГГГГ, в которые проходил сбор информации.
    """
    df1 = period_info.explode('transaction_mnth')
    df2 = transactions.copy()

    # каждому пользователю соответствует массив дат всех транзакций за месяц
    df2 = df2.groupby(
        by=['user_id', 'transaction_mnth'],
        )['transaction_dt'].apply(np.array)
    df2 = df2.reset_index()

    # добавление дат ко всем месяцам наблюдений
    df = df1.merge(df2, on=['user_id', 'transaction_mnth'], how='left')

    # добавление даты первого и последнего дня месяца наблюдения
    df['t_mnth_start'] = df['transaction_mnth'].dt.start_time.dt.date
    df['t_mnth_end'] = df['transaction_mnth'].dt.end_time.dt.date

    # nan_indices - индексы наблюдений, где за месяц у клиента нет транзакций,
    # true_indices - наоборот
    nan_indices = df[df['transaction_dt'].isna()].index
    true_indices = df[~df['transaction_dt'].isna()].index

    # добавление к массиву с датами транзакций дат начала и конца месяца
    df.loc[nan_indices, 'transaction_dt'] = df.apply(
        lambda x: np.hstack((x['t_mnth_start'], x['t_mnth_end'])), axis=1,
        )

    df.loc[true_indices, 'transaction_dt'] = df.apply(
        lambda x: np.hstack(
            (x['t_mnth_start'], x['transaction_dt'],  x['t_mnth_end'])
            ),
        axis=1,
        )

    # максимальная дистанция между транзакциями с имитацией транзакций в
    # начале и конце месяца
    df['daydiff_med'] = df['transaction_dt'].apply(
        lambda x: np.median(x[1:] - x[:-1]).days,
        )

    # использование в качестве индексов id клиента и месяца наблюдения
    df = df.set_index(keys=['user_id', 'transaction_mnth'])['daydiff_med']

    return df


def get_cyclic_t_count(transactions: pd.DataFrame, cycle_base: str):
    """
    Возвращает кол-во повторяющихся за прошлым месяцем mcc-кодов транзакций по
    одному из оснований.

    Parameters:
        transactions:
            Датафрейм с логами транзакций.

        cycle_base:
            Критерий для определения транзакции как повторной.
    """

    df = transactions.copy()
    feature_name = f'cycled_by_{cycle_base}_t_cnt'

    # критерий: день
    # транзакция считается повторной, если в прошлом месяце была транзакция
    # с этим же mcc-кодом в этот же день
    if cycle_base == 'day':
        b_cols = ['user_id', 'mcc_code', 'transaction_day'] # basic columns
        df = df[df[b_cols].duplicated(keep=False)]

    # критерий: конец месяца
    # транзакция в последний день месяца считается повторной, если в последний
    # день прошлого месяца была транзакция с этим же mcc-кодом
    elif cycle_base == 'month_end':
        b_cols = ['user_id', 'mcc_code']
        df = df[(df['is_month_end']) & (df[b_cols].duplicated(keep=False))]

    # критерий: номер недели и день недели
    # транзакция считается повторной, если в прошлом месяце была транзакция с
    # этим же mcc-кодом в эту же по порядку неделю в этот же день недели
    elif cycle_base == 'weekday & weeknum':
        b_cols = ['user_id', 'mcc_code', 'day_of_week', 'num_of_dow_in_month']
        df = df[df[b_cols].duplicated(keep=False)]

    dd_cols = b_cols + ['transaction_mnth'] # drop_duplicates columns
    all_cols = dd_cols + ['transaction_dttm'] # all used columns

    df = df.drop_duplicates(subset=dd_cols)[all_cols]
    df = df.sort_values(by='transaction_dttm')

    df = pd.merge_asof(
        df,
        df,
        on='transaction_dttm',
        by=b_cols,
        suffixes=(None, '_prev'),
        allow_exact_matches=False,
        )

    df = df[df['transaction_mnth'] - 1 == df['transaction_mnth_prev']]

    df = df.groupby(by=['user_id', 'transaction_mnth']).agg(
        feature=('mcc_code', 'count')
        )

    df = df.rename(columns={'feature': f'{feature_name}'})

    return df


def get_mcc_intersect_ratio(transactions: pd.DataFrame,
                            period_info: pd.DataFrame):
    """
    Возвращает долю от числа mcc-кодов за месяц, по которым были транзакции
    в прошлом месяце.

    Parameters:
        transactions:
            Датафрейм с логами транзакций.

        period_info:
            Датафрейм, где каждому пользователю соответствует список дат в
            формате ММ.ГГГГ, в которые проходил сбор информации.        
    """
    df1 = period_info.explode('transaction_mnth', ignore_index=True)
    df2 = transactions.copy()

    # массив уникальных mcc-кодов за каждый месяц
    df2 = df2.groupby(by=['user_id', 'transaction_mnth'], as_index=False).agg(
        mcc_codes=('mcc_code', 'unique'))

    df = df1.merge(df2, on=['user_id', 'transaction_mnth'], how='left')

    # замещение пустым списком случаев, где у клиента нет транзакций за месяц
    nan_indices = df[df['mcc_codes'].isna()].index
    nan_lists = df.loc[nan_indices, 'mcc_codes'].apply(lambda x: [])
    df.loc[nan_indices, 'mcc_codes'] = nan_lists

    # сортировка по месяцу наблюдения
    df.sort_values(by='transaction_mnth', inplace=True)

    # присоединение справа массивов mcc-кодов за предыдущий месяц
    df = pd.merge_asof(
        df,
        df,
        on='transaction_mnth',
        by='user_id',
        suffixes=(None, '_prev'),
        allow_exact_matches=False,
        )

    # замещение пустым списком mcc-кодов за предшествующий до наблюдений месяц
    nan_indices = df[df['mcc_codes_prev'].isna()].index
    nan_lists = df.loc[nan_indices, 'mcc_codes_prev'].apply(lambda x: [])
    df.loc[nan_indices, 'mcc_codes_prev'] = nan_lists

    # массив с пересекающимися mcc-кодами
    df['mcc_isect'] = df.apply(
        lambda x: np.intersect1d(
            x['mcc_codes'],
            x['mcc_codes_prev'],
            assume_unique=True,
            ),
        axis=1,
        )

    # доля пересекающихся mcc-кодов от числа всех
    df['mcc_isect_ratio'] = df.apply(
        lambda x: np.round(
            a=len(x['mcc_isect']) / len(x['mcc_codes']),
            decimals=2,
            ) if len(x['mcc_codes']) > 0 else 0,
        axis=1,
        )

    df.set_index(keys=['user_id', 'transaction_mnth'], inplace=True)
    return df['mcc_isect_ratio']


def get_daydiff_max(transactions: pd.DataFrame, period_info: pd.DataFrame):
    """
    Возвращает максимальную дистанцию в днях между транзакциями за месяц.

    Parameters:
        transactions:
            Датафрейм с логами транзакций.

        period_info:
            Датафрейм, где каждому пользователю соответствует список дат в
            формате ММ.ГГГГ, в которые проходил сбор информации.            
    """
    df1 = period_info.explode('transaction_mnth', ignore_index=True)
    df2 = transactions.copy()

    # каждому пользователю соответствует массив дат всех транзакций за месяц
    df2 = df2.groupby(
        by=['user_id', 'transaction_mnth'],
        )['transaction_dt'].apply(np.array)
    df2 = df2.reset_index()

    # добавление дат ко всем месяцам наблюдений
    df = df1.merge(df2, on=['user_id', 'transaction_mnth'], how='left')

    # добавление даты первого и последнего дня месяца наблюдения
    df['t_mnth_start'] = df['transaction_mnth'].dt.start_time.dt.date
    df['t_mnth_end'] = df['transaction_mnth'].dt.end_time.dt.date

    # nan_indices - индексы наблюдений, где за месяц у клиента нет транзакций,
    # true_indices - наоборот
    nan_indices = df[df['transaction_dt'].isna()].index
    true_indices = df[~df['transaction_dt'].isna()].index

    # добавление к массиву с датами транзакций дат начала и конца месяца
    df.loc[nan_indices, 'transaction_dt'] = df.apply(
        lambda x: np.hstack((x['t_mnth_start'], x['t_mnth_end'])), axis=1,
        )

    df.loc[true_indices, 'transaction_dt'] = df.apply(
        lambda x: np.hstack(
            (x['t_mnth_start'], x['transaction_dt'],  x['t_mnth_end'])
            ),
        axis=1,
        )

    # максимальная дистанция между транзакциями с имитацией транзакций в
    # начале и конце месяца
    df['daydiff_max'] = df['transaction_dt'].apply(
        lambda x: np.max(x[1:] - x[:-1]).days,
        )

    # использование в качестве индексов id клиента и месяца наблюдения
    df = df.set_index(keys=['user_id', 'transaction_mnth'])['daydiff_max']

    return df


def get_grouped_t_count(transactions: pd.DataFrame, mcc_dict: dict):
    """
    Возвращает кол-во транзакций за месяц по группам mcc-кодов.

    Parameters:
        transactions:
            Датафрейм с логами транзакций.

        mcc_dict:
            Словарь со схемой {mcc-код: группа mcc-кодов}.
    """

    df = transactions.copy()

    # кол-во транзакций каждого пользователя за каждый месяц по каждой группе
    # кодов
    df['mcc_group'] = df['mcc_code'].map(mcc_dict)
    df = df.groupby(
        by=['user_id', 'transaction_mnth', 'mcc_group'],
        )['mcc_code'].count()

    df = df.unstack(level='mcc_group')
    df.columns = [f'mcc_gr{col}_t_cnt' for col in df.columns]

    return df