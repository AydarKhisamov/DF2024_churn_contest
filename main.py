import numpy as np
import pandas as pd
from auton_survival.models.dsm import DeepRecurrentSurvivalMachines
from utils import Dataset

transactions = pd.read_csv('.../transactions.csv')
clients = pd.read_csv('.../clients.csv')
train = pd.read_csv('.../train.csv')
report_dates = pd.read_csv('.../report_dates.csv')

# операции с датафреймом транзакций
# представление datetime транзакции в виде date
transactions['transaction_dttm'] = pd.to_datetime(transactions['transaction_dttm'])
transactions['transaction_dt'] = transactions['transaction_dttm'].dt.date

# добавление признаков год-месяц транзакции и день месяца транзакции
transactions['transaction_mnth'] = transactions['transaction_dttm'].dt.to_period('M')
transactions['transaction_day'] = transactions['transaction_dttm'].dt.day

# бинарный признак "конец месяца"
transactions['is_month_end'] = transactions['transaction_dttm'].dt.is_month_end

# признак "день недели"
transactions['day_of_week'] = transactions['transaction_dttm'].dt.day_of_week

# порядковый номер дня недели в этом месяце
transactions['num_of_dow_in_month'] = np.ceil(transactions['transaction_day'] / 7)

# операции с датафреймом данных по клиентам
# добавление к данным о клиенте даты отчёта
clients = clients.merge(report_dates, on='report')
clients['report_dt'] = pd.to_datetime(clients['report_dt'])

# словарь для кодирования признака численности сотрудников в
# компании-работодателе
employee_count_dict = {
    'НЕТ ДАННЫХ': 0, 'ДО 10': 1, 'ОТ 11 ДО 30': 2, 'ОТ 11 ДО 50': 3,
    'ОТ 31 ДО 50': 4, 'ОТ 51 ДО 100': 5, 'ОТ 101 ДО 500': 6,
    'ОТ 501 ДО 1000': 7, 'БОЛЕЕ 500': 8, 'БОЛЕЕ 1001': 9
    }

# заполнение пропущенных данных и кодирование признака
clients['employee_count_nm'] = clients['employee_count_nm'].fillna('НЕТ ДАННЫХ')
clients['employee_count_nm'] = clients['employee_count_nm'].map(employee_count_dict)

# one-hot encoding
clients = pd.get_dummies(
    data=clients,
    columns=['customer_age', 'employee_count_nm'],
    dtype=int,
    )

# датафрейм с периодами сбора данных пользователя
period_info = clients[['user_id', 'report_dt']].copy()
period_info['transaction_mnth'] = period_info['report_dt'].apply(
    lambda x: pd.period_range(
        start=x - pd.DateOffset(months=8),
        end=x - pd.DateOffset(months=3),
        freq='M',
        ),
    )

period_info.drop(columns='report_dt', inplace=True)

# создание тренировочной и тестовой выборок
train_users = train['user_id'].unique()
test_users = np.setdiff1d(
    transactions['user_id'].unique(), train['user_id'].unique()
    )

train_data = transactions[transactions['user_id'].isin(train_users)].copy()
test_data = transactions[transactions['user_id'].isin(test_users)].copy()
del transactions

# инициализация датасета
dataset = Dataset(standardize=False)

_, x_train, t_train, e_train = dataset.build(
    transactions=train_data,
    client_info=clients,
    period_info=period_info,
    targets=train,
    fit_params=True,
    sample='train',
    )

ids_test, x_test = dataset.build(
    transactions=test_data,
    client_info=clients,
    period_info=period_info,
    sample='test',
    )

# инициализация модели и её обучение
model = DeepRecurrentSurvivalMachines(k=4, hidden=64, layers=3)
model.fit(x_train, t_train, e_train, iters=7, learning_rate=1e-3)

# сохранение предсказаний в датафрейм
submission = pd.DataFrame({
    'user_id': ids_test,
    'predict': model.predict_survival(x_test, 91).reshape(-1,6)[:, -1]
    })

submission.to_csv('.../submission.csv', index=False)

