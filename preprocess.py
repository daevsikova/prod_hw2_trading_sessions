import pandas as pd
import sqlite3
import numpy as np


def get_raw_data(path='data/trade_info.sqlite3'):
    # reading data from database
    con = sqlite3.connect(path)
    data =  pd.read_sql(
        """
        SELECT * FROM Chart_data C
        JOIN Trading_session T ON C.session_id=T.id
        WHERE T.trading_type = 'monthly'
        """,
        con
    )

    return data


def calc_weighted_mean(data):
    data['revenue'] = data['lot_size'] * data['price']

    weighted_mean_by_session = (
        data
        .groupby('session_id')
        .agg({'revenue': 'sum', 'lot_size': 'sum'})
        .apply(lambda x: x['revenue'] / x['lot_size'], axis=1)
    )

    weighted_mean_by_session = pd.DataFrame(weighted_mean_by_session).reset_index()
    weighted_mean_by_session.columns = ['session_id', 'weighted_price']
    weighted_mean_by_session['session_id'] += 1

    return weighted_mean_by_session, data


def time_preparation(data):
    data['time'] = pd.to_datetime(data.time)
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute

    # Определим сессии, где операции вышли за диапазон 60 минут
    hour_diff_per_sess = pd.DataFrame(
        data
        .groupby('session_id')
        .agg({'hour': ['min', 'max']})['hour']
    ).reset_index().assign(hours_diff=lambda x: x['max'] - x['min'], axis=1)

    hour_diff_per_sess = hour_diff_per_sess[['session_id', 'hours_diff']]

    # Заменяем время для операций, которые относятся к данной торговой сессии,
    # но вышли за пределы 60 минут. Для них заменяем час на `hour - 1`, 
    # минуты на `59`
    temp = pd.merge(data, hour_diff_per_sess, on='session_id', how='left')

    mask = ((temp.hours_diff == 1) & 
            (temp.platform_id == 1) & 
            (temp.hour == 12) & 
            ~(temp.session_id == 53)).values
    data.loc[mask, 'hour'] = 11
    data.loc[mask, 'minute'] = 59

    mask = ((temp.hours_diff == 1) & (temp.hour == 13)).values
    data.loc[mask, 'hour'] = 12
    data.loc[mask, 'minute'] = 59

    mask = ((temp.hours_diff == 1) & 
            (temp.hour == 12) & 
            (temp.session_id == 54)).values
    data.loc[mask, 'hour'] = 11
    data.loc[mask, 'minute'] = 59

    return data


def explode_minutes(data_sessions):
    session_hour = pd.DataFrame(
        np.unique(data_sessions[['session_id', 'hour']].values, axis=0),
        columns = ['session_id', 'hour']
    )

    session_hour['minute'] = [[i for i in range(60)]] * len(session_hour)
    session_hour = session_hour.explode('minute')

    data_sessions = pd.merge(session_hour, data_sessions, how='left', on=['session_id', 'hour', 'minute'])
    
    return data_sessions


def preprocessing(path='data/trade_info.sqlite3'):
    data = get_raw_data(path)

    # Удаляем дубликаты с одинаковыми deal_id
    data = data[(data.groupby('deal_id').cumcount() == 0)]

    # Считаем взвешенное среднее для предыдущих торговых сессий, 
    # чтобы затем заполнить пропуски этими данными
    weighted_mean_by_session, data = calc_weighted_mean(data)

    # Определяем час и минуту для каждой операции
    data = time_preparation(data)

    # Считаем взвешенную среднюю цену для каждой минуты внутри сессии,
    # чтобы получить датасет с ровно 60 наблюдениями на 1 торговую сессию
    data_sessions = (
        data
        .groupby(['session_id', 'hour', 'minute'])
        .agg({'revenue': 'sum', 'lot_size': 'sum', 'time': 'min'})
        .reset_index()
        .assign(weighted_price = lambda x: x['revenue'] / x['lot_size'])
    )

    # Добавляем исходную информацию
    data_sessions = pd.merge(
        data_sessions, 
        data[['session_id', 'hour', 'minute', 'date', 'platform_id']], 
        how='left', 
        on=['session_id', 'hour', 'minute']
    ).drop_duplicates()

    # Добавляем недостающие данные по минутам в датафрейм
    data_sessions = explode_minutes(data_sessions)

    # Заполняем начальную точку в торговых сессиях с помощью значения средневзвешенной цены
    # для предыдущей сессии
    data_sessions = pd.merge(data_sessions, weighted_mean_by_session, how='left', on=['session_id'])

    data_sessions['weighted_price'] = np.where(
        (data_sessions.minute == 0) & (data_sessions.weighted_price_x.isna()), 
        data_sessions.weighted_price_y,
        data_sessions.weighted_price_x
    )

    data_sessions = data_sessions.drop(['weighted_price_y', 'weighted_price_x'], axis=1)

    # Для неизвестных предыдущих сессий заполняем цену значением 0
    data_sessions['weighted_price'] = np.where(
        (data_sessions.minute == 0) & (data_sessions.weighted_price.isna()),
        0,
        data_sessions.weighted_price
    )

    # Заполняем пропуски
    data_sessions['weighted_price'] = data_sessions.weighted_price.ffill()

    return data_sessions
