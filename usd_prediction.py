import pandas as pd
import requests
from io import StringIO
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


def download_exchange_rate(series_id: str = "DEXUSEU") -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = pd.read_csv(StringIO(resp.text))
    data['observation_date'] = pd.to_datetime(data['observation_date'])
    data.rename(columns={'observation_date': 'ds', series_id: 'y'}, inplace=True)
    data.dropna(inplace=True)
    return data


def predict_arima(df: pd.DataFrame, steps: int = 7) -> pd.Series:
    model = ARIMA(df['y'], order=(5, 1, 0))
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    return forecast


def predict_prophet(df: pd.DataFrame, periods: int = 7) -> pd.DataFrame:
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(periods)


def main():
    df = download_exchange_rate()
    print("Datos cargados:\n", df.tail())

    print("\nPronóstico ARIMA (siguientes 7 días):")
    arima_forecast = predict_arima(df)
    print(arima_forecast)

    print("\nPronóstico Prophet (siguientes 7 días):")
    prophet_forecast = predict_prophet(df)
    print(prophet_forecast)


if __name__ == "__main__":
    main()
