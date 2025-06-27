# colombianhearmap
Mapa de calor Homicidios en Colombia

## Predicción del precio del dólar

Se añadió el script `usd_prediction.py` que descarga datos históricos de la tasa de cambio USD/EUR desde la API pública de la Reserva Federal (FRED) y realiza pronósticos de los próximos 7 días mediante dos métodos populares:

1. ARIMA (implementado con `statsmodels`).
2. Prophet (implementado con `prophet`).

### Requisitos

```bash
pip install pandas numpy statsmodels prophet requests
```

### Uso

Ejecutar el script directamente:

```bash
python usd_prediction.py
```

El programa mostrará los últimos datos descargados y los pronósticos generados por ambos métodos.
