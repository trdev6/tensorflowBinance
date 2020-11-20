import json
import investpy
import pandas as pd


def dataHistoricalCripto(crypto, from_date, to_date):
    df = investpy.get_crypto_historical_data(crypto=crypto, as_json=True, from_date=from_date, to_date=to_date)

    result_json = df
    result_decoded_json = json.loads(result_json)['historical']
    resultado = json.dumps(result_decoded_json)

    out_file = open("consulta_historica_archivo.json", "w")
    out_file.writelines(str(resultado))
    out_file.close()
    df = pd.read_json(r'consulta_historica_archivo.json')
    # export_csv = df.to_csv(r'historico_' + crypto + '.csv', index=None, header=True)
    salida = df

    return salida


def dataHistoricalCurrencyCross(currency_cross, from_date, to_date):
    df = investpy.get_currency_cross_historical_data(currency_cross=currency_cross, as_json=True, from_date=from_date,
                                                     to_date=to_date)
    # print(df)

    result_json = df
    result_decoded_json = json.loads(result_json)['historical']
    resultado = json.dumps(result_decoded_json)

    out_file = open("consulta_historica_archivo.json", "w")
    out_file.writelines(str(resultado))
    out_file.close()
    df = pd.read_json(r'consulta_historica_archivo.json')
    # export_csv = df.to_csv(r'historico_moneda.csv', index=None, header=True)
    salida = df

    return salida
