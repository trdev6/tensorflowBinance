import time
import json
import hmac
import hashlib
import requests
import pandas as pd
import xlrd
from urllib.parse import urljoin, urlencode

ambiente = "PROD"

if ambiente == "DEV":
    API_KEY = 'L7xYEFFmaiBcqLEM1iCa5eofDW23v6rJoizDybYHcb6jav0Id1OAJeaiBSd6dfYk'
    SECRET_KEY = 'F2PYYMKmYYUYzwOwZYeRQOIT0b6TwhfzOnSD4m9lMciJjk5MkOynoUQ1wuKvzzYt'
    BASE_URL = 'https://testnet.binance.vision'
    headers = {
        'X-MBX-APIKEY': API_KEY
    }
if ambiente == "PROD":
    API_KEY = ''
    SECRET_KEY = ''
    BASE_URL = 'https://api.binance.com'
    headers = {
        'X-MBX-APIKEY': API_KEY
    }


class BinanceException(Exception):
    def __init__(self, status_code, data):

        self.status_code = status_code
        if data:
            self.code = data['code']
            self.msg = data['msg']
        else:
            self.code = None
            self.msg = None
        message = f"{status_code} [{self.code}] {self.msg}"

        # Python 2.x
        # super(BinanceException, self).__init__(message)
        super().__init__(message)


def Account():
    # Ejemplo: Account()
    PATH = '/api/v3/account'
    timestamp = int(time.time() * 1000)
    params = {'timestamp': timestamp}
    query_string = urlencode(params)
    params['signature'] = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    url = urljoin(BASE_URL, PATH)
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        response = json.dumps(response.json())
        # data = json.loads(response)
        return response
    else:
        raise BinanceException(status_code=response.status_code, data=response.json())


def MyTrades(symbol):
    # Ejemplo: MyTrades("ETHUSDT")
    PATH = '/api/v3/myTrades'
    timestamp = int(time.time() * 1000)
    params = {'symbol': symbol,
              'timestamp': timestamp}
    query_string = urlencode(params)
    params['signature'] = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    url = urljoin(BASE_URL, PATH)
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        response = json.dumps(response.json())
        # data = json.loads(response)
        return response
    else:
        raise BinanceException(status_code=response.status_code, data=response.json())


def GetOpenOrders(symbol):
    # Ejemplo: GetOpenOrders("")
    #          GetOpenOrders("ETHUSDT")
    PATH = '/api/v3/openOrders'
    timestamp = int(time.time() * 1000)
    if symbol:
        params = {'symbol': f'{symbol}',
                  'timestamp': timestamp
                  }
    else:
        params = {'timestamp': timestamp
                  }
    query_string = urlencode(params)
    params['signature'] = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    url = urljoin(BASE_URL, PATH)
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        response = json.dumps(response.json())
        # data = json.loads(response)
        return response
    else:
        raise BinanceException(status_code=response.status_code, data=response.json())


def AllOrders(symbol):
    # Ejemplo: AllOrders("ETHUSDT")
    PATH = '/api/v3/allOrders'
    timestamp = int(time.time() * 1000)
    params = {'symbol': f'{symbol}',
              'timestamp': timestamp
              }
    query_string = urlencode(params)
    params['signature'] = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    url = urljoin(BASE_URL, PATH)
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        response = json.dumps(response.json())
        data = json.loads(response)
        return data
    else:
        raise BinanceException(status_code=response.status_code, data=response.json())


def CreateOrder(symbol, side, quantity, price):
    # Ejemplo: # CreateOrder("ETHUSDT", "SELL", 0.1, 461)
    #            CreateOrder("ETHUSDT", "BUY", 0.1, 461)
    PATH = '/api/v3/order'
    timestamp = int(time.time() * 1000)
    params = {
        'symbol': symbol,
        'side': side,
        'type': 'LIMIT',
        'timeInForce': 'GTC',
        'quantity': float(quantity),
        'price': float(price),
        'recvWindow': 5000,
        'timestamp': timestamp
    }

    query_string = urlencode(params)
    params['signature'] = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    url = urljoin(BASE_URL, PATH)
    response = requests.post(url, headers=headers, params=params)
    if response.status_code == 200:
        # response = json.dumps(response.json())
        # data = json.loads(response)
        return response
    else:
        raise BinanceException(status_code=response.status_code, data=response.json())


def DeleteOrder(symbol, orderId):
    # Ejemplo: DeleteOrder("ETHUSDT", "725")
    PATH = '/api/v3/order'
    timestamp = int(time.time() * 1000)

    params = {
        'symbol': symbol,
        'orderId': orderId,
        'recvWindow': 5000,
        'timestamp': timestamp
    }

    query_string = urlencode(params)
    params['signature'] = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    url = urljoin(BASE_URL, PATH)
    response = requests.delete(url, headers=headers, params=params)
    if response.status_code == 200:
        response = json.dumps(response.json())
        # data = json.loads(response)
        return response
    else:
        raise BinanceException(status_code=response.status_code, data=response.json())


def AvgPrice(symbol):
    # Ejemplo: AvgPrice("ETHUSDT")
    binance = "https://api.binance.com"
    avg_price = "/api/v3/avgPrice"
    response = requests.get(binance + avg_price + "?symbol=" + symbol)
    response = json.dumps(response.json())
    data = json.loads(response)
    return data


def GetPrice(symbol):
    # Ejemplo: getPrice("ETHUSDT")
    PATH = "/api/v3/ticker/price"
    if symbol:
        response = requests.get(BASE_URL + PATH + "?symbol=" + symbol)
    else:
        response = requests.get(BASE_URL + PATH)
    response = json.dumps(response.json())
    data = json.loads(response)
    return data


def Exchangeinfo():
    # Ejemplo: getPrice("ETHUSDT")
    PATH = "/api/v3/exchangeInfo"
    response = requests.get(BASE_URL + PATH)
    response = json.dumps(response.json())
    # data = json.loads(response)
    return response


# def Getklines(symbol, interval, startTime, endTime):
def Getklines(symbol, interval):
    # Ejemplo: Getklines("ETHUSDT", "5m")
    #    1499040000000,      // Open time
    #    "0.01634790",       // Open
    #    "0.80000000",       // High
    #    "0.01575800",       // Low
    #    "0.01577100",       // Close
    #    "148976.11427815",  // Volume
    #    1499644799999,      // Close time
    #    "2434.19055334",    // Quote asset volume
    #    308,                // Number of trades
    #    "1756.87402397",    // Taker buy base asset volume
    #    "28.46694368",      // Taker buy quote asset volume
    #    "17928899.62484339" // Ignore.
    PATH = '/api/v3/klines'
    timestamp = int(time.time() * 1000)
    hora_inicio = timestamp - (86400 * 1000)
    hora_fin = timestamp

    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': int(hora_inicio),
        'endTime': int(hora_fin)
    }

    url = urljoin(BASE_URL, PATH)
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        response = json.dumps(response.json())
        data = json.loads(response)
        return data
    else:
        raise BinanceException(status_code=response.status_code, data=response.json())


def ConvertToXsl(data, archivo):
    # Ejemplo: ConvertToXsl(data, "ETHUSD")
    dataConvert = json.dumps(data)
    out_file = open(f'{archivo}.json', "w")
    out_file.writelines(str(data))
    out_file.close()

    df = pd.read_json(fr'{archivo}.json')
    export_csv = df.to_csv(fr'{archivo}.csv', index=None, header=True)
    df = pd.read_csv(fr'{archivo}.csv')
    export_xsl = df.to_excel(fr'{archivo}.xlsx', index=None, header=True)


def ConvertXslToJson(archivo):
    # Ejemplo: binance.ConvertXslToJson("Criptomonedas")
    df = pd.read_excel(fr'{archivo}.xlsx')
    export_json = df.to_json(f'{archivo}.json')
    # out_file = open(f'{archivo}.json', "w")
    # out_file.writelines(str(export_json))
    # out_file.close()
    print(export_json)
