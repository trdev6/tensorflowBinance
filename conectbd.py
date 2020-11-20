import sqlite3

from sqlite3 import Error

db_file = "Criptomoneda.db"


def create_connection():
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def InsertData(conn, task):
    # Ejemplo:
    # task = (i, symbol, baseAsset, quoteAsset, status)
    # conectbd.InsertData(conn=conectbd.create_connection(), task=task)
    """
    Create a new task
    :param conn:
    :param task:
    :return:
    """

    sql = ''' INSERT INTO criptomonedas(id,symbol,baseAsset,quoteAsset,status)
              VALUES(?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, task)
    conn.commit()


def InsertDataTensorflow(conn, task):
    # Ejemplo:
    # task = (i, symbol,update_model,last_update,params,mean_absolute_error,priceMin,
    #     priceMax,priceClose)
    # conectbd.InsertData(conn=conectbd.create_connection(), task=task)
    """
    Create a new task
    :param conn:
    :param task:
    :return:
    """

    sql = ''' INSERT INTO tensorflow(symbol,update_model,last_update,epoch,params,precision,mae,priceMin,
    priceMax,priceClose)
              VALUES(?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, task)
    conn.commit()


def updateData(conn, task):
    # updateData(create_connection(), (20000.200, "BTCUSDT"))
    """
    update priority, begin_date, and end date of a task
    :param conn:
    :param task:
    :return: project id
    """
    sql = ''' UPDATE criptomonedas
              SET price = ?
              WHERE symbol = ?'''
    cur = conn.cursor()
    cur.execute(sql, task)
    conn.commit()


def selectAll(conn, params):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute(f"SELECT {params} FROM criptomonedas")

    # rows = cur.fetchall()
    rows = [r[0] for r in cur]
    # for row in rows:
    #     print(row[0])
    return rows

# selectAll(create_connection(), "symbol")
