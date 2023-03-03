# pylint:disable=C0111,C0103

def query_orders(db):
    # return a list of orders displaying each column
    return db.execute('SELECT * FROM orders ORDER BY OrderID').fetchall()


def get_orders_range(db, date_from, date_to):
    # return a list of orders displaying all columns with OrderDate between
    # date_from and date_to (excluding date_from and including date_to)
    query = "SELECT * FROM orders WHERE OrderDate > ? AND  OrderDate < ?"
    return db.execute(query, [date_from, date_to]).fetchall()


def get_waiting_time(db):
    # get a list with all the orders displaying each column
    # and calculate an extra TimeDelta column displaying the number of days
    # between OrderDate and ShippedDate, ordered by ascending TimeDelta¡
    query = """ SELECT *, JULIANDAY(ShippedDate)-JULIANDAY(OrderDate) AS TimeDelta
                FROM orders ORDER BY TimeDelta ASC """
    return db.execute(query).fetchall()
