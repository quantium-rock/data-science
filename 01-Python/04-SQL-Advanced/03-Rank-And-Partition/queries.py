# pylint:disable=C0111,C0103

def order_rank_per_customer(db):
    query = """ SELECT Orders.OrderID, Customers.CustomerID, Orders.OrderDate,
                RANK() OVER (
                    PARTITION BY Customers.CustomerID
                    ORDER BY Orders.OrderDate
                ) as OrderRank FROM Customers
                JOIN Orders ON Orders.CustomerID = Customers.CustomerID
                JOIN OrderDetails ON Orders.OrderID = OrderDetails.OrderID
                GROUP BY Orders.OrderID """
    return db.execute(query).fetchall()


def order_cumulative_amount_per_customer(db):
    query = """ SELECT Orders.OrderID, Orders.CustomerID, Orders.OrderDate,
                ROUND(SUM(SUM(OrderDetails.UnitPrice*OrderDetails.Quantity)) OVER (
                    PARTITION BY Orders.CustomerID
                    ORDER BY Orders.OrderDate
                ),2) AS Amount FROM Orders JOIN OrderDetails ON OrderDetails.OrderID = Orders.OrderID
                GROUP BY OrderDetails.OrderID """
    return db.execute(query).fetchall()
