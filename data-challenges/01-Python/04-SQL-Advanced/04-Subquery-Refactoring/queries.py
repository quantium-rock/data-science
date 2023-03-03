# pylint:disable=C0111,C0103

def get_average_purchase(db):
    # return the average amount spent per order for each customer ordered by customer ID
    # $CHALLENGIFY_BEGIN
    request = '''
        WITH OrderValues AS (
          SELECT
            SUM(od.UnitPrice * od.Quantity) AS value,
            od.OrderID
          FROM OrderDetails od
          GROUP BY od.OrderID
        )
        SELECT
            c.CustomerID,
            ROUND(AVG(ov.value), 2) AS average
        FROM Customers c
        JOIN Orders o ON c.CustomerID = o.CustomerID
        JOIN OrderValues ov ON ov.OrderID = o.OrderID
        GROUP BY c.CustomerID
        ORDER BY c.CustomerID
    '''
    return db.execute(request).fetchall()
    # $CHALLENGIFY_END

def get_general_avg_order(db):
    # return the average amount spent per order
    # $CHALLENGIFY_BEGIN
    request = '''
        WITH OrderValues AS (
          SELECT SUM(od.Quantity * od.UnitPrice) AS value
          FROM OrderDetails od
          GROUP BY od.OrderID
        )
        SELECT ROUND(AVG(ov.value), 2)
        FROM OrderValues ov
    '''
    return db.execute(request).fetchone()[0]
    # $CHALLENGIFY_END

def best_customers(db):
    # return the customers who have an average purchase greater than the general average purchase
    # $CHALLENGIFY_BEGIN
    request = '''
        WITH OrderValues AS (
          SELECT
            SUM(od.UnitPrice * od.Quantity) AS value,
            od.OrderID
          FROM OrderDetails od
          GROUP BY od.OrderID
        ),
        GeneralOrderValue AS (
          SELECT ROUND(AVG(ov.value), 2) AS average
          FROM OrderValues ov
        ),
        CustomerOrderValue AS (
          SELECT
            c.CustomerID,
            ROUND(AVG(ov.value), 2) AS average
          FROM Customers c
          JOIN Orders o ON c.CustomerID = o.CustomerID
          JOIN OrderValues ov ON ov.OrderID = o.OrderID
          GROUP BY c.CustomerID
          ORDER BY c.CustomerID
        )
        SELECT
            CustomerOrderValue.CustomerID,
            CustomerOrderValue.average
        FROM CustomerOrderValue
        WHERE CustomerOrderValue.average > (SELECT average FROM GeneralOrderValue)
        ORDER BY CustomerOrderValue.average DESC
    '''
    return db.execute(request).fetchall()
    # $CHALLENGIFY_END

def top_ordered_product_per_customer(db):
    # return the list of the top ordered product
    # by each customer based on the total ordered amount in USD
    # $CHALLENGIFY_BEGIN
    query = """
        WITH OrderedProducts AS (
            SELECT
                CustomerID,
                ProductID, SUM(OrderDetails.Quantity * OrderDetails.UnitPrice) AS ProductValue
            FROM OrderDetails
            JOIN Orders ON OrderDetails.OrderID = Orders.OrderID
            GROUP BY Orders.CustomerID, OrderDetails.ProductID
            ORDER BY ProductValue DESC
        )
        SELECT
            CustomerID,
            ProductID,
            MAX(ProductValue) AS TopProductValue
        FROM OrderedProducts
        GROUP BY CustomerID
        ORDER BY TopProductValue DESC
    """
    return db.execute(query).fetchall()
    # $CHALLENGIFY_END

def average_number_of_days_between_orders(db):
    # return the average number of days between two consecutive orders of the same customer
    # $CHALLENGIFY_BEGIN
    query = """
        WITH DatedOrders AS (
            SELECT
                CustomerID,
                OrderID,
                OrderDate,
                LAG(OrderDate, 1, 0) OVER (
                    PARTITION BY CustomerID
                    ORDER By OrderDate
                ) PreviousOrderDate
            FROM Orders
        )
        SELECT ROUND(AVG(JULIANDAY(OrderDate) - JULIANDAY(PreviousOrderDate))) AS delta
        FROM DatedOrders
        WHERE PreviousOrderDate != 0
    """
    return int(db.execute(query).fetchone()[0])
    # $CHALLENGIFY_END
