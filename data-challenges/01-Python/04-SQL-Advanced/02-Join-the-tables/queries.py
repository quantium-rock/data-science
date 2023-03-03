# pylint:disable=C0111,C0103

def detailed_orders(db):
    '''return a list of all orders (order_id, customer.contact_name,
    employee.firstname) ordered by order_id'''
    query = """ SELECT Orders.OrderID, Customers.ContactName,Employees.FirstName FROM Orders
                JOIN Customers ON Customers.CustomerID = Orders.CustomerID 
                JOIN Employees ON Employees.EmployeeID = Orders.EmployeeID
                ORDER BY Orders.OrderID ASC """
    return db.execute(query).fetchall()


def spent_per_customer(db):
    '''return the total amount spent per customer ordered by ascending total
    amount (to 2 decimal places)
    Exemple :
        Jean   |   100
        Marc   |   110
        Simon  |   432
        ...
    '''
    query = """ SELECT Customers.ContactName,
                ROUND(SUM(OrderDetails.UnitPrice*OrderDetails.Quantity),2) AS Spent FROM Customers
                JOIN Orders ON Orders.CustomerID = Customers.CustomerID
                JOIN OrderDetails ON OrderDetails.OrderID = Orders.OrderID
                GROUP BY Orders.CustomerID
                ORDER BY Spent ASC """
    return db.execute(query).fetchall()


def best_employee(db):
    '''Implement the best_employee method to determine who’s
    the best employee! By “best employee”, we mean the one who sells the most.
    We expect the function to return a tuple like:
    ('FirstName', 'LastName', 6000 (the sum of all purchase)).
    The order of the information is irrelevant'''
    query = """ SELECT Employees.FirstName, Employees.LastName,
                ROUND(SUM(OrderDetails.UnitPrice*OrderDetails.Quantity),2) AS Sells FROM Employees 
                JOIN Orders ON Orders.EmployeeID = Employees.EmployeeID
                JOIN OrderDetails ON OrderDetails.OrderID = Orders.OrderID
                GROUP BY Orders.EmployeeID 
                ORDER BY Sells DESC
                LIMIT 1 """
    return db.execute(query).fetchone()


def orders_per_customer(db):
    '''TO DO: return a list of tuples where each tupe contains the contactName
    of the customer and the number of orders they made (contactName,
    number_of_orders). Order the list by ascending number of orders'''
    query = """ SELECT Customers.ContactName, COUNT(Orders.CustomerID) AS n FROM Customers
                LEFT JOIN Orders ON Customers.CustomerID = Orders.CustomerID 
                GROUP BY Orders.CustomerID
                ORDER BY n ASC """
    return db.execute(query).fetchall()
