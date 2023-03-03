
import csv

with open('data/phone_book.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    print(type(csv_reader))
    #next(csv_reader, None)
    for row in csv_reader:
        print(f"{row['last_name']}: {row['phone_number']}")
