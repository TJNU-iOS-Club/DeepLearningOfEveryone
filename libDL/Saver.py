import csv

def save(data):
    with open('./result/result.csv', 'w') as csv_writer:
        writer = csv.writer(csv_writer)
        writer.writerows(data)
