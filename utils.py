import csv

def loadCSV(csvFileName):
    with open(csvFileName, mode='r') as infile:
        reader = csv.DictReader(infile)
        result=[]
        for row in reader: 
            result.append(row)
        #print({rows[0]:rows[1] for rows in reader})
        return result

def writeCSV(csvFileName,dict):
    with open(csvFileName, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dict.keys())
        writer.writeheader()
        writer.writerow(dict)
