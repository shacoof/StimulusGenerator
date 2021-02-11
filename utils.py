import csv

def loadCSV(csvFileName):
    with open(csvFileName, mode='r') as infile:
        reader = csv.DictReader(infile)
        reasult=[]
        for row in reader: 
            reasult.append(row)
        #print({rows[0]:rows[1] for rows in reader})
        return reasult