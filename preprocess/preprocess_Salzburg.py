import pandas as pd
import gzip
import struct
import os
import csv



current_dir = os.getcwd()
os.chdir(current_dir)


def SetRawValues(row, dictwriter, n):
    t = int(row['Offset'])
    data = bytes.fromhex(row["rawdata"][2:])  # deserialize hex string to bytes
    for i in range(int(len(data) / 4)):
        if data[i * 4] == 0 and data[i * 4 + 1] == 0 and data[i * 4 + 2] == 0 and data[i * 4 + 3] == 0:
            continue  # no null values
        n = n + 1  # new primary key
        newrow = row.copy()
        if newrow["DataID"] != '703':
            continue
        del newrow["DataID"]
        del newrow["rawdata"]  # not needed
        del newrow["cnt"]  # not needed
        del newrow["id"]  # not needed
        # newrow["id"] = n  # primary key
        newrow["Val"] = struct.unpack('<f', data[i * 4:i * 4 + 4])[0]  # bytes to float
        newrow["Offset"] = t + i * 60  # new offset
        dictwriter.writerow(newrow)
    return n


n = 0
with open('data_float_m_703.csv', 'w', newline='') as csvfile:
    dict_writer = csv.DictWriter(csvfile, ['id', 'CaseID', 'DataID', 'Offset', 'Val'])
    dict_writer = csv.DictWriter(csvfile, ['CaseID', 'Offset', 'Val'])
    dict_writer.writeheader()
# counter = 10000
    with gzip.open('data_float_h.csv.gz', 'rt') as gzf:
        for row in csv.DictReader(gzf):
#             # counter -= 1
#             # if counter == 0:
#             #     break
            n = SetRawValues(row, dict_writer, n)
            if (n % 1000 == 0):
                 print("Processing entry " + str(n))



