from functions import *
import pandas as pd
import os
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
# reading csv file
df = pd.read_csv("FSHS.csv")
x = processCSV(df)
while True:
    print("1.Show Data in Each Region, Sector Yearly")
    print("2. Show Data of Total Students by Sector")
    print("3. Show Overall Students in Each Region")
    print("4. Export Aggregated Data each Region by Strand")
    print("5. Export Overall Aggregated Data by Strand")
    print("6. Export Aggregated Data Count of Overall Students by Strand")
    print("0. Exit")
    choice = int(input("Ano gusto mo matabo: "))
    match choice:
        case 1:
            totalbyStrandEachRegion(x)
        case 2:
            totalStudentsbySector(x)
        case 3:
            totalStudentsbyRegion(x)
        case 4:
            aggregateregion(x)
        case 5:
            aggregateall(x)
        case 6:
            agg20162021(x)
        case 0:
            print("Exiting...")
            break
        case _:
            print("Bobo kaba.")
    os.system('cls')

