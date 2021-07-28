
# ----------------------------------------------------------------#
# Import Modules
# ----------------------------------------------------------------#

import json
import pandas as pd
import xlsxwriter

# ----------------------------------------------------------------#
# Class Printer
# ----------------------------------------------------------------#

def print_class(instance):
    keys = []
    class_var = instance.__dict__
    for k in class_var:
        keys.append(k)
    for j in keys:
        if type(class_var) is not str: 
            print(j + ' :    ' + str(class_var[j]))
        else:
            print(j + ' :    ' + class_var[j])
    return None

# ----------------------------------------------------------------#
# Print JSON Response from Spotify API
# ----------------------------------------------------------------#

def print_json(object):
    print(json.dumps(object, sort_keys = True, indent = 4))
    return None

# ----------------------------------------------------------------#
# Print New Line
# ----------------------------------------------------------------#

def new_line():
    print('\n')

# ----------------------------------------------------------------#
# Export Dataframe to Microsoft Excel
# ----------------------------------------------------------------#

def export_file(df_conv, filename, extension):
    outputFile = filename + '.' + extension
    writer = pd.ExcelWriter(outputFile, engine = 'xlsxwriter')
    df_conv.to_excel(writer, 'Top 10')
    workbook = writer.book
    worksheet = writer.sheets["Top 10"]
    worksheet.set_column('A:Z', 18)
    writer.save()

# ----------------------------------------------------------------#
# Read CSV File into DataFrame
# ----------------------------------------------------------------#

def read_file(csv_file):
    return pd.read_csv(csv_file)


