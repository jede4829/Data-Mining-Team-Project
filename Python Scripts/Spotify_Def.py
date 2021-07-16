
# ----------------------------------------------------------------#
# Import Modules
# ----------------------------------------------------------------#

import json

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