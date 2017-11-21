from function import F

file_name = 'data.txt'
x0 = eps = 0

with open(file_name) as f:
    x0, eps = [float(x) for x in f.readline().split(' ')]  # read first line

