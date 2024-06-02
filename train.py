from packet import *
from dataset import *


# def trainClinetModel(args, groups):
#     i = 0 #group number
#     j = 0 #client number
#     for group in groups:
#         i += 1
#         for client in group:
#             j += 1
#             client.train(i, j)

def trainClinetModel(args, group):
    i = 0 #group number
    # print('group[0]', len(group[0]), len(group[1]))
    print('group[0]', len(group[0]))
    for client in group[0]:
        client.train(i)
        i += 1
