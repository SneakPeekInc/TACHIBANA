import numpy as np
## input example
"""
5 2 4
100
90
82
70
65
"""

def get_info():
    info = input().split():
    info = map(int,info)
    return info[0], info[1],info[2]

def get_score(n):
    score_list = list()
    for i in range(n):
        score_list.append(int(input()))
    return score_list

def get_diff(n,score_list):
    return score_list[n-1]-score_list[n]


def main():
    num, Min, Max = get_info
    score_list = get_score(num)
    diff_ls = list()
    for i in range(Min, Max+1):
        diff_ls.append(get_diff(i,score_list))
    print(diff_ls)
    border = np.argmax(diff_ls) + Min

main()
