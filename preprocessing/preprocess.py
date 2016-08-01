# coding:utf-8
import numpy as np
from TACHIBANA.shogi_ban import GameState
from TACHIBANA.preprocessing.categorize import category

import copy
import os
from multiprocessing import Pool

C = category()

def load_kif(filename):
	#hands = np.load("/home/kento/TACHIBANA_project/kif_npy/"+filename)
	hands = np.load("../kif_npy/"+filename)
	#print(hands)

	return hands

def make_sente_datasets(filenames):
	x_sente_dataset = []
	y_sente_dataset = []
	error_list = []
	for filename in filenames:
		try:
			#print(filename)
			hands = load_kif(filename)
			print("read " +filename + "!!")
			#print(hands)
		except:
			print("{} is failed!!".format(filename))
			continue

		state = GameState()
		for n in hands:
			next_hand = C.del_turns(n)
			current_board = copy.copy(state.board)
			if state.is_end_game:break
			if state.next_player > 0:
				try:
					y_sente_dataset.append(C.get_index(1,next_hand))
					x_sente_dataset.append(current_board)
				except:
					error_list.append(filename)
					print("######## kif_{}.npy has any problem!! #########".format(filename))
					break
			state.update_board(n)

	with open("error_list.txt", "w") as f:
		for x in error_list:
			f.write(str(x) + "\n")

	return x_sente_dataset, y_sente_dataset

def make_gote_datasets(filenames):
	x_gote_dataset = []
	y_gote_dataset = []
	gote_error_list = []
	for filename in filenames:
		try:
			hands = load_kif(filename)
			print("read"+filename+"!!")
			#print(hands)
		except:
			print("{} is failed!!".format(filename))
			continue

		state = GameState()
		for n in hands:
			next_hand = C.del_turns(n)
			current_board = copy.copy(state.board)
			if state.is_end_game:break
			if state.next_player < 0:
				try:
					# 後手のinfoを取ってきたい時は引数に−1を選択
					y_gote_dataset.append(C.get_index(-1,next_hand))
					x_gote_dataset.append(current_board)
				except:
					gote_error_list.append(filename)
					print("######## kif_{}.npy has any problem!! #########".format(filename))
					break
			state.update_board(n)

	return x_gote_dataset, y_gote_dataset

def get_n_filenames(n):
	#name_list = os.listdir("../kif_npy")
	name_list = os.listdir("/home/kosuda/work/wars_kif/")
	return name_list[0:n]

def split_namelist(core, n):
	name_list = get_n_filenames(n)
	size = int(len(name_list)/core)
	split_data = [name_list[x:x + size] for x in range(0, len(name_list), size)]
	return split_data[0:-1]

def make_dataset_with_multiprocess(player,core,split_data):
	results_list = [None for row in range(core)]
	pool = Pool(processes=core)
	if player==1:
		results_list = pool.map(make_sente_datasets, split_data)
	else:
		results_list = pool.map(make_gote_datasets, split_data)

	x_dataset = [results_list[i][0] for i in range(len(results_list))]
	y_dataset = [results_list[i][1] for i in range(len(results_list))]

	x_dataset = [item for sublist in x_dataset for item in sublist]
	y_dataset = [item for sublist in y_dataset for item in sublist]

	return x_dataset, y_dataset

def load_kif_one_core(i):
    hands = np.load("../kif_npy/kif_{}.npy".format(i))
    #hands = np.load("/home/kento/TACHIBANA_project/kif_npy/kif_{}.npy".format(i))
    return hands

def make_sente_datasets_one_core(m,n):
    x_sente_dataset = []
    y_sente_dataset = []
    error_list = []
    for i in range(m,n):
        try:
            hands = load_kif_one_core(i)
            print("read kif_{}.npy!!".format(i))
        except:
            print("kif_{}.npy is failed!!".format(i))
            continue

        state = GameState()
        for n in hands:
            next_hand = C.del_turns(n)
            current_board = copy.copy(state.board)
            if state.is_end_game:break
            if state.next_player > 0:
                try:
                    y_sente_dataset.append(C.get_index(1,next_hand))
                    x_sente_dataset.append(current_board)
                except:
                    #error_list.append(i)
                    print("######## kif_{}.npy has any problem!! #########".format(i))
                    break
            state.update_board(n)

    return x_sente_dataset, y_sente_dataset


if __name__ == "__main__":

	x_dataset, y_dataset = make_gote_datasets_for_dense(0,10)

	x_dataset = np.asarray(x_dataset)
	y_dataset = np.asarray(y_dataset)

	nb_data = x_dataset.shape[0]

	x_train,x_test = np.split(x_dataset,[nb_data*0.9])
	y_train,y_test = np.split(y_dataset,[nb_data*0.9])

	x_train = x_train.reshape(x_train.shape[0], 1, 15, 9)
	x_test = x_test.reshape(x_test.shape[0], 1, 15, 9)

	print(len(y_train[0]))
	"""
	a, b=make_sente_datasets(1,3)
	print(len(a),len(b))
	for i,j  in zip(a,b):
		print(i,j)
	print(C.gote_category[j])

	for i,j,k in zip(range(len(a)), a, b):
		if i%2 == 0:
			x_sente_dataset.append(j)
			y_sente_dataset.append(k)
	print(x_sente_dataset[len(x_sente_dataset)-1])
	print(y_sente_dataset[len(x_sente_dataset)-1])
	"""
