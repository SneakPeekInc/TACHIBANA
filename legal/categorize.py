# coding:utf-8
import numpy as np
from sente_shogi_legal import SenteShogi
from gote_shogi_legal import GoteShogi

class category():
	def __init__(self):
		self.table = self.make_table()
		self.sente_category = self.make_sente_category()
		self.gote_category = self.make_gote_category()

		self.sente_category = self.refine_sente_category()
		self.gote_category = self.refine_gote_category()


		"""## FOR JUDGING LEGAL ##
		self.keima_legal_sente = [(2,1),(2,-1)]
		self.keima_legal_gote = [(-2,1),(-2,-1)]

		self.gin_legal_sente = [(1,1),(1,0),(1,-1),(-1,1),(-1,-1)]
		self.gi_legal_gote = [(-1,-1),(-1,0),(-1,1),(1,1),(1,-1)]

		self.kin_legal_sente = [(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,0)]
		self.kin_legal_gote = [(-1,-1),(-1,0),(-1,-1),(0,1),(0,-1),(1,0)]
		"""

	@staticmethod
	def del_turns(ls):
		ls = ls.tolist()
		return ls[:-1]

	@staticmethod
	def make_sente_category():
		# legalを考えない駒が先手のinfoを全て生成する
		a = [(x,y) for x in range(9) for y in range(9)]
		b = [(x,y) for x in range(9) for y in range(9)]
		b.append((-1,9))
		c = [i for i in range(1,15)]
		sente_category = [[x, y, z] for x in b for y in a for z in c]
		# 終局のinfo [(0,0),(0,0),0] を加える
		sente_category.append([(0,0),(0,0),0])
		return sente_category

	@staticmethod
	def make_gote_category():
		# legalを考えない駒が後手のinfoを全て生成する
		a = [(x,y) for x in range(9) for y in range(9)]
		b = [(x,y) for x in range(9) for y in range(9)]
		b.append((-1,9))
		c = [i for i in range(1,15)]
		gote_category = [[x, y, z] for x in b for y in a for z in c]
		# 終局のinfo [(0,0),(0,0),0] を加える
		gote_category.append([(0,0),(0,0),0])
		return gote_category

	def get_index(self,teban,ls):
		if teban > 0:
			return self.sente_category.index(ls)
		if teban < 0:
			return self.gote_category.index(ls)
	@staticmethod
	def make_table():
		## definition
		row = ["１","２","３","４","５","６","７","８","９"]
		col = ["一","二","三","四","五","六","七","八","九"]
		komaDict = {1:"歩",9:"と",
		            2:"香",10:"成香",
		            3:"桂",11:"成桂",
		            4:"銀",12:"成銀",
		            5:"角",13:"馬",
		            6:"飛",14:"竜",
		            7:"金",
		            8:"王",
		            0:"投了"}

		fugo = [str(i)+str(j) for i in row for j in col]
		matrix = [(i,j) for j in range(9) for i in range(9)]

		return matrix


	def refine_sente_category(self):
		## 先手から見て可能な動きをするInfoだけ残す。
		legal = SenteShogi()
		sente_category = self.sente_category
		refined_sente_category = []
		narigoma = [9,10,11,12]
		for info in sente_category:
			if info[2] == 0: #終局
					#print("終局入りましたー")
					#print(info)
					refined_sente_category.append(info)
			if info[2] == 1: #歩
				if legal.fu(info):
					refined_sente_category.append(info)
			elif info[2] == 2: #香車
				if legal.kyosya(info):
					refined_sente_category.append(info)
			elif info[2] == 3:
				if legal.keima(info):
					refined_sente_category.append(info)
			elif info[2] == 4:
				if legal.gin(info):
					refined_sente_category.append(info)
			elif info[2] == 5:
				if legal.kaku(info):
					refined_sente_category.append(info)
			elif info[2] == 6:
				if legal.hisya(info):
					refined_sente_category.append(info)
			elif info[2] == 7:
				if legal.kin(info):
					refined_sente_category.append(info)
			elif info[2] == 8:
				if legal.gyoku(info):
					refined_sente_category.append(info)
			elif info[2] == 9:
				if info[0] != (-1,9):
					if legal.kin(info):
						refined_sente_category.append(info)
			elif info[2] == 10:
				if info[0] != (-1,9):
					# 香車か金の動きしたらOK
					# ここでクラスが作成されるのはやむえない.
					if legal.kyosya(info) or legal.kin(info):
						refined_sente_category.append(info)
			elif info[2] == 11:
				if info[0] != (-1,9):
					# 桂馬か金の動きしたらOK
					if legal.keima(info) or legal.kin(info):
						refined_sente_category.append(info)
			elif info[2] == 12:
				if info[0] != (-1,9):
					# 銀と成銀の範囲を考えると玉と同じ
					if legal.gyoku(info):
						refined_sente_category.append(info)
			elif info[2] == 13:
				if legal.uma(info):
					refined_sente_category.append(info)
			elif info[2] == 14:
				if legal.ryu(info):
					refined_sente_category.append(info)

		self.sente_category = refined_sente_category
		return refined_sente_category

	def refine_gote_category(self):
		## 先手から見て可能な動きをするInfoだけ残す。
		legal = GoteShogi()
		gote_category = self.gote_category
		refined_gote_category = []
		narigoma = [9,10,11,12]
		for info in gote_category:
			if info[2] == 0: #終局
					#print("終局入りましたー")
					#print(info)
					refined_gote_category.append(info)
			if info[2] == 1: #歩
				if legal.fu(info):
					refined_gote_category.append(info)
			elif info[2] == 2: #香車
				if legal.kyosya(info):
					refined_gote_category.append(info)
			elif info[2] == 3:
				if legal.keima(info):
					refined_gote_category.append(info)
			elif info[2] == 4:
				if legal.gin(info):
					refined_gote_category.append(info)
			elif info[2] == 5:
				if legal.kaku(info):
					refined_gote_category.append(info)
			elif info[2] == 6:
				if legal.hisya(info):
					refined_gote_category.append(info)
			elif info[2] == 7:
				if legal.kin(info):
					refined_gote_category.append(info)
			elif info[2] == 8:
				if legal.gyoku(info):
					refined_gote_category.append(info)
			elif info[2] == 9:
				if info[0] != (-1,9):
					if legal.kin(info):
						refined_gote_category.append(info)
			elif info[2] == 10:
				if info[0] != (-1,9):
					# 香車か金の動きしたらOK
					# ここでクラスが作成されるのはやむえない.
					if legal.kyosya(info) or legal.kin(info):
						refined_gote_category.append(info)
			elif info[2] == 11:
				if info[0] != (-1,9):
					# 桂馬か金の動きしたらOK
					if legal.keima(info) or legal.kin(info):
						refined_gote_category.append(info)
			elif info[2] == 12:
				if info[0] != (-1,9):
					# 銀と成銀の範囲を考えると玉と同じ
					if legal.gyoku(info):
						refined_gote_category.append(info)
			elif info[2] == 13:
				if legal.uma(info):
					refined_gote_category.append(info)
			elif info[2] == 14:
				if legal.ryu(info):
					refined_gote_category.append(info)

		self.gote_category = refined_gote_category
		return refined_gote_category


if __name__ == "__main__":
	c = category()
	#print(c.sente_category[13])
	print(len(c.gote_category),"gote")
	print(len(c.sente_category),"sente")
	#print(c.get_index(1, [(6, 8), (5, 1), 1]))
	#print(c.get_index(-1, [(0, 0), (0, 0), 0]))

	print(c.gote_category[2138])

	print("sente 1st", c.sente_category[5947])
	print("sente 2nd", c.sente_category[6570])

	print("gote 1st", c.gote_category[2761])
	print("gote 2nd", c.gote_category[2138])
	print("gote 3rd", c.gote_category[2511])

	sente_cg = np.asarray(c.sente_category)
	gote_cg = np.asarray(c.gote_category)

	np.save("sente_category.npy", sente_cg)
	np.save("gote_category.npy", gote_cg)
