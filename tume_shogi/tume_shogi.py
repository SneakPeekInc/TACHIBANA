import numpy as np
import copy
from TACHIBANA.legal.sente_shogi_legal import SenteShogi as sl
from TACHIBANA.legal.gote_shogi_legal import GoteShogi as gl
from TACHIBANA.shogi_ban import GameState as gs

class tsume_shogi(object):
    def __init__(self, board, sente_komadai, gote_komadai):
        self.board = board
        self.sente_komadai = sente_komadai
        self.gote_komadai  = gote_komadai
        self.sl = sl()
        self.gl = gl()
        path_to_sente_cate  = "/Users/kento_watanabe/Desktop/TACHIBANA_project/TACHIBANA/preprocessing/sente_category.npy"
        path_to_gote_cate   = "/Users/kento_watanabe/Desktop/TACHIBANA_project/TACHIBANA/preprocessing/gote_category.npy"
        self.sente_category = np.load(path_to_sente_cate).tolist()
        self.gote_category  = np.load(path_to_gote_cate).tolist()

        mate_infos = self.get_mate_infos()
        self.tree = self.make_tree_root(mate_infos)

    def get_mate_infos(self):
        mate_infos = list()
        for info in self.sente_category:
            board = copy.deepcopy(self.board)
            komadai = copy.deepcopy(self.sente_komadai) #initialize
            if self.sl.judge(board, info, komadai):
                #print("through judgement",info)
                board,komadai = self.update_board(info, \
                                board, komadai, player=1)
                #print(board)
                if self.gl.is_ote(board):
                    mate_infos.append(info)
        print(mate_infos)
        return mate_infos

    def update_board(self, info, board, komadai, player):
        board   = copy.deepcopy(board)
        komadai = copy.deepcopy(komadai)
        b,a,koma = self.cut_info(info)
        if player < 0: koma *= -1

        if b == (-1,9):
            board[a[0],a[1]] = koma
            komadai.remove(koma)
        else:
            board[b[0],b[1]] = 0
            if board[a[0],a[1]] != 0:
                komadai.append(board[a[0],a[1]])
            board[a[0],a[1]] = koma
        return board, komadai

    def cut_info(self,info):
        before = info[0]
        after  = info[1]
        koma   = info[2]
        return before, after, koma

    def make_tree_root(self, mate_infos):
        tree_root = {"next_node":{}}
        for node_num,info in enumerate(mate_infos):
            board   = copy.deepcopy(self.board)
            komadai = copy.deepcopy(self.sente_komadai)
            board, komadai = self.update_board(info,board,komadai,1)
            tree_root["next_node"][node_num] = \
                {"player"    : 1,
                 "board"     : board, #next_board
                 "komadai"   : komadai,
                 "info"      : info,
                 "next_node" : {}}
            #print(tree_root["next_node"][node_num]["board"])
        return tree_root

    def make_next_node(self,parent_node):
        #if parent["is_tsumi"]==False
        player    = parent_node["player"]
        next_node = parent_node["next_node"]
        if player > 0: #gote
            node_num = 0
            for info in self.gote_category:
                board     = parent_node["board"]
                komadai   = parent_node["komadai"]
                if self.gl.judge(board, info, komadai):
                    print("through judgement",info)
                    board, koma = \
                        self.update_board(info, board, komadai, -1)
                    if not self.gl.is_ote(board):
                        print(board)
                        next_node[node_num] = \
                            {"player"   : -1,
                             "info"     : info,
                             "board"    : board,
                             "komadai"  : komadai,
                             "next_node": {}}
                        node_num += 1
        else:
            node_num = 0
            for info in self.sente_category:
                board     = parent_node["board"]
                komadai   = parent_node["komadai"]
                if sl.judge(board, info, komadai):
                    self.update_board(info, board, komdai, 1)
                    if self.gl.is_ote(board):
                        next_node[node_num] = \
                            {"player"   : 1,
                             "info"     : info,
                             "board"    : board,
                             "komadai"  : komadai,
                             "next_node": {}}
                        node_num += 1

    def check_tsumi(self,parent_node):
        player = parent_node["player"]
        assert player > 0, "Current turn is not sente!"
        nb_node = len(parent_node["next_node"])
        if nb_node == 0:
            return True
        else:
            return False





def main():
    board = np.zeros((15, 9))
    board[0,7] = -8
    board[2,7] = 9
    board[2,6] = 1
    print("current_board")
    print(board)
    sente_komadai = []
    gote_komadai = [-1,-1,-1,-2,-2,-3,-4,-5,-6,-7]
    mate = tsume_shogi(board, sente_komadai, gote_komadai)
    root_node = mate.tree["next_node"]
    for node_num in root_node:
        print("node_num",node_num)
        mate.make_next_node(root_node[node_num])



if __name__ == "__main__":
    main()
