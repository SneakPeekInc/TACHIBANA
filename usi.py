import numpy as np

class USI(object):
    def __init__(self):
        self.KOMA_DICT = {
            "P":1,   "p":-1,
            "L":2,   "l":-2,
            "N":3,   "n":-3,
            "S":4,   "s":-4,
            "B":5,   "b":-5,
            "R":6,   "r":-6,
            "G":7,   "g":-7,
            "K":8,   "k":-8,
            "+P":9,  "+p":-9,
            "+L":10, "+l":-10,
            "+N":11, "+n":-11,
            "+S":12, "+s":-12,
            "+B":13, "+b":-13,
            "+R":14, "+r":-14,
            "0":0
            }

        #逆引き辞書の作成
        self.REV_KOMA_DICT = {v:k for k,v in self.KOMA_DICT.items()}

        #   ---- reference -----
        #col:       9 8 7 6 5 4 3 2 1
        #row:   a
        #       b
        #       c
        #       d
        #       e
        #       f
        #       g
        #       h
        #       i

        self.BOARD_POINTS_DICT = {
        "9a":(0,0),"8a":(0,1),"7a":(0,2),"6a":(0,3),"5a":(0,4),"4a":(0,5),"3a":(0,6),"2a":(0,7),"1a":(0,8),
        "9b":(1,0),"8b":(1,1),"7b":(1,2),"6b":(1,3),"5b":(1,4),"4b":(1,5),"3b":(1,6),"2b":(1,7),"1b":(1,8),
        "9c":(2,0),"8c":(2,1),"7c":(2,2),"6c":(2,3),"5c":(2,4),"4c":(2,5),"3c":(2,6),"2c":(2,7),"1c":(2,8),
        "9d":(3,0),"8d":(3,1),"7d":(3,2),"6d":(3,3),"5d":(3,4),"4d":(3,5),"3d":(3,6),"2d":(3,7),"1d":(3,8),
        "9e":(4,0),"8e":(4,1),"7e":(4,2),"6e":(4,3),"5e":(4,4),"4e":(4,5),"3e":(4,6),"2e":(4,7),"1e":(4,8),
        "9f":(5,0),"8f":(5,1),"7f":(5,2),"6f":(5,3),"5f":(5,4),"4f":(5,5),"3f":(5,6),"2f":(5,7),"1f":(5,8),
        "9g":(6,0),"8g":(6,1),"7g":(6,2),"6g":(6,3),"5g":(6,4),"4g":(6,5),"3g":(6,6),"2g":(6,7),"1g":(6,8),
        "9h":(7,0),"8h":(7,1),"7h":(7,2),"6h":(7,3),"5h":(7,4),"4h":(7,5),"3h":(7,6),"2h":(7,7),"1h":(7,8),
        "9i":(8,0),"8i":(8,1),"7i":(8,2),"6i":(8,3),"5i":(8,4),"4i":(8,5),"3i":(8,6),"2i":(8,7),"1i":(8,8),
        }

        #逆引き辞書の作成
        self.REV_BOARD_POINTS_DICT = {v:k for k,v in self.BOARD_POINTS_DICT.items()}

    def SFEN2board(self,com):
        #初期局面
        #lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL

        #ここの関数では上記のような局面を、boardの型に変換できるようにする関数。
        # input  : SFENのposition
        # output : numpy.array型(15*9)

        new_com = ""
        for ix,koma in enumerate(com):
            try:
                nb_zeros = int(koma)
                koma = "0"*nb_zeros
                new_com += koma
            except:
                new_com += koma
                continue

        board = np.zeros((15,9))
        rows = new_com.split("/")

        for r,row in enumerate(rows):
            for c,koma in enumerate(row):
                board[r,c] = self.KOMA_DICT[koma]
        return board

    def com2info(self, com, board):
        # input: command
        # output: info

        # コマンドをinfoに変換する関数
        assert len(com) in (4,5), "Length of command is invalid."

        #打つの動きはこれでオッケー
        if com.find("*") > 0:
            koma = abs(self.KOMA_DICT[before[0]])
            before = (-1,9)
            after  = self.BOARD_POINTS_DICT[after]
            info = [before, after, koma]
            return info

        before, after = com[0:2], com[2:4]
        before = self.BOARD_POINTS_DICT[before]
        after = self.BOARD_POINTS_DICT[after]
        koma = self._get_koma(before,board)

        #成りの場合
        if com.find("+") > 0:
            koma += 8

        info = [before, after, koma]
        return info

    def _position2list(self,com):
        # input: position command
        # output: [board_info,

        com_infos = com.split()
        return com_infos

    def _get_koma(self, before, board):
        # beforeにいる駒を取得
        return board[before[0],before[1]]

    ################## エンジンからGUIに送る為の関数群 ##################

    def info2com(self, info, player, board):
        # input: info = [before, after, koma], player = 1 or -1, current_state
        # output: command, ex) "7g7f"
        before, after, koma = self._cut_info(info)

        after = self.REV_BOARD_POINTS_DICT[after]
        # 打つの時
        if before == (-1,9):
            koma = self.REV_KOMA_DICT[koma*player]
            return koma+"*"+after

        before = self.REV_BOARD_POINTS_DICT[before]

        #成りの時の判定
        if koma > 8:
            if self._get_koma(board) != koma:
                return before+after+"+"

        return before+after

    def _cut_info(self, info):
        return info[0], info[1], info[2]

if __name__ == "__main__":

    ### TEST ###
    usi = USI()
    init_board = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL"
    certain_board = "lnsgkgsn1/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1"
    state = usi.SFEN2board(init_board)
    #state = usi._position2list(certain_board)[0]
    #print(state)
    #print(usi.SFEN2board(state))
    #info = usi.com2info("7g7f+",state)
    #print(info)

    print(usi.info2com([(6,2),(5,2),1],1,state))
