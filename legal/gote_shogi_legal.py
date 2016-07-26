# coding:utf-8
import numpy as np
import math

class GoteShogi(object):
    def __init__(self):
        ## FOR JUDGING LEGAL ##
        self.KEIMA_LEGAL_GOTE = [(-2,1),(-2,-1)]
        self.GIN_LEGAL_GOTE = [(-1,1),(-1,0),(-1,-1),(1,1),(1,-1)]
        self.KIN_LEGAL_GOTE = [(-1,1),(-1,0),(-1,-1),(0,1),(0,-1),(1,0)]
        self.GYOKU_LEGAL_GOTE = [(r, c) for r in [-1,0,1] for c in [-1,0,1]]
        self.GYOKU_LEGAL_GOTE.remove((0,0))
        self.oute_dic = {
            (1,1)  :[4,5,8,13,14],
            (1,0)  :[6,7,8,9,10,11,12,13,14],
            (1,-1) :[4,5,8,13,14],
            (0,1)  :[6,7,8,9,10,11,12,13,14],
            (0,-1) :[6,7,8,9,10,11,12,13,14],
            (-1,1) :[4,5,7,8,9,10,11,12,13,14],
            (-1,0) :[1,2,4,6,7,8,9,10,11,12,13,14],
            (-1,-1):[4,5,7,8,9,10,11,12,13,14]
            }
        self.KEIMA_ls = [(2,-1),(2,1)]
        self.GOTE_KOMADAI = {
            -koma : (10,c) for koma,c in zip(range(1,8), range(0,7)) }


    ## 後手から見た座標に基づく将棋のルールを定義する ##
    @staticmethod
    def get_coordinate_info(ls, key=0):
        # key==0なら差のタプルを返して、key==1ならintとして返す。
        before = ls[0]
        after = ls[1]
        # (row,column) = (r, c)
        r1, c1 = before[0], before[1]
        r2, c2 = after[0], after[1]
        if key == 0:
            # 今いるところを中心に考えて、(before - after)のタプルを返す。
            return (r1-r2, c1-c2)
        # 差の情報だけでは角や飛車のlegalを定義できないため。
        if key == 1:
            return r1,c1,r2,c2
        if key == 2:
            return (r2-r1, c2-c1)

    # 以下の駒legalで受け取るlsは[(before),(after),koma]のリストを想定している。
    ## ここからそれぞれのコマの動きを定義 ##
    def fu(self,ls):
        # "打"に対する判定(1行目には打てない)
        if ls[0] == (-1,9):
            after = ls[1]
            if after[0] == 8:
                return False
            else:
                return True
        else:
            tp = self.get_coordinate_info(ls)
            if tp == (-1,0):
                return True
            else: return False

    def kyosya(self,ls):
        # "打"に対する判定(1行目には打てない)
        if ls[0] == (-1,9):
            after = ls[1]
            if after[0] == 8:
                return False
            else:
                return True
        else:
            tp = self.get_coordinate_info(ls)
            if tp[0] < 0 and tp[1] == 0:
                return True
            else: return False

    def keima(self,ls):
        # "打"に対する判定(1,2行目には打てない)
        if ls[0] == (-1,9):
            after = ls[1]
            if after[0] > 6:
                return False
            else:
                return True
        else:
            tp = self.get_coordinate_info(ls)
            if tp in self.KEIMA_LEGAL_GOTE:
                return True
            else: return False

    def gin(self,ls):
        # "打"に対する判定(どこにでもおける)
        if ls[0] == (-1,9):
            return True
        else:
            tp = self.get_coordinate_info(ls)
            if tp in self.GIN_LEGAL_GOTE:
                return True
            else: return False

    def kin(self,ls):
        # "打"に対する判定(どこにでもおける)
        if ls[0] == (-1,9):
            return True
        else:
            tp = self.get_coordinate_info(ls)
            if tp in self.KIN_LEGAL_GOTE:
                return True
            else: return False

    def kaku(self,ls):
        # "打"に対する判定(どこにでもおける)
        # 左右上下対象に動ける駒は先手も後手もlegalは変わらない
        if ls[0] == (-1,9):
            return True
        else:
            r1,c1,r2,c2 = self.get_coordinate_info(ls,key=1)
            if r1 == r2 and c1 == c2:
                return False
            if r1+c1 == r2+c2 or r1-c1 == r2-c2:
                return True
            else: return False

    def hisya(self,ls):
        # "打"に対する判定(どこにでもおける)
        if ls[0] == (-1,9):
            return True
        else:
            r1,c1,r2,c2 = self.get_coordinate_info(ls,key=1)
            r1,c1,r2,c2 = self.get_coordinate_info(ls,key=1)
            if r1 == r2 and c1 == c2:
                return False
            if r1 == r2 or c1 == c2:
                return True
            else: return False

    def uma(self,ls):
        # "打"は存在しないのでカテゴリから消す。
        if ls[0] == (-1,9):
                return False
        else:
            r1,c1,r2,c2 = self.get_coordinate_info(ls,key=1)
            tp = self.get_coordinate_info(ls)
            r1,c1,r2,c2 = self.get_coordinate_info(ls,key=1)
            if r1 == r2 and c1 == c2:
                return False
            elif r1+c1 == r2+c2 or r1-c1 == r2-c2:
                return True
            elif tp in [(1,0),(-1,0),(0,1),(0,-1)]:
                return True
            else: return False

    def ryu(self,ls):
        # "打"は存在しないのでカテゴリから消す。
        if ls[0] == (-1,9):
                return False
        else:
            tp = self.get_coordinate_info(ls)
            r1,c1,r2,c2 = self.get_coordinate_info(ls,key=1)
            if r1 == r2 and c1 == c2:
                return False
            daiagonal = [(1,1),(1,-1),(-1,-1),(-1,1)]
            if tp in daiagonal:
                return True
            elif r1 == r2 or c1 == c2:
                return True
            else: return False

    def gyoku(self,ls):
        # "打"は存在しないのでカテゴリから消す。
        if ls[0] == (-1,9):
                return False
        else:
            tp = self.get_coordinate_info(ls)
            if tp in self.GYOKU_LEGAL_GOTE:
                return True
            else: return False


    ## ここからルールの定義 ##

    def judge_nifu(self, board, c_a):
        # Falseだったら2歩ってことにする
        for i in range(9):
            if board[i][c_a] == -1:
                #print("それは２歩だよ！やっぱりダメ！")
                return False
        return True


    def is_uchifudume():
        pass

    def is_sennichite():
        pass

    def is_ote(self, board):
        r,c = self._get_gyoku_point(board)
        if self._around_ote(r,c,board):
            #print("王手やぞ！！")
            return True
        #print("around_ote is False")
        if self._hisya_kaku_ote(r,c,board):
            #print("飛車、角おるで香車も忘れずに！")
            return True
        #print("hisya_kaku_ote is False")
        if self._keima_ote(r,c,board):
            #print("桂馬から王手されてるよん！")
            return True
        #print("keima_ote is False")
        return False

    @staticmethod
    def _get_gyoku_point(board):
        r,c = np.where(board[0:9] == -8)
        return r[0] ,c[0]

    def _around_ote(self,r,c,board):
        ## 玉の周り８マスから王手されているかの判定 ##
        for v in self.GYOKU_LEGAL_GOTE:
            r1 = r-v[0]
            c1 = c-v[1]
            #print("見ているマス目：",r1,c1)
            if r1 < 0 or r1 > 8 or c1 < 0 or c1 > 8: continue
            koma = board[r1][c1]
            #print("そこにいる駒：",koma)
            if koma > 0:
                if koma in self.oute_dic[v]:
                    return True
        return False


    def _hisya_kaku_ote(self,r,c,board):
        ## 飛車(龍)、角(馬)から王手されてるかの判定 ##
        for direction in self.GYOKU_LEGAL_GOTE:
            #print(ote)
            y = r + direction[0]
            x = c + direction[1]
            while True:
                # 盤外にマス目が来たらループを抜ける(while)
                if y < 0 or y > 8 or x < 0 or x > 8: break
                koma = board[y][x]
                #print("見ているマス目：",y,x)

                # 方向の判定(斜めor縦横)
                if direction[0] == 1 and direction[1] == 0:
                    #print("前方判定",koma)
                    # 敵の飛車(龍)以外の駒がいたらループを抜ける。香車忘れてた  (while)
                    if koma != 6 and koma != 14 and koma != 2 and koma != 0: break
                    # 縦下方向:飛車か香車がいるかどうか
                    if koma == 6 or koma == 14 or koma == 2:
                        return True

                elif direction[0] == 0 or direction[1] == 0:
                    #print("縦横判定",koma)
                    # 敵の飛車(龍)以外の駒がいたらループを抜ける。  (while)
                    if koma != 6 and koma != 14 and koma != 0: break
                    # 縦横:飛車がいるかどうか
                    if koma == 6 or koma == 14:
                        return True
                else:
                    #print("斜め判定",koma)
                    # 敵の角(馬)以外の駒がいたらループを抜ける。  (while)
                    if koma != 5 and koma != 13 and koma != 0: break
                    # 斜め:角がいるかどうか
                    if koma == 5 or koma == 13:
                        return True
                y = y + direction[0]
                x = x + direction[1]
        return False


    def _keima_ote(self,r,c,board):
        ## 桂馬から王手されているかの判定 ##
        for v in self.KEIMA_ls:
            rk = r + v[0]
            ck = c + v[1]
            #print("見ているマス目：",rk,ck)
            if rk < 0 or ck < 0 or ck > 8:
                continue
            elif board[rk][ck] == 3:
                #print("そこにいる駒：",board[rk][ck])
                return True

        return False


    ## --- 選んだinfoに矛盾がないか判定する関数 --- ##
    # 引数はランダム、もしくはNNの出力で選択されたinfoと現在のstate
    def judge(self, board, info, komadai):
        # TrueだったらそのInfoをstateに与えるための関数
        r_b, c_b ,r_a, c_a = self.get_coordinate_info(info,key=1)
        # ベクトルを取得
        v1, v2 = self.get_coordinate_info(info, key=2)

        koma = -1 * info[2] #後手の駒は負なので−1を先にかける

        if (r_b,c_b) == (-1, 9):
            #print("鬼いちゃん。その駒、本当に打てるの？")
            if self._judge_put(board, r_a, c_a, koma, komadai):
                if koma == -1:
                    return self.judge_nifu(board, c_a)
                else:
                    return True

        elif board[r_a][c_a] < 0:
            #print("先輩！自分の駒を取ろうとするなんて、とんでもない変態だな！")
            return False

        elif board[r_b][c_b] != koma:
            # ここで成りゴマが選択されていた場合は分岐が必要
            # 4パターン存在している
            # その１、飛び越えることを考えなくて良い、歩、桂馬、銀の成り。
            # その２、駒飛び越えて成る可能性がある、香車のなり
            # その３、そもそもそれも含めて考えられている馬と龍
            # その４、愚かなのか
            # ここで注意しなければいけないのが、成るためにはr_a<3でならないということ。
            # まずここで元いた場所に元の姿が存在するのか確認
            if (board[r_b][c_b] == koma+8) and (r_a > 5 or r_b > 5):
                # こっちに入るのは、駒ナンバーが9より大きいものだけ。
                if koma == -9: #歩成
                    #print("ときん作れるのはでかい")
                    return self.fu(info)

                elif koma == -10:
                    #print("かかっ。香成りか。")
                    return self._judge_kyosya(board, r_b, c_b, v1, v2)

                elif koma == -11:
                    #print("桂成いきます")
                    return self.keima(info)

                elif koma == -12:
                    #print("銀なるで〜")
                    return self.gin(info)

                elif koma == -13:
                    # 角成
                    # 成る瞬間の動きは角と同じはずなので、角の審査を通す。
                    if (v1,v2) in [(1,0),(-1,0),(0,-1),(0,1)]:
                        #print("おい、有象無象。")
                        return False
                    #print("先輩！角をなろうとしてるんですか！？")
                    return self._judge_kaku(board, r_b, c_b, v1, v2)

                elif koma == -14:
                    # 飛成
                    # 成る瞬間の動きは飛車と同じはずなので、飛車の審査を通す。
                    if (v1,v2) in [(1,1),(1,-1),(-1,-1),(-1,1)]:
                        #print("今回の件からお前が得るべき教訓は、飛車が斜めに動いたら龍だと思えということだ。")
                        return False
                    #print("あ、先輩！飛車をなるんですね！")
                    return self._judge_hisya(board, r_b, c_b, v1, v2)

            else:
                #print("そこにその駒はないじゃないですか。愚かですねー。")
                return False


        else:
            ## 香車、角、飛車が飛び越えていないかの判定
            if koma == -2:#香車
                #print("かかっ。香車か。わしの出番じゃのう。")
                return self._judge_kyosya(board, r_b, c_b, v1, v2)
            elif koma == -5:#角
                #print("失礼、かくみました。")
                return self._judge_kaku(board, r_b, c_b, v1, v2)
            elif koma == -6:#飛車
                #print("なんでもは知らないわよ。知ってることだけ。")
                return self._judge_hisya(board, r_b, c_b, v1, v2)
            elif koma in (-9,-10,-11,-12):
                # 成駒クラスには進化前の動きをするのも入っているので、
                # 金の動きをしているか審査する。
                #print("成駒を動かすのかい？")
                return self.kin(info)
            elif koma == -13:#馬
                #print("なーでこーだよー！お馬さんの確認だね！")
                return self._judge_uma(board, r_b, c_b, v1, v2)
            elif koma == -14:#龍
                #print("知りたいか。教えてやろう、金を払え。")
                return self._judge_ryu(board, r_b, c_b, v1, v2)
            else:
                #print("愚かでゴミのようなあなたでも、手の選択はできるのね。")
                return True

    def _judge_kyosya(self, board, r_b, c_b, v1, v2):
        #香車はc軸の差が必ず0になるので、r軸だけ見れば良い
        #print(r,c)
        v1 -= 1
        while v1 != 0:
            # rを現在いる位置まで近づけていく
            if self._does_exist(board, r_b, c_b, v1, v2):
                #print("おい、うぬ。今いる座標({},{})+方向({},{})にコマを発見したぞ。".format(r_b, c_b, v1, v2))
                #print("Falseじゃ")
                return False
            v1 -= 1
        #print("この手はさせるようじゃのう。指すぞ？")
        return True

    def _judge_kaku(self, board, r, c, v1, v2):
        v1 -= np.sign(v1)
        v2 -= np.sign(v2)
        while v1 != 0 or v2 != 0:
            #今いる位置までafterから近づけて捜査する
            if self._does_exist(board, r, c, v1, v2):
                #print("おや？間に駒がいるみたいですね。")
                return False
            v1 -= np.sign(v1)
            v2 -= np.sign(v2)
        #print("大丈夫みたいです、修羅羅木さん！")
        return True

    def _judge_hisya(self, board, r, c, v1, v2):
        v1 -= np.sign(v1)
        v2 -= np.sign(v2)
        if v2 == 0:
            #print("縦を確認するね。")
            while not v1 == 0:
                if self._does_exist(board, r, c, v1, v2):
                    #print("ダメじゃない、阿良々木君。飛車は駒を飛び越えられないんだよ。")
                    return False
                v1 -= np.sign(v1)
            #print("うん、大丈夫。")
            return True

        elif v1 == 0:
            #print("横を確認するね。")
            while not v2 == 0:
                if self._does_exist(board, r, c, v1, v2):
                    #print("ダメじゃない、阿良々木君。飛車は駒を飛び越えられないんだよ。")
                    return False
                v2 -= np.sign(v2)
            #print("うん。大丈夫。")
            return True

    def _judge_uma(self, board, r, c, v1, v2):
        # まずは上下左右を捜査
        if (v1,v2) in [(1,0),(-1,0),(0,-1),(0,1)]:
            #print("おい、有象無象。")
            return True
        # そのあと角と同様の判定
        else:
            #print("上下左右には動いてないみたいだよ。")
            #print("ここからは八九寺さんに確認してもらうね！")
            return self._judge_kaku(board, r, c, v1, v2)

    def _judge_ryu(self, board, r, c, v1, v2):
        # まずは斜めを捜査
        if (v1,v2) in [(1,1),(1,-1),(-1,-1),(-1,1)]:
            #print("今回の件からお前が得るべき教訓は、飛車が斜めに動いたら龍だと思えということだ。")
            return True
        else:
            #print("おい、羽川。これはお前の仕事だ。")
            return self._judge_hisya(board, r, c, v1, v2)

    @staticmethod
    def _does_exist(board, r, c, v1, v2):
        if board[r+v1][c+v2] != 0:
            return True
        return False

    @staticmethod
    def _judge_put(board, r, c, koma, komadai):
        if koma in komadai:
            if board[r][c] == 0:
                #print("あ。持ってるね。ごめんよ。")
                return True
            else:
                #print("そこには敵の駒がいるから打てないよ")
                return False
        #print("なんだよ。その駒持ってないじゃん。やりなおし。")
        return False

if __name__ == "__main__":
    legal = GoteShogi()

     ## TEST JUDGE FUNCTION ##

    ## make board you wanna test
    board = np.zeros((15, 9))
    #歩の配置
    board[0:8][6] = 1
    board[0:8][2] = -1
    #香車の配置
    board[8][0],board[8][8] = 2,2
    board[0][0],board[0][8] = -2,-2
    #桂馬の配置
    board[8][1],board[8][7] = 3,3
    board[0][1],board[0][7] = -3,-3
    #銀の配置
    board[8][2],board[8][6] = 4,4
    board[0][2],board[0][6] = -4,-4
    #金の配置
    board[8][3],board[8][5] = 7,7
    board[0][3],board[0][5] = -7,-7
    #王の配置
    board[8][4] = 8
    board[0][4] = -8
    #角の配置
    board[7][1] = 5
    board[1][7] = -5
    #飛車の配置
    board[7][7] = -6
    board[1][1] = -6

    init_board = board

    # 角なりの確認
    komadai = []
    board[2,4] = -6
    board[5,4] = -1
    board[6,4] = -5
    #board[4,6] = -3
    #board[6,5] = -11
    info = [(6,4),(6,5),13]
    print(board)
    print("False",legal.judge(board, info, komadai))
    #info = [(0,0),(6,0),10]
