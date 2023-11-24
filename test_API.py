import sys
import ctypes

so_file = sys.path[0] + "/lichess_bot/engines/ChessEngine.so"
ChessEngine = ctypes.CDLL(so_file)
ChessEngine.lichess.restype = ctypes.c_char_p

res = ChessEngine.lichess(bytes("rnbqkbnr/pp3ppp/4p3/1N1p4/3P1B2/8/PPP1PPPP/R2QKBNR w KQkq - 0 1", 'utf-8'), "")
print(res.decode())