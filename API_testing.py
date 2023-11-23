import sys
import ctypes

so_file = sys.path[0] + "/lichess_bot/engines/ChessEngine.so"
ChessEngine = ctypes.CDLL(so_file)
ChessEngine.lichess.restype = ctypes.c_char_p

res = ChessEngine.lichess(bytes("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 'utf-8'), "")
print(res.decode())