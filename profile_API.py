import sys, os, select
import ctypes
import time
import numpy as np
import pandas as pd

so_file = sys.path[0] + "/lichess_bot/engines/ChessEngine.so"
ChessEngine = ctypes.CDLL(so_file)
# Lichess function
ChessEngine.lichess.restype = ctypes.c_char_p

test_fens = [  # (FEN string, best move)
    # OPENINGS
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", None),  # Starting board
    ("rnbqkbnr/pppppppp/8/8/8/6P1/PPPPPP1P/RNBQKBNR b KQkq - 0 1", None),
    ("rnbqkbnr/ppp2ppp/4p3/1N1p4/3P1B2/8/PPP1PPPP/R2QKBNR w KQkq - 0 1", "b5c7"),
    ("r1bqk2r/ppp2ppp/2np1n2/4p1B1/1bBPP3/2N2N2/PPP2PPP/R2QK2R b KQkq - 0 1", None),
    ("r1bqkb1r/ppp2ppp/2n1p1n1/3pP2P/3P4/5N2/PPP2PP1/RNBQKB1R b KQkq - 0 8", "g6e7"),
    ("r1bqk2r/ppp1bppp/2n1p3/3pPn1P/3P2P1/2P2N2/PP1N1P2/R1BQKB1R b KQkq - 0 11", "f5h6"),

    # MIDGAME
    (" 2rqkr2/1ppbb1p1/p1n1pnPp/3p3P/PP1P4/1NPB1N2/5P2/R1BQK2R b Q - 1 20", None),
    ("2rqkr2/1ppN2p1/p1n1pbPp/7P/PP1Pp3/1NP5/5P2/R1BQK2R b Q - 0 23", "d8d7"),
    ("2r1kr2/1ppq2p1/p1n1pbPp/2N4P/PP1Pp3/2P5/5P2/R1BQK2R b Q - 1 24", "d7d5"),
    ("2r1kr2/1pp3p1/p1n1pbPp/2Nq3P/PP1Pp1Q1/2P5/5P2/R1B1K2R b Q - 3 25", None),
    ("2r1kr2/1pp3p1/p1n1QbPp/2Nq3P/PP1P4/2P1p3/5P2/R1B1K2R b Q - 0 26", "d5e6"),

    # ENDGAME
    ("2r1kr2/1pp3p1/p1n1N1Pp/7P/PP6/2b1B3/4K3/R6R b - - 1 30", "c3a1"),
    ("2r1kr2/1pp3p1/p1n1N1Pp/7P/PP6/4B3/4K3/R7 b - - 0 31", "f8g8"),
    ("4k1r1/1pr3p1/2n3Pp/1P5P/8/4B3/4K3/R7 b - - 0 34", "c6d8"),
    ("8/3k4/3P4/3K4/8/8/8/8 b - - 0 1", None),
    ("8/8/8/3k4/8/8/8/5RQK w - - 0 1", None)
]

numThreads = [1, 2, 4, 8]


def profile_engine(numRuns=10):
    # Create dataframes for logging
    columns = [str(numThread) + "core(s)" for numThread in numThreads]
    times = pd.DataFrame(np.zeros((len(test_fens), len(numThreads))), 
                         columns=columns)
    speedup = pd.DataFrame(np.zeros((len(test_fens), len(numThreads))), 
                           columns=columns)
    for col, OMPThreads in enumerate([str(numThread) for numThread in numThreads]):
        for row, (test_fen, best_move) in enumerate(test_fens):
            # Set number of threads (sequential vs parallel) and run AI
            os.environ['OMP_NUM_THREADS'] = OMPThreads
            for i in range(numRuns):
                start = time.time()
                res = ChessEngine.lichess(bytes(test_fen, 'utf-8'), "")
                elapsed = time.time() - start

                # Add times to dataframe
                times.iloc[row, col] += elapsed

                # Check correctness
                res = res.decode()
                if best_move is not None and res != best_move:
                    raise Exception("Correctness failed")
    # Process dataframes
    times /= numRuns
    speedup = 1 / times.div(times[columns[0]], axis=0)
    times.to_csv("times.csv")
    speedup.to_csv("speedup.csv")
    print(times)
    print(speedup)
    print(f"Average speedup: {speedup.mean(axis=0)}")


def get_nodes_visited(path, numRuns=10):
    # Gather all nodes visited in a list
    nodesVisited = []
    with open(path + '.txt','r') as f:
        for line in f:
            if "Nodes visited: " in line:
                nodesVisited.append(int(line.split("Nodes visited: ")[-1]))
    assert(len(nodesVisited) == numRuns * len(test_fens) * len(numThreads))
    
    # Group by numThreads, then test_fen, then numRuns
    columns = [str(numThread) + "core(s)" for numThread in numThreads]
    df = pd.DataFrame(np.zeros((len(test_fens), len(numThreads))), 
                                columns=columns)
    for col in range(len(numThreads)):
        for row in range(len(test_fens)):
            for i in range(numRuns):
                idx = i + row * numRuns + col * len(test_fens) * numRuns
                df.iloc[row, col] += nodesVisited[idx]
    df /= numRuns
    print(df)
    print(f"Average nodes visited: \n{df.mean(axis=0)}")
    df.to_csv(path + '.csv')


if __name__ == '__main__':
    profile_engine()
    # get_nodes_visited("")  # Insert path here
