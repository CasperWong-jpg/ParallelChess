/**
 *  TODO: Add Castling (FEN tokens) + en passant (FEN tokens)
 *  Optimization: Move ordering
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <stdatomic.h>
#include <stdint.h>
#include <omp.h>

#include "lib/contracts.h"
#include "dataStructs.h"
#include "dev_tools.h"
#include "board_manipulations.h"

#ifndef DEBUG
#define DEBUG (0)  // Debug is off by default
#endif

#define QUIESCE (1)
#define DEPTH (3)  // Number of moves to think ahead

/********************
 * GENERIC AI HELPERS
*********************/
/**
 * Initialize midgame and endgame eval tables. Add piece value to piece location tables
 */
void initEvalTables(void) {
    for (enum EPieceType piece = 0; piece < whiteAll; piece++) {
        for (enum enumSquare sq = a1; sq < totalSquares; sq++) {
            // Tables initialized for black position already
            mg_table[piece + colorOffset][sq] = mg_value[piece] + mg_pesto_table[piece][sq];
            eg_table[piece + colorOffset][sq] = eg_value[piece] + eg_pesto_table[piece][sq];

            // Tables need to be flipped vertically for white position
            mg_table[piece][sq] = mg_value[piece] + mg_pesto_table[piece][FLIP(sq)];
            eg_table[piece][sq] = eg_value[piece] + eg_pesto_table[piece][FLIP(sq)];
        }
    }
}


/**
 * Evaluates material balance of position boards
 * @return Sum(weighting * (playingColorCount - waitingColorCount)) ie. +ve means advantage for playing team
 * @cite https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function
 * @cite https://www.chessprogramming.org/Simplified_Evaluation_Function
 */
int evaluateMaterial(uint64_t *BBoard, bool whiteToMove) {
    int mgScore = 0;  // Middle game score
    int egScore = 0;  // End game score
    int gamePhase = 0;
    for (enum EPieceType piece = 0; piece < whiteAll; piece++) {
        for (enum EPieceType color = 0; color < numPieceTypes; color += colorOffset) {
            uint64_t board = BBoard[piece + color];
            while (board) {
                enum enumSquare sq = bitScanForward(board);
                // Add piece value and piece location to score. Black pieces -ve
                mgScore = !color ? mgScore + mg_table[piece][sq] : mgScore - mg_table[piece + color][sq];
                egScore = !color ? egScore + eg_table[piece][sq] : egScore - eg_table[piece + color][sq];
                // Update game phase
                gamePhase += gamePhaseInc[piece];
                board &= board - 1;
            }
        }
    }
    // tapered eval
    int mgPhase = gamePhase > 24 ? 24: gamePhase;
    int egPhase = 24 - mgPhase;
    int score = (mgScore * mgPhase + egScore * egPhase) / 24;

    // Change score depending on team color's move
    if (!whiteToMove) score *= -1;
    return score;
}


/************************
 * SLIDING MOVE FUNCTIONS
************************/
uint64_t generateSlidingMoves(enum enumSquare index, uint64_t *BBoard, bool whiteToMove,
                              uint64_t (*rays[4]) (enum enumSquare)) {
    // Get useful boards and initialize attacks board to return
    uint64_t friendlyBoard = BBoard[whiteAll + !whiteToMove * colorOffset];
    uint64_t occupied = BBoard[whiteAll] | BBoard[blackAll];
    uint64_t attacks = 0;
    for (int i = 0; i < 4; i++) {
        // Loop through the attack rays
        uint64_t (*ray)(enum enumSquare sq) = rays[i];
        // Get attack squares and blocking pieces
        uint64_t attack = ray(index);
        uint64_t blockers = attack & occupied;
        if (blockers) {
            // Get the square of first piece blocking ray and create an attack ray from here
            enum enumSquare block_sq;
            if (i % 2) {block_sq = bitScanReverse(blockers);}  // Odd indexes store negative rays
            else {block_sq = bitScanForward(blockers);}  // Even indexes store positive rays
            // Subtract blocker rays from attack ray
            attack = attack ^ ray(block_sq);
        }
        attacks |= attack;
    }
    return attacks ^ (attacks & friendlyBoard);
}

/************************
 * BISHOP MOVE FUNCTIONS
************************/
/**
 * Generates all pseudo-legal bishop moves
 * @param bishop_index index of current bishop
 * @return bitboard containing the possible positions that bishop can move to
 * @cite https://www.chessprogramming.org/Sliding_Piece_Attacks
 */
uint64_t generateBishopMoves(enum enumSquare bishop_index, uint64_t *BBoard, bool whiteToMove) {
    // Store functions to get four rays of piece in an array to loop through.
    // Require functions to alternate between positive rays (even indices) and negative rays (odd indices)
    uint64_t (*rays[4]) (enum enumSquare sq);
    rays[0] = northEastRay; rays[1] = southWestRay; rays[2] = northWestRay; rays[3] = southEastRay;
    return generateSlidingMoves(bishop_index, BBoard, whiteToMove, rays);
}


/************************
 * ROOK MOVE FUNCTIONS
************************/
/**
 * Generates all pseudo-legal rook moves
 * @param rook_index index of current rook
 * @return bitboard containing the possible positions that rook can move to
 */
uint64_t generateRookMoves(enum enumSquare rook_index, uint64_t *BBoard, bool whiteToMove) {
    // Store functions to get four rays of piece in an array to loop through.
    // Require functions to alternate between positive rays (even indices) and negative rays (odd indices)
    uint64_t (*rays[4]) (enum enumSquare sq);
    rays[0] = northRay; rays[1] = southRay; rays[2] = eastRay; rays[3] = westRay;

    return generateSlidingMoves(rook_index, BBoard, whiteToMove, rays);
}


/************************
 * QUEEN MOVE FUNCTIONS
************************/
/**
 * Generates all pseudo-legal queen moves
 * @param queen_index index of current queen
 * @return bitboard containing the possible positions that queen can move to
 */
uint64_t generateQueenMoves(enum enumSquare queen_index, uint64_t *BBoard, bool whiteToMove) {
    return generateRookMoves(queen_index, BBoard, whiteToMove) |
            generateBishopMoves(queen_index, BBoard, whiteToMove);
}


/************************
 * KNIGHT MOVE FUNCTIONS
************************/
/**
 * Generates all pseudo-legal knight moves
 * @param knight bitboard containing knight position
 * @return bitboard containing the possible positions that knight can move to
 * @cite Multiple Knight Attacks: https://www.chessprogramming.org/Knight_Pattern
 */
uint64_t generateKnightMoves(enum enumSquare knight_index, uint64_t *BBoard, bool whiteToMove) {
    uint64_t friendlyBoard = BBoard[whiteAll + !whiteToMove * colorOffset];
    uint64_t knight = 1UL << knight_index;
    uint64_t left1 = (knight >> 1) & not_h_file;
    uint64_t left2 = (knight >> 2) & not_hg_file;
    uint64_t right1 = (knight << 1) & not_a_file;
    uint64_t right2 = (knight << 2) & not_ab_file;
    uint64_t hori1 = left1 | right1;
    uint64_t hori2 = left2 | right2;
    uint64_t moveSet = (hori1<<16) | (hori1>>16) | (hori2<<8) | (hori2>>8);
    return moveSet ^ (moveSet & friendlyBoard);
}


/************************
 * KING MOVE FUNCTIONS
************************/
/**
 * Generates all (possibly illegal) king moves
 * @param king bitboard containing king position
 * @return bitboard containing the possible positions that king can move to
 * @cite Multiple King Attacks: https://www.chessprogramming.org/King_Pattern
 */
uint64_t generateKingMoves(enum enumSquare king_index, uint64_t *BBoard, bool whiteToMove, uint64_t castling) {
    (void) castling;
    /* Castling
     * 1. Check if castling token allows queenside & kingside castling
     *  - If yes, this means king and rook has not been moved yet
     * 2. Obtain all pieces board
     * 3. Check squares in between are free (ie. use magic number (1<<f1 | 1<<g1 etc.)
     * 4. Add this move in make_move
     */

    // Normal moves
    uint64_t friendlyBoard = BBoard[whiteAll + !whiteToMove * colorOffset];
    uint64_t king = 1UL << king_index;
    uint64_t l1 = (king >> 1) & not_h_file;
    uint64_t r1 = (king << 1) & not_a_file;
    uint64_t hori1 = king | l1 | r1;
    uint64_t moveSet = king ^ (hori1 | (hori1<<8) | (hori1>>8));
    return moveSet ^ (moveSet & friendlyBoard);
}


/************************
 * PAWN MOVE FUNCTIONS
************************/
uint64_t wSinglePushTargets(uint64_t wpawns, uint64_t empty) {
    return (wpawns << 8) & empty;
}

uint64_t wDoublePushTargets(uint64_t wpawns, uint64_t empty) {
    const uint64_t rank4 = 0x00000000FF000000;int64_t singlePushes = wSinglePushTargets(wpawns, empty);
    return (singlePushes << 8) & empty & rank4;
}

uint64_t wAttackTargets(uint64_t wpawns, uint64_t enemy) {
    return ((wpawns << 7 & not_h_file) | (wpawns << 9 & not_a_file)) & enemy;
}

uint64_t bSinglePushTargets(uint64_t bpawns, uint64_t empty) {
    return (bpawns >> 8) & empty;
}

uint64_t bDoublePushTargets(uint64_t bpawns, uint64_t empty) {
    const uint64_t rank5 = 0x000000FF00000000;
    uint64_t singlePushes = bSinglePushTargets(bpawns, empty);
    return (singlePushes >> 8) & empty & rank5;
}

uint64_t bAttackTargets(uint64_t bpawns, uint64_t enemy) {
    return ((bpawns >> 7 & not_a_file) | (bpawns >> 9 & not_h_file)) & enemy;
}


/**
 * Generates all pseudo-legal white pawn moves
 * @param pawn bitboard containing pawn position
 * @return bitboard containing the possible positions that pawn can move to
 * @cite https://www.chessprogramming.org/Pawn_Pushes_(Bitboards)
 * @cite https://www.chessprogramming.org/Pawn_Attacks_(Bitboards)
 */
uint64_t generateWhitePawnMoves(enum enumSquare pawn_index, uint64_t *BBoard) {
    uint64_t pawn = 1UL << pawn_index;
    uint64_t empty = ~(BBoard[whiteAll] | BBoard[blackAll]);
    uint64_t pushSet = wSinglePushTargets(pawn, empty) | wDoublePushTargets(pawn, empty);
    uint64_t attackSet = wAttackTargets(pawn, BBoard[blackAll]);
    return pushSet | attackSet;
}

/**
 * Generates all pseudo-legal black pawn moves
 */
uint64_t generateBlackPawnMoves(enum enumSquare pawn_index, uint64_t *BBoard) {
    uint64_t pawn = 1UL << pawn_index;
    uint64_t empty = ~(BBoard[whiteAll] | BBoard[blackAll]);
    uint64_t pushSet = bSinglePushTargets(pawn, empty) | bDoublePushTargets(pawn, empty);
    uint64_t attackSet = bAttackTargets(pawn, BBoard[whiteAll]);
    return pushSet | attackSet;
}

/**
 * Wrapper function that calls the pawn move generation function for corresponding color
 */
uint64_t generatePawnMoves(enum enumSquare pawn_index, uint64_t *BBoard, bool whiteToMove, uint64_t enPassant) {
    (void) enPassant;
    if (whiteToMove) {return generateWhitePawnMoves(pawn_index, BBoard);}
    else {return generateBlackPawnMoves(pawn_index, BBoard);}
}


/**********************
 * MAIN MOVE HELPERS
**********************/
/**
 * Initialize a linked list of piece types for one color + corresponding functions used in move generation
 * @return Head of a generic linked list, which contains generic_get_move pointers as data
 */
node get_pieces_struct(uint64_t castling, uint64_t enPassant, bool whiteToMove) {
    generic_fp func_table[whiteAll] = {(generic_fp)generatePawnMoves, (generic_fp)generateKnightMoves,
                                       (generic_fp)generateBishopMoves, (generic_fp)generateRookMoves,
                                       (generic_fp)generateQueenMoves, (generic_fp)generateKingMoves };
    node head = NULL;
    node piece_node;
    generic_get_move piece_list;

    for (enum EPieceType pieceType = whitePawns; pieceType < whiteAll; pieceType++) {
        if (head == NULL) {  // Create list head
            head = malloc(sizeof(node));
            piece_node = head;
        }
        else {  // Extend existing list
            piece_node->next = malloc(sizeof(node));
            piece_node = piece_node->next;
        }

        piece_list = malloc(sizeof(struct generic_get_move_struct));
        piece_list->pieceType = pieceType + !whiteToMove * colorOffset;
        switch (pieceType) {  // Pawns and kings need additional data for their moves
            case whitePawns:
                piece_list->move_gen_func_ptr.additional = (additional_move_fp)func_table[pieceType];
                piece_list->initialized = true;
                piece_list->additional_data = enPassant;
                break;
            case whiteKing:
                piece_list->move_gen_func_ptr.additional = (additional_move_fp)func_table[pieceType];
                piece_list->initialized = true;
                piece_list->additional_data = castling;
                break;
            default:
                piece_list->move_gen_func_ptr.normal = (normal_move_fp)func_table[pieceType];
                piece_list->initialized = false;
                break;
        }

        piece_node->data = (void *) piece_list;  // Add data to list node
        piece_node->next = NULL;
    }

    return head;
}


/*******************
 * LEGALITY CHECKING
*******************/
/**
 * Check whether king of specified color is put in check. Used for legality checking.
 * @param BBoard
 * @param checkWhite
 * @return
 */
bool isInCheck(uint64_t *BBoard, bool whiteMoved) {
    /**
     Pseudocode:
     - Get king board for friendly color, and all piece type boards for enemy color
     - Loop through each enemy color board, and generate moves (could use get_pieces_struct)
       - Intersect with king board. Return true if bitboard not empty
     - After all loops, return false!
    */
    // Get king board for friendly color, and all piece types
    uint64_t kingBoard = BBoard[whiteKing + !whiteMoved * colorOffset];
    node piece_list = get_pieces_struct(0, 0, !whiteMoved);  // Castling and en-passant irrelevant

    for (node piece_node = piece_list; piece_node != NULL; piece_node = piece_node->next) {
        // Loop through enemy piece types & boards
        generic_get_move piece = (generic_get_move) piece_node->data;
        uint64_t pieceBoard = BBoard[piece->pieceType];

        while (pieceBoard) {
            // For each piece, generate all pseudo-legal moves
            enum enumSquare piece_index = bitScanForward(pieceBoard);
            uint64_t pieceMoves;
            if (piece->initialized) {
                pieceMoves = (piece->move_gen_func_ptr.additional)(
                        piece_index, BBoard, !whiteMoved, piece->additional_data
                );
            } else {
                pieceMoves = (piece->move_gen_func_ptr.normal)(piece_index, BBoard, !whiteMoved);
            }
            if (pieceMoves & kingBoard) {  // Friendly king within enemy moves list, in check!
                return true;
            }
            pieceBoard &= pieceBoard - 1;
        }
    }
    return false;  // King is not in moves list of any enemy piece, so is safe :)
}


/**
 * Make move to see if it is legal
 * @param BBoard
 * @param whiteToMove
 * @param m
 * @return
 */
bool checkMoveLegal(uint64_t *BBoard, bool whiteToMove, move m) {
    // Make move on a copy of BBoard, since we want to unmake the move
    uint64_t tmpBBoard[numPieceTypes];
    memcpy(tmpBBoard, BBoard, numPieceTypes * sizeof(uint64_t));

    for (enum EPieceType i = 0; i < numPieceTypes; i++) {
        ASSERT(BBoard[i] == tmpBBoard[i]);
    }

    make_move(tmpBBoard, m);
    bool inCheck = isInCheck(tmpBBoard, whiteToMove);

    // Free the temporary bitboard
    return !inCheck;
}


/**********************
 * MAIN MOVE GENERATION
**********************/
/**
 * Generates all legal moves for player to move
 * @param BBoard
 * @param whiteToMove Player to move
 * @param castling Bitboard containing all castling squares (all colors included)
 * @param enPassant Bitboard containing all en-passant squares (all colors included)
 * @return An linked list of move_info pointers that contain all legal knight moves
 */
node getMoves(uint64_t *BBoard, bool whiteToMove, uint64_t castling, uint64_t enPassant) {
    node piece_list = get_pieces_struct(castling, enPassant, whiteToMove);
    node move_head = NULL;
    node move_list = move_head;

    for (node piece_node = piece_list; piece_node != NULL; piece_node = piece_node->next) {
        generic_get_move piece = (generic_get_move) piece_node->data;
        uint64_t pieceBoard = BBoard[piece->pieceType];
        while (pieceBoard) {
            // For each piece, generate all pseudo-legal moves
            enum enumSquare piece_index = bitScanForward(pieceBoard);
            uint64_t pieceMoves;
            if (piece->initialized) {  // Kings & pawns have additional status (castling & en passant) to keep track of
                pieceMoves = (piece->move_gen_func_ptr.additional)(
                        piece_index, BBoard, whiteToMove, piece->additional_data
                );
            }
            else {  // Normal pieces have normal function signature without additional uint64_t argument
                pieceMoves = (piece->move_gen_func_ptr.normal)(piece_index, BBoard, whiteToMove);
            }
            while (pieceMoves) {
                // For each move, check if legal
                enum enumSquare piece_move = bitScanForward(pieceMoves);
                move m = malloc(sizeof(struct move_info));
                m->from = piece_index;
                m->to = piece_move;
                m->piece = piece->pieceType;
                if (checkMoveLegal(BBoard, whiteToMove, m)) {
                    // Legal move: add to list of possible moves
                    if (move_list == NULL) {
                        move_list = malloc(sizeof(struct Node));
                        move_head = move_list;
                    }
                    else {
                        move_list->next = malloc(sizeof(struct Node));
                        move_list = move_list->next;
                    }
                    move_list->data = (void *) m;
                    move_list->next = NULL;
                }
                else {
                    free(m);
                }
                pieceMoves &= pieceMoves - 1;
            }
            pieceBoard &= pieceBoard - 1;
        }
    }
    free_linked_list(piece_list);
    return move_head;
}


/**
 * Search performed at the end of a main search to only evaluate "quiet" positions.
 * This is performed to avoid the horizon effect (ie. an imminent capture that was just out of our depth)
 * @param BBoard
 * @return
 */
int quiesce(uint64_t *BBoard, bool whiteToMove, uint64_t castling, uint64_t enPassant, uint32_t depth, int alpha, int beta, int* nodesVisited) {
    (*nodesVisited)++;

    // Get a "stand pat" score - need to stop searching without necessarily searching all available captures
    int stand_pat = evaluateMaterial(BBoard, whiteToMove);

    // Pruning / early exit conditions
    if (depth == 0) return stand_pat;  // Hit a depth of 0
    if (stand_pat >= beta) return beta;  // Stand pat score causes beta cutoff
    if (stand_pat > alpha) alpha = stand_pat;  // Current position is better than what we have encontered

    // Generate all enemy moves
    uint64_t enemyBoard = BBoard[whiteAll + colorOffset * whiteToMove];
    node move_list = getMoves(BBoard, whiteToMove, castling, enPassant);
    uint64_t tmpBBoard[numPieceTypes];
    for (node move_node = move_list; move_node != NULL; move_node = move_node->next) {
        move m = (move) move_node->data;
        if (((1UL << m->to) & enemyBoard) != 0) {
            // We only further evaluate captures
            memcpy(tmpBBoard, BBoard, numPieceTypes * sizeof(uint64_t));
            make_move(tmpBBoard, m);
            int currScore = -quiesce(tmpBBoard, !whiteToMove, castling, enPassant, depth-1, -beta, -alpha, nodesVisited);

            if (currScore >= beta) {
                alpha = beta;  // Return beta
                break;
            }
            if (currScore > alpha) {
                alpha = currScore;
            }
        }
    }
    free_linked_list(move_list);
    return alpha;
}


/**
 * Variant of minimax algorithm, which is used to determine score after a certain number of moves.
 * Each side is trying to make the best move they can possibly make (by maximizing their score)
 * @param depth
 * @return Best score
 * @cite https://www.chessprogramming.org/Negamax
 */
int negaMax(uint64_t *BBoard, bool whiteToMove, uint64_t castling, uint64_t enPassant, uint32_t depth, int alpha, int beta, int* nodesVisited) {
    (*nodesVisited)++;

    if (depth == 0) {
        // Quiesce considers horizon effect, but sacrifices time for more depth
#if QUIESCE
        int score = quiesce(BBoard, whiteToMove, castling, enPassant, 3, alpha, beta, nodesVisited);
#else
        int score = evaluateMaterial(BBoard, whiteToMove);
#endif
        return score;
    }

    node move_list = getMoves(BBoard, whiteToMove, castling, enPassant);
    if (move_list == NULL) {  // Checkmate. No possible moves
        free_linked_list(move_list);
        return INT_MIN + 2;  // Default worst case is INT_MIN+1. This is still a move, which is better than nothing
    }
    uint64_t tmpBBoard[numPieceTypes];
    for (node move_node = move_list; move_node != NULL; move_node = move_node->next) {
        // Make move, evaluate it, (and unmake move if necessary)
        move m = (move) move_node->data;
        memcpy(tmpBBoard, BBoard, numPieceTypes * sizeof(uint64_t));
        make_move(tmpBBoard, m);
        int currScore = -negaMax(tmpBBoard, !whiteToMove, castling, enPassant, depth-1, -beta, -alpha, nodesVisited);

        // If this is the best move, then return this score
        if (currScore >= beta) {
            alpha = beta;  // Return beta
            break;
        }
        if (currScore > alpha) {
            alpha = currScore;
        }
    }

    free_linked_list(move_list);
    return alpha;
}


/**
 * Given bitboards and metadata, calculates the best move using MiniMax algorithm
 * @param tokens ie. bitboards, whiteToMove, castling, other FEN info
 * @param bestMove Pointer to the best move
 * @return move, a pointer to a move_info struct
 */
move AIMove(FEN tokens, move bestMove, int* nodesVisited) {
    // Generate all possible legal moves
    uint64_t *BBoard = tokens->BBoard;
    bool whiteToMove = tokens->whiteToMove;
    uint64_t castling = tokens->castling;
    uint64_t enPassant = tokens->enPassant;
    node move_list = getMoves(BBoard, whiteToMove, castling, enPassant);
    if (move_list == NULL) {
        printf("No moves possible. Stalemate or checkmate\n");
        exit(1);
    }

    // With all possible moves, use negaMax to find best move
    // AIMove serves as a root negaMax, which returns the best move instead of score
    int depth = DEPTH;
    ASSERT(depth > 0);
    int bestScore = INT_MIN + 1;
    uint64_t tmpBBoard[numPieceTypes];

    // OpenMP parallel for requires simple addition for loop variable
    // So we convert our linked list to array of moves (void* type)
    int num_moves = find_length(move_list);
    void **move_array = malloc(sizeof(void*) * num_moves);
    convert_to_array(move_list, move_array);

    #pragma omp parallel for private(tmpBBoard) schedule(dynamic)
    for (int i = 0; i < num_moves; i++) {
        // Make move, evaluate it, (and unmake move if necessary)
        move m = (move) move_array[i];
        memcpy(tmpBBoard, BBoard, numPieceTypes * sizeof(uint64_t));
        make_move(tmpBBoard, m);
        int currScore = -negaMax(tmpBBoard, !whiteToMove, castling, enPassant, depth-1, INT_MIN+1, INT_MAX, nodesVisited);

        // If this is the best move, then store it
        #pragma omp critical
        {
            if (currScore > bestScore) {
                bestScore = currScore;
                bestMove->from = m->from;
                bestMove->to = m->to;
                bestMove->piece = m->piece;
            }
        }
    }
    #pragma omp taskwait

    free_linked_list(move_list);
    free(move_array);
    return bestMove;
}


/**
 * The meat of script, does anything and everything right now
 * @return An exit code (0 = successful exit)
 */
char *lichess(char *board_fen, char *move_string) {
    // Set up number of workers
    int num_threads = 0;
    if (getenv("OMP_NUM_THREADS") != NULL) {  // atoi cannot read NULL
        num_threads = atoi(getenv("OMP_NUM_THREADS"));
    }
    num_threads = 0 ? 1 : num_threads;  // atoi error returns 0
    omp_set_num_threads(num_threads);
    #pragma omp parallel 
    {
        #pragma omp single
        printf("Running with %d thread(s)\n", omp_get_num_threads());
    }

    // Initialize data tables
    initEvalTables();

    // Extract info from FEN string
    FEN tokens = extract_fen_tokens(board_fen);

    /// Do AI stuff here
    move bestMove = calloc(1, sizeof(struct move_info));
    time_t start = clock();
    int nodesVisited = 0;
    AIMove(tokens, bestMove, &nodesVisited);
    time_t finish = clock();

    printf("Before AI move - ");
    render_all(tokens->BBoard);
    int score = evaluateMaterial(tokens->BBoard, tokens->whiteToMove);
    printf("Score: %d \n", score);

    make_move(tokens->BBoard, bestMove);

    printf("After AI move - ");
    render_all(tokens->BBoard);
    score = evaluateMaterial(tokens->BBoard, tokens->whiteToMove);
    printf("Score: %d \n", score);
    double time_elapsed = (double) (finish - start) / (num_threads * CLOCKS_PER_SEC);
    printf("Time elapsed: %f \n", time_elapsed);
    printf("Nodes visited: %d \n", nodesVisited);
    printf("---------------------------------\n");

    enumSquare_to_string(move_string, bestMove->from);
    enumSquare_to_string(&move_string[2], bestMove->to);
    move_string[4] = '\0';

    // Free pointers
    free(bestMove);
    free_tokens(tokens);
    return move_string;
}


/**
 * Wrapper for lichess function, useful for easy debugging in CLion
 * Specify a FEN string in body of function
 * @return An exit code (0 = successful exit)
 */
int main(void) {
    char *board_fen = malloc(sizeof(char) * 100);
    char *move_string = malloc(sizeof(char) * 5);

    /// Input FEN String here
    strcpy(board_fen, "2r1kr2/1pp3p1/p1n1pbPp/2Nq3P/PP1Pp1Q1/2P5/5P2/R1B1K2R b Q - 3 25");  /// Input a FEN_string here!

    // Call lichess and make AI move
    move_string = lichess(board_fen, move_string);

    // Free pointers
    free(board_fen);
    free(move_string);
    return 0;
}
