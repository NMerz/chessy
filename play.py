import chess
import json

from amazon import MoveParts


def make_play(board, *moves):
    piece = []
    source = []
    capture = []
    rank = []
    file = []
    check = []
    for move_index, move in enumerate(moves):
        if move == None:
            continue
        print(f"candidate {move_index}:", move.to_play())
    for move in moves:
        if move == None:
            continue
        try:
            print(f"Trying:", move)
            confirm = input()
            if confirm == "":
                board.push_san(move.to_play())
                print("Playing", move.to_play())
                return move.to_play()
            else:
                board.push_san(confirm)
                print("Playing", confirm)
                return confirm
        except:
            piece.append(move.piece)
            source.append(move.source)
            capture.append(move.capture)
            rank.append(move.rank)
            file.append(move.file)
            check.append(move.check)
    for p in piece:
        for f in file:
            for r in rank:
                try:
                    print("Trying", move)
                    move = MoveParts(
                        p, source[0], capture[0], r, f, check[0], False, False
                    )
                    confirm = input()
                    if confirm == "":
                        board.push_san(move.to_play())
                        print("Playing", move.to_play())
                        return move.to_play()
                    else:
                        board.push_san(confirm)
                        print("Playing", confirm)
                        return confirm
                except:
                    pass

    next_move = list(board.legal_moves)[0]
    next_move = input("Next move: ")
    # board.push(next_move)
    board.push_san(next_move)
    print("Playing", next_move)


def arbitrate_moves(past_moves, openai_moves, google_moves, amazon_moves):
    board = chess.Board()
    plays = []
    print("Total turns", len(openai_moves))
    for turn_number, turn in enumerate(openai_moves):
        if turn_number < len(past_moves) / 2:
            board.push_san(past_moves[turn_number * 2])
            plays.append(past_moves[turn_number * 2])
            board.push_san(past_moves[turn_number * 2 + 1])
            plays.append(past_moves[turn_number * 2 + 1])
            continue
        print("Turn ", turn_number)
        print("trying ", turn, google_moves[turn_number])
        try:
            if turn_number >= len(google_moves) or turn_number >= len(amazon_moves):
                plays.append(make_play(board, turn[0]))
            else:
                plays.append(
                    make_play(
                        board,
                        google_moves[turn_number][0],
                        amazon_moves[turn_number][0],
                        turn[0],
                    )
                )
        except Exception as err:
            print(f"Error {err} in white turn {turn_number}: {turn}")
        try:
            if turn_number >= len(google_moves) or turn_number >= len(amazon_moves):
                plays.append(make_play(board, turn[1]))
            else:
                plays.append(
                    make_play(
                        board,
                        google_moves[turn_number][1],
                        amazon_moves[turn_number][1],
                        turn[1],
                    )
                )
        except Exception as err:
            print(f"Error {err} in black turn {turn_number}: {turn}")
        print(json.dumps(plays))


# print(json.dumps(google_moves, cls=ChessEncoder))
# print(json.dumps(amazon_moves, cls=ChessEncoder))
# print(json.dumps(openai_moves, cls=ChessEncoder))

moves = []

for file in ["openai_moves", "google_moves", "amazon_moves"]:
    with open(file, "r") as move_file:
        move_contents = move_file.read()
    moves.append(json.loads(move_contents, object_hook=lambda mp: MoveParts(**mp)))

past_moves = []
with open("past_moves", "r") as pm:
    past_moves = json.loads(pm.read())

print(moves[2])
arbitrate_moves(past_moves, *moves)
# arbitrate_moves(openai_moves, google_moves, amazon_moves)
