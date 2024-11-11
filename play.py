import chess
import json

from amazon import MoveParts


import asyncio
import os
from flask import Flask, request
from cloudevents.http import from_http

app = Flask(__name__)


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
        print(move)
        print(f"candidate {move_index}:", move.to_play())
    for move in moves:
        if move == None:
            continue
        try:
            print(f"Trying:", move)
            # confirm = input()
            # if confirm == "":
            board.push_san(move.to_play())
            print("Playing", move.to_play())
            return move.to_play()
            # else:
            # board.push_san(confirm)
            # print("Playing", confirm)
            # return confirm
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
                    # confirm = input()
                    # if confirm == "":
                    board.push_san(move.to_play())
                    print("Playing", move.to_play())
                    return move.to_play()
                    # else:
                    # board.push_san(confirm)
                    # print("Playing", confirm)
                    # return confirm
                except:
                    pass

    # next_move = list(board.legal_moves)[0]
    # next_move = input("Next move: ")
    # board.push(next_move)
    # board.push_san(next_move)
    # print("Playing", next_move)
    board.push_san("not found")


def arbitrate_moves(past_moves, openai_moves, google_moves, amazon_moves):
    board = chess.Board()
    plays = []
    print("Total turns", len(openai_moves))
    print(past_moves)
    print(openai_moves)
    print(google_moves)
    print(amazon_moves)
    if past_moves and past_moves[-1] == "":
        del past_moves[-1]
    print(past_moves)
    for turn_number, turn in enumerate(openai_moves):
        if past_moves and turn_number * 2 < len(past_moves):
            board.push_san(past_moves[turn_number * 2])
            plays.append(past_moves[turn_number * 2])
            if turn_number * 2 + 1 < len(past_moves):
                board.push_san(past_moves[turn_number * 2 + 1])
                plays.append(past_moves[turn_number * 2 + 1])
                continue
        print("Turn ", turn_number)
        if not past_moves or turn_number * 2 + 1 != len(past_moves):
            try:
                if turn[0]:
                    openai_move = MoveParts(**turn[0])
                else:
                    openai_move = None
                try:
                    if (
                        google_moves
                        and google_moves[turn_number]
                        and google_moves[turn_number][0]
                    ):
                        google_move = MoveParts(**google_moves[turn_number][0])
                        print("google move", google_moves)
                    else:
                        google_move = None
                except Exception as err:
                    print(f"Error {err} with google move - white")
                    google_move = None
                try:
                    if (
                        amazon_moves
                        and amazon_moves[turn_number]
                        and amazon_moves[turn_number][0]
                    ):
                        amazon_move = MoveParts(**amazon_moves[turn_number][0])
                        print("amazon move", amazon_move)
                    else:
                        amazon_move = None
                except Exception as err:
                    print(f"Error {err} with amazon move - white")
                    amazon_move = None
                print("openai move", openai_move)
                plays.append(
                    make_play(
                        board,
                        google_move,
                        amazon_move,
                        openai_move,
                    )
                )
            except Exception as err:
                print(f"Error {err} in white turn {turn_number}: {turn}")
                break
        try:
            if turn[1]:
                openai_move = MoveParts(**turn[1])
            else:
                openai_move = None
            try:
                if (
                    google_moves
                    and google_moves[turn_number]
                    and google_moves[turn_number][1]
                ):
                    google_move = MoveParts(**google_moves[turn_number][1])
                    print("google move", google_moves)
                else:
                    google_move = None
            except Exception as err:
                print(f"Error {err} with google move - black")
                google_move = None
            try:
                if (
                    amazon_moves
                    and amazon_moves[turn_number]
                    and amazon_moves[turn_number][1]
                ):
                    amazon_move = MoveParts(**amazon_moves[turn_number][1])
                    print("amazon move", amazon_move)
                else:
                    amazon_move = None
            except Exception as err:
                print(f"Error {err} with amazon move - black")
                amazon_move = None
            plays.append(
                make_play(
                    board,
                    google_move,
                    amazon_move,
                    openai_move,
                )
            )
        except Exception as err:
            print(f"Error {err} in black turn {turn_number}: {turn}")
            break
        print(json.dumps(plays))
    if len(plays) < len(openai_moves) * 2:
        plays.append("")
    return plays


# print(json.dumps(google_moves, cls=ChessEncoder))
# print(json.dumps(amazon_moves, cls=ChessEncoder))
# print(json.dumps(openai_moves, cls=ChessEncoder))


"""
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
"""


@app.route("/", methods=["POST"])
def render_dom():
    # print(request)
    # print(request.get_data())
    request_contents = json.loads(request.get_data().decode("utf-8"))
    plays = arbitrate_moves(
        request_contents.get("past_moves", None),
        request_contents.get("openai", None),
        request_contents.get("google", None),
        request_contents.get("amazon", None),
    )
    return (
        json.dumps(
            plays,
        ),
        200,
    )
