import boto3

from typing import Optional, Sequence
from itertools import chain
import re


class MoveParts:
    piece: Optional[str]
    source: Optional[str]
    capture: bool
    rank: Optional[str]
    file: Optional[str]
    check: bool
    ks_castle: bool
    qs_castle: bool

    def __init__(
        self,
        piece: Optional[str],
        source: Optional[str],
        capture: Optional[str],
        rank: Optional[str],
        file: Optional[str],
        check: Optional[str],
        ks_castle: bool,
        qs_castle: bool,
    ):
        self.piece = piece
        if piece == "":
            self.piece = None
        self.source = source
        if source == "":
            self.source = None
        self.capture = capture == "x"
        self.rank = rank
        if rank == "":
            self.rank = None
        self.file = file
        if file == "":
            self.file = None
        self.check = check == "+"
        self.ks_castle = ks_castle
        self.qs_castle = qs_castle

    def __repr__(self):
        return f"MoveParts(piece={self.piece}, source={self.source}, capture={self.capture}, rank={self.rank}, file={self.file}, check={self.check}, ksc={self.ks_castle}, qsc={self.qs_castle})"

    def to_play(self):
        if self.ks_castle:
            return "O-O"
        if self.qs_castle:
            return "O-O-O"
        play_str = ""
        if self.piece != None:
            play_str = play_str + self.piece
        if self.source != None:
            play_str = play_str + self.source
        if self.capture == True:
            play_str = play_str + "x"
        if self.rank != None:
            play_str = play_str + self.rank
        if self.file != None:
            play_str = play_str + self.file
        if self.check == True:
            play_str = play_str + "+"

        return play_str


def extract_groups(to_extract: str) -> Sequence[str]:
    matches = re.compile(r"^([QRBKN])?([a-h]?)(x?)([a-h])([0-9])(\+?)$").match(
        to_extract
    )
    if not matches:
        matches = re.compile(r"^([QRBKN])?([a-h]?)(x?)([a-h])([0-9]?)(\+?)$").match(
            to_extract
        )
    if not matches:
        matches = re.compile(r"^.*(0-9])(\+?)$").match(to_extract)
        if matches:
            matches = matches.groups()
            matches = (None, None, None, None, matches[0], matches[1])
    else:
        matches = matches.groups()
    return matches


def get_move_parts(to_disect: str) -> Optional[MoveParts]:
    matches = extract_groups(to_disect)
    if to_disect in ["O-O", "O-O-O"]:
        return MoveParts(
            None, None, None, None, None, None, to_disect == "O-O", to_disect == "O-O-O"
        )
    if matches:
        return MoveParts(*matches, to_disect == "O-O", to_disect == "O-O-O")
    else:
        return None


def set_pos(to_alter: str, at_pos: int, new_insert: str) -> str:
    new_str = to_alter[:at_pos] + new_insert
    if len(to_alter) > at_pos + 1:
        return new_str + to_alter[at_pos + 1 :]
    return new_str


def fix_ocr_raw(to_fix: str) -> Optional[str]:
    print(f"fixing {to_fix}")
    try:
        to_fix = to_fix.replace("'", "")
        to_fix = to_fix.replace("l", "1")
        to_fix = to_fix.replace("t", "+")
        to_fix = to_fix.replace("²", "2")
        to_fix = to_fix.replace("0-0", "O-O")
        to_fix = to_fix.replace("o-o", "O-O")
        to_fix = to_fix.replace("O-o", "O-O")
        to_fix = to_fix.replace("o-O", "O-O")
        to_fix = to_fix.replace("00", "O-O")
        if "O-O" in to_fix:
            return get_move_parts(to_fix)
        col_index = 0
        if to_fix[0] in ["Q", "R", "B", "K", "N"]:
            col_index = 1
        if to_fix[col_index].lower() == "x":
            to_fix = set_pos(to_fix, col_index, to_fix[col_index].lower())
            col_index += 1
        if to_fix[col_index] == "6":
            to_fix = set_pos(to_fix, col_index, "b")
        if to_fix[col_index] == "<":
            to_fix = set_pos(to_fix, col_index, "c")
        if to_fix[col_index] == "P":
            to_fix = set_pos(to_fix, col_index, "f")
        if to_fix[col_index] == "L":
            to_fix = set_pos(to_fix, col_index, "h")
        to_fix = set_pos(to_fix, col_index, to_fix[col_index].lower())
        row_index = col_index + 1
        if len(to_fix) > row_index:
            if to_fix[row_index] == "b":
                to_fix = set_pos(to_fix, row_index, "6")
            if to_fix[row_index] == "G":
                to_fix = set_pos(to_fix, row_index, "6")
            if to_fix[row_index].lower() == "s":
                to_fix = set_pos(to_fix, row_index, "5")
            if to_fix[row_index].lower() == "z":
                to_fix = set_pos(to_fix, col_index, "2")

        return get_move_parts(to_fix)
    except Exception as err:
        print(f"err with {to_fix}: {err}")
        return None


def extract_from_table(header_row, table_rows):
    white_moves_seperate = []
    black_moves_seperate = []
    white_cols = []
    black_cols = []
    for col_index, col in enumerate(header_row):
        col = col.replace("'", "")
        if col.lower() == "white":
            white_cols.append(col_index)
            white_moves_seperate.append([])
        if col.lower() == "black" or col.lower() == "blac":
            black_cols.append(col_index)
            black_moves_seperate.append([])

    # Print body rows
    print("Table body data:")
    print(table_rows)

    for row in table_rows:
        for white_col_num, table_col_num in enumerate(white_cols):
            if (
                row[table_col_num]
                # and row[table_col_num] != ""
                # and row[table_col_num] != "''"
            ):
                print(
                    f"fixing {row[table_col_num]} from {white_col_num}, {table_col_num}"
                )
                corrected = fix_ocr_raw(row[table_col_num])
                print("raw vs corrected", row[table_col_num], corrected)
                white_moves_seperate[white_col_num].append(corrected)
            else:
                white_moves_seperate[white_col_num].append(None)
        for black_col_num, table_col_num in enumerate(black_cols):
            if (
                row[table_col_num]
                # and row[table_col_num] != ""
                # and row[table_col_num] != "''"
            ):
                black_moves_seperate[black_col_num].append(
                    fix_ocr_raw(row[table_col_num])
                )
            else:
                black_moves_seperate[black_col_num].append(None)

    print(white_moves_seperate)
    print(black_moves_seperate)
    return list(
        zip(
            chain.from_iterable(white_moves_seperate),
            chain.from_iterable(black_moves_seperate),
        )
    )


def get_amazon():
    client = boto3.client("textract")
    with open("test_image", "rb") as image:
        image_content = image.read()
    results = client.analyze_document(
        Document={
            "Bytes": image_content,
        },
        FeatureTypes=[
            "TABLES",
        ],
    )
    blocks = results["Blocks"]
    blocks_by_ids = dict()
    for block in blocks:
        blocks_by_ids[block["Id"]] = block
        if block["BlockType"] == "TABLE":
            table_id = block["Id"]

    header_row = []
    rows = []
    col_num = 0

    for cell_id in blocks_by_ids[table_id]["Relationships"][0]["Ids"]:
        if "EntityTypes" in blocks_by_ids[cell_id].keys():
            header_row.append(
                blocks_by_ids[blocks_by_ids[cell_id]["Relationships"][0]["Ids"][0]][
                    "Text"
                ]
            )
        else:
            if col_num == 0:
                rows.append([])
            if "Relationships" in blocks_by_ids[cell_id].keys():
                cell_text = blocks_by_ids[
                    blocks_by_ids[cell_id]["Relationships"][0]["Ids"][0]
                ]["Text"]
            else:
                cell_text = None
            rows[-1].append(cell_text)
            if len(header_row) == 0:
                return list()
            col_num = (col_num + 1) % len(header_row)

    print("Amazon raw")
    print(rows)
    return extract_from_table(header_row, rows)
