import boto3

from typing import Optional, Sequence
from itertools import chain
import re
import os


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
        piece: Optional[str] = None,
        source: Optional[str] = None,
        capture: Optional[str] = None,
        rank: Optional[str] = None,
        file: Optional[str] = None,
        check: Optional[str] = None,
        ks_castle: bool = False,
        qs_castle: bool = False,
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
    if len(to_extract.split(" ")) != 1:
        return extract_groups(to_extract.split(" ")[-1])
    matches = re.compile(r"^([QRBKN])?([a-h]?)(x?)([a-h])([0-9])(\+?)#?$").match(
        to_extract
    )
    if not matches:
        matches = re.compile(r"^([QRBKN])?([a-h]?)(x?)([a-h])([0-9]?)(\+?)#?$").match(
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
        if to_fix[col_index] == "(":
            to_fix = set_pos(to_fix, col_index, "c")
        to_fix = set_pos(to_fix, col_index, to_fix[col_index].lower())
        row_index = col_index + 1
        if len(to_fix) > row_index:
            if to_fix[row_index] == "b":
                to_fix = set_pos(to_fix, row_index, "6")
            if to_fix[row_index] == "G":
                to_fix = set_pos(to_fix, row_index, "6")
            if to_fix[row_index].lower() == "s":
                to_fix = set_pos(to_fix, row_index, "5")
            if to_fix[row_index].lower() == "y":
                to_fix = set_pos(to_fix, row_index, "4")
            if to_fix[row_index].lower() == "z":
                to_fix = set_pos(to_fix, col_index, "2")

        return get_move_parts(to_fix)
    except Exception as err:
        print(f"err with {to_fix}: {err}")
        return None


def get_white_black_cols(header_row, table_rows):
    white_cols = []
    black_cols = []
    for col_index, col in enumerate(header_row):
        if col == None:
            continue
        col = col.replace("'", "")
        if "white" in col.lower():
            white_cols.append(col_index)
        if "blac" in col.lower():
            black_cols.append(col_index)

    if (
        len(white_cols) == 0
        and len(black_cols) == 0
        and table_rows
        and table_rows[0]
        and len(table_rows[0]) == 6
    ):
        return get_white_black_cols(
            header_row=[
                "#",
                "white",
                "black",
                "#",
                "white",
                "black",
            ],  # guess the most common
            table_rows=table_rows,
        )
    if (
        len(white_cols) == 0
        and len(black_cols) == 0
        and table_rows
        and table_rows[0]
        and len(table_rows[0]) == 4
    ):
        return get_white_black_cols(
            header_row=[
                "white",
                "black",
                "white",
                "black",
            ],  # guess the most common
            table_rows=table_rows,
        )

    return white_cols, black_cols


def extract_from_table(header_row, table_rows):
    white_moves_seperate = []
    black_moves_seperate = []
    if not table_rows or len(table_rows) == 0:
        return []

    white_cols, black_cols = get_white_black_cols(header_row, table_rows)
    for white_col in white_cols:
        white_moves_seperate.append([])
    for black_col in black_cols:
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


access_key = os.environ.get("AWS_ACCESS_KEY_ID")
secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
region = "us-east-1"


def get_amazon(image_content: [bytes]):
    client = boto3.client(
        "textract",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
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
    table_ids = []
    for block in blocks:
        blocks_by_ids[block["Id"]] = block
        if block["BlockType"] == "TABLE":
            table_ids.append(block["Id"])

    final_header = []
    sortable_rows = []
    print("table ids:", table_ids)
    for table_id in table_ids:
        header_row = []
        rows = []
        col_num = 0

        for cell_id in blocks_by_ids[table_id]["Relationships"][0]["Ids"]:
            print(blocks_by_ids[cell_id])
            if "EntityTypes" in blocks_by_ids[cell_id].keys():
                if "Relationships" in blocks_by_ids[cell_id].keys():
                    cell_text = blocks_by_ids[
                        blocks_by_ids[cell_id]["Relationships"][0]["Ids"][0]
                    ]["Text"]
                else:
                    cell_text = None
                header_row.append(cell_text)
            else:
                if "Relationships" in blocks_by_ids[cell_id].keys():
                    cell_text = blocks_by_ids[
                        blocks_by_ids[cell_id]["Relationships"][0]["Ids"][0]
                    ]["Text"]
                else:
                    cell_text = None
                print(cell_text)
                if len(header_row) == 0:
                    header_row.append(cell_text)
                    continue
                if col_num == 0:
                    rows.append([])
                rows[-1].append(cell_text)
                col_num = (col_num + 1) % len(header_row)
        is_relevant_table = False
        print("header row:", header_row)
        for col_index, col in enumerate(header_row):
            if col == None:
                continue
            col = col.replace("'", "")
            if (
                col.lower() == "white"
                or col.lower() == "black"
                or col.lower() == "blac"
            ):
                is_relevant_table = True
                break
        if not is_relevant_table:
            continue
        final_header = header_row
        sort_score = 0
        for row in rows:
            if len(row) == 0:
                continue
            if row[0] == None:
                continue
            sort_score += len(row[0])
        sortable_rows.append((sort_score, rows))

    final_rows = []
    for _, rows in sorted(sortable_rows):
        final_rows.extend(rows)

    print("Amazon raw")
    print(final_rows)
    return extract_from_table(header_row, final_rows)
