from pydantic import BaseModel


from openai import OpenAI
from dotenv import load_dotenv

from typing import Any
from typing import Optional, Sequence
import uuid
import datetime
import base64
import json
import re
import statistics
import itertools

from amazon import (
    MoveParts,
    get_move_parts,
    fix_ocr_raw,
    extract_from_table,
    get_amazon,
    get_white_black_cols,
)

from google.protobuf.json_format import MessageToDict

local = False

load_dotenv()

client = OpenAI()


class Turn(BaseModel):
    white: str
    black: str


class ChessGame(BaseModel):
    turns: list[Turn]


class ChessEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Turn):
            return {"white": self.encode(obj.white), "black": self.encode(obj.black)}
        if isinstance(obj, MoveParts):
            return {
                "piece": obj.piece,
                "source": obj.source,
                "capture": obj.capture,
                "rank": obj.rank,
                "file": obj.file,
                "check": obj.check,
                "ks_castle": obj.ks_castle,
                "qs_castle": obj.qs_castle,
            }
        return json.JSONEncoder.default(self, obj)


def feed_to_openai(image_base64: str) -> Any:
    completion = client.beta.chat.completions.parse(
        # model="gpt-4o-mini",
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": f"Transcribe Chess Game as PGN, but allow invalid moves",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        response_format=ChessGame,
        # logprobs=True,
        # top_logprobs=20,
    )
    if completion.choices[0].message.refusal:
        print(completion.choices[0].message.refusal)
    print(completion.choices[0].message.content)
    print(completion.choices)
    return completion.choices[0].message.parsed


import asyncio
import os
from flask import Flask, request
from cloudevents.http import from_http

app = Flask(__name__)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_path = "test_image"


def get_openai_moves(base64_image: str):
    openai_raw = feed_to_openai(base64_image).turns

    openai_white_moves = []
    openai_black_moves = []

    for turn in openai_raw:
        openai_white_moves.append(get_move_parts(turn.white))
        openai_black_moves.append(get_move_parts(turn.black))

    print("OpenAI moves:")
    openai_moves = list(zip(openai_white_moves, openai_black_moves))
    return openai_moves
    print(openai_moves)


if local:
    # Getting the base64 string
    base64_image = encode_image(image_path)
    openai_moves = get_openai_moves(base64_image)


from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore


# Google stuff apache 2.0: https://github.com/GoogleCloudPlatform/python-docs-samples/blob/HEAD/documentai/snippets/handle_response_sample.py


def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
    """
    Document AI identifies text in different parts of the document by their
    offsets in the entirety of the document"s text. This function converts
    offsets to a string.
    """
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    return "".join(
        text[int(segment.start_index) : int(segment.end_index)]
        for segment in layout.text_anchor.text_segments
    )


def array_rows(
    table_rows: Sequence[documentai.Document.Page.Table.TableRow], text: str
) -> list:
    out_table_rows = []
    for table_row in table_rows:
        row = []
        for cell in table_row.cells:
            print(cell)
            print(cell.layout.bounding_poly)
            # print(MessageToDict(cell.layout.bounding_poly))
            cell_text = layout_to_text(cell.layout, text)
            vertices = []
            for vertex in cell.layout.bounding_poly.vertices:
                vertices.append({"x": vertex.x, "y": vertex.y})
            row.append(
                (
                    f"{repr(cell_text.strip())}",
                    vertices,
                )
            )
        out_table_rows.append(row)

    return out_table_rows


def print_table_rows(
    table_rows: Sequence[documentai.Document.Page.Table.TableRow], text: str
) -> str:
    table_text = ""
    for table_row in table_rows:
        row_text = ""
        for cell in table_row.cells:
            cell_text = layout_to_text(cell.layout, text)
            row_text += f"{repr(cell_text.strip())} | "
        table_text += row_text + "\n"

    return table_text


def google_ocr(
    image_content: [bytes],
):
    processor_display_name: str = (
        "projects/965053369291/locations/us/processors/c522227897ac7837"
    )
    # You must set the `api_endpoint`if you use a location other than "us".
    opts = ClientOptions(api_endpoint=f"us-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # The full resource name of the location, e.g.:
    # `projects/{project_id}/locations/{location}`
    parent = client.common_location_path("965053369291", "us")

    # Load binary data
    raw_document = documentai.RawDocument(
        content=image_content,
        mime_type="image/jpeg",  # Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types
    )

    # Configure the process request
    # `processor.name` is the full resource name of the processor, e.g.:
    # `projects/{project_id}/locations/{location}/processors/{processor_id}`
    request = documentai.ProcessRequest(
        name=processor_display_name, raw_document=raw_document
    )

    result = client.process_document(request=request)

    # For a full list of `Document` object attributes, reference this page:
    # https://cloud.google.com/document-ai/docs/reference/rest/v1/Document
    document = result.document

    # Read the text recognition output from the processor
    # print(document)
    print("The document contains the following text:")
    print(document.text)
    text = document.text
    for page in document.pages:
        print(f"\n\n**** Page {page.page_number} ****")

        print(f"\nFound {len(page.tables)} table(s):")
        for table in page.tables:
            num_columns = len(table.header_rows[0].cells)
            num_rows = len(table.body_rows)
            print(f"Table with {num_columns} columns and {num_rows} rows:")

            # Print header rows
            print("Columns:")
            header_text = print_table_rows(table.header_rows, text)
            header_row = table.header_rows[:1]
            body_rows = table.body_rows

            row1_text = print_table_rows(body_rows[:1], text)
            print(body_rows)
            print(body_rows[0])
            print(body_rows[0].cells[0])
            if "white" in row1_text.lower() and "black" in row1_text.lower():
                header_row = body_rows[:1]
                print(row1_text)
                body_rows = body_rows[1:]
            elif (
                not "white" in header_text.lower() or not "black" in header_text.lower()
            ):
                if body_rows[0] and len(body_rows[0].cells) == 6:
                    header_row = [
                        "#",
                        "white",
                        "black",
                        "#",
                        "white",
                        "black",
                    ]  # guess the most common
                else:
                    print("failed to find header")
                    continue
            else:
                print(header_text)
            table_rows = array_rows(body_rows, text)
            table_rows = [
                row
                for row in table_rows
                # if statistics.fmean([len(cell) for cell in row]) <= 6
                if not ("white" in f"{row}".lower() and "black" in f"{row}".lower())
            ]  # Avoid picking up a bottom row like ["'RESULT'", "'WHITE'", "'WON'", "'DRAW'", "'BLACK'", "'WON'"]
            header_row = array_rows(header_row, text)[0]
            header_row = [cell[0] for cell in header_row]
            print(header_row)
            cell_bounds = []
            print("get_white_black_cols", get_white_black_cols(header_row, table_rows))
            print("table rows", table_rows)
            for white_col, black_col in zip(
                *get_white_black_cols(header_row, table_rows)
            ):
                cell_bounds.extend(
                    itertools.chain.from_iterable(
                        [[row[white_col][1], row[black_col][1]] for row in table_rows]
                    )
                )
            print("cell bounds", cell_bounds)
            table_rows = [[cell[0] for cell in row] for row in table_rows]

            return extract_from_table(header_row, table_rows), cell_bounds
    return [], []


if local:
    image_raw = base64.b64decode(base64_image)
    google_moves, cell_bounds = google_ocr(image_raw)
    print("Google moves:")
    print(google_moves)
    print(
        json.dumps(
            {
                "cell_bounds": cell_bounds,
            },
            cls=ChessEncoder,
        )
    )


@app.route("/", methods=["POST"])
def render_dom():
    # print(request)
    # print(request.get_data())
    request_contents = request.get_data().decode("utf-8")
    # contents = feed_to_openai(request_contents)
    openai_moves = get_openai_moves(request_contents)
    image_raw = base64.b64decode(request_contents)
    google_moves, cell_bounds = google_ocr(image_raw)
    max_move = len(openai_moves)
    for num, google_move in enumerate(google_moves):
        if google_move:
            if num + 1 > max_move:
                max_move += 1
                openai_moves.append([None, None])
    amazon_moves = get_amazon(image_raw)
    for num, amazon_move in enumerate(amazon_moves):
        if amazon_move:
            if num + 1 > max_move:
                max_move += 1
                openai_moves.append([None, None])
    print(
        json.dumps(
            {
                "openai": openai_moves,
                "google": google_moves,
                "amazon": amazon_moves,
                "cell_bounds": cell_bounds,
            },
            cls=ChessEncoder,
        )
    )
    return (
        json.dumps(
            {
                "openai": openai_moves,
                "google": google_moves,
                "amazon": amazon_moves,
                "cell_bounds": cell_bounds,
            },
            cls=ChessEncoder,
        ),
        200,
    )


if local:
    amazon_moves = get_amazon(image_raw)
    print(amazon_moves)

    max_move = len(openai_moves)
    for num, google_move in enumerate(google_moves):
        if google_move and google_move != ["''", "''"] and google_move != (None, None):
            if num + 1 > max_move:
                print("g extending due to ", google_move)
                max_move += 1
                openai_moves.append([None, None])
    for num, amazon_move in enumerate(amazon_moves):
        if amazon_move and amazon_move != (None, None):
            if num + 1 > max_move:
                print("a extending due to ", amazon_move)
                max_move += 1
                openai_moves.append([None, None])
    # print("ocr moves:")
    # print(json.dumps(openai_moves, cls=ChessEncoder))
    # print(json.dumps(google_moves, cls=ChessEncoder))
    # print(json.dumps(amazon_moves, cls=ChessEncoder))
    for turn_number, turn in enumerate(openai_moves):
        if len(google_moves) > turn_number and google_moves[turn_number][0]:
            google_move_white = google_moves[turn_number][0].to_play()
        else:
            google_move_white = ""
        if len(amazon_moves) > turn_number and amazon_moves[turn_number][0]:
            amazon_move_white = amazon_moves[turn_number][0].to_play()
        else:
            amazon_move_white = ""
        if turn[0]:
            openai_move_white = turn[0].to_play()
        else:
            openai_move_white = ""
        if len(google_moves) > turn_number and google_moves[turn_number][1]:
            google_move_black = google_moves[turn_number][1].to_play()
        else:
            google_move_black = ""
        if len(amazon_moves) > turn_number and amazon_moves[turn_number][1]:
            amazon_move_black = amazon_moves[turn_number][1].to_play()
        else:
            amazon_move_black = ""
        if turn[1]:
            openai_move_black = turn[1].to_play()
        else:
            openai_move_black = ""
        print(
            f"{turn_number + 1}:\n\tg ({google_move_white}) \t ({google_move_black})\n\ta ({amazon_move_white}) \t ({amazon_move_black}) \n\toa ({openai_move_white}) \t ({openai_move_black})\n\n"
        )
# arbitrate_moves(openai_moves, google_moves, amazon_moves)

# if __name__ == "__main__":
# app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
