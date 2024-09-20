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

from amazon import (
    MoveParts,
    get_move_parts,
    fix_ocr_raw,
    extract_from_table,
    get_amazon,
)

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
                "content": f"transcribe the chess game",
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


# Path to your image
image_path = "test_image"

# Getting the base64 string
base64_image = encode_image(image_path)
openai_raw = feed_to_openai(base64_image).turns

openai_white_moves = []
openai_black_moves = []

for turn in openai_raw:
    openai_white_moves.append(get_move_parts(turn.white))
    openai_black_moves.append(get_move_parts(turn.black))

print("OpenAI moves:")
openai_moves = list(zip(openai_white_moves, openai_black_moves))
print(openai_moves)

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
            cell_text = layout_to_text(cell.layout, text)
            row.append(f"{repr(cell_text.strip())}")
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
    file_path: str,
):
    processor_display_name: str = (
        "projects/424251658231/locations/us/processors/3f540b5a91f4d91e"
    )
    # You must set the `api_endpoint`if you use a location other than "us".
    opts = ClientOptions(api_endpoint=f"us-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # The full resource name of the location, e.g.:
    # `projects/{project_id}/locations/{location}`
    parent = client.common_location_path("424251658231", "us")

    """
        # Create a Processor
        processor = client.create_processor(
            parent=parent,
            processor=documentai.Processor(
                type_="OCR_PROCESSOR",  # Refer to https://cloud.google.com/document-ai/docs/create-processor for how to get available processor types
                display_name=processor_display_name,
            ),
        )

        # Print the processor information
        print(f"Processor Name: {processor.name}")
    """

    # Read the file into memory
    with open(file_path, "rb") as image:
        image_content = image.read()

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
            if not "white" in header_text.lower() or not "black" in header_text.lower():
                header_row = None
                header_text = print_table_rows(body_rows[:1], text)
                if "white" in header_text.lower() and "black" in header_text.lower():
                    header_row = body_rows[:1]
                    print(header_text)
                    body_rows = body_rows[1:]
            else:
                print(header_text)
            table_rows = array_rows(body_rows, text)

            header_row = array_rows(header_row, text)[0]
            print(header_row)

            return extract_from_table(header_row, table_rows)


google_moves = google_ocr(image_path)
print("Google moves:")
print(google_moves)


@app.route("/", methods=["POST"])
def render_dom():
    # print(request)
    # print(request.get_data())
    request_contents = request.get_data().decode("utf-8")
    contents = feed_to_openai(request_contents)
    print(contents.moves)
    print(json.dumps(contents.turns, cls=ChessEncoder))
    return (
        json.dumps(contents.turns, cls=ChessEncoder),
        200,
    )


amazon_white_moves = [None] * len(openai_moves)
amazon_black_moves = [None] * len(openai_moves)
amazon_black_moves[2] = MoveParts("B", None, None, "c", None, None, False, False)
amazon_moves = get_amazon()  # list(zip(amazon_white_moves, amazon_black_moves))
print(amazon_moves)

print("ocr moves:")
print(json.dumps(openai_moves, cls=ChessEncoder))
print(json.dumps(google_moves, cls=ChessEncoder))
print(json.dumps(amazon_moves, cls=ChessEncoder))
# arbitrate_moves(openai_moves, google_moves, amazon_moves)

# if __name__ == "__main__":
# app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
