# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Tuple

# Please sort it alphabetically
_static_files = (
    ("/css/bootstrap.min.css", "text/css"),
    ("/css/bootstrap.min.css.map", "text/css"),
    ("/index.html", "text/html"),
    ("/js/bootstrap.min.js", "application/javascript"),
    ("/js/bootstrap.min.js.map", "application/javascript"),
    ("/js/jquery-3.6.0.min.js", "application/javascript"),
    ("/js/offline_record.js", "application/javascript"),
    ("/js/offline_record.js", "application/javascript"),
    ("/js/popper.min.js", "application/javascript"),
    ("/js/popper.min.js.map", "application/javascript"),
    ("/js/streaming_record.js", "application/javascript"),
    ("/js/upload.js", "application/javascript"),
    ("/k2-logo.png", "image/png"),
    ("/nav-partial.html", "text/html"),
    ("/offline_record.html", "text/html"),
    ("/streaming_record.html", "text/html"),
    ("/upload.html", "text/html"),
)

_404_page = r"""
<!doctype html><html><head>
<title>Speech recognition with next-gen Kaldi</title><body>
<h1>404 ERROR! Please re-check your URL</h1>
</body></head></html>
"""


def read_file(root: str, name: str) -> str:
    try:
        with open(f"{root}/{name}") as f:
            return f.read()
    except:  # noqa
        with open(f"{root}/{name}", "rb") as f:
            return f.read()


class HttpServer:
    """
    A simple HTTP server that hosts only static files
    """

    def __init__(self, doc_root: str):
        content = dict()
        for f, mime_type in _static_files:
            content[f] = (read_file(doc_root, f), mime_type)
        self.content = content

    def process_request(self, f: str) -> Tuple[str, str, str]:
        """
        Args:
          f:
            The filename to read.
        Returns:
          Return a tuple:
            - a bool, True if the given file is found. False otherwise.
            - a str, the content of the file if found. Otherwise, it
              contains the content for the 404 page
            - a str, the MIME type of the returned content
        """
        if f in self.content:
            return True, self.content[f][0], self.content[f][1]
        else:
            return False, _404_page, "text/html"
