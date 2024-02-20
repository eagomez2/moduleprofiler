from datetime import datetime


class Logger:
    _TEXT_COLORS = {
        "red": "\033[91m",
        "green": "\033[92m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "magenta": "\033[35m",
        "yellow": "\033[93m",
        "end_color": "\033[0m",
    }

    _TEXT_DECORATORS = {
        "bold": "\033[1m",
        "italic": "\033[3m",
        "underline": "\033[4m",
        "end_decoration": "\033[0m",
    }

    _TEXT_DECORATOR_TAGS = {
        "<b>": _TEXT_DECORATORS["bold"],
        "</b>": _TEXT_DECORATORS["end_decoration"],
        "<i>": _TEXT_DECORATORS["italic"],
        "</i>": _TEXT_DECORATORS["end_decoration"],
        "<u>": _TEXT_DECORATORS["underline"],
        "</u>": _TEXT_DECORATORS["end_decoration"]
    }

    _TEXT_COLOR_TAGS = {
        "<error>": _TEXT_COLORS["red"],
        "</error>": _TEXT_COLORS["end_color"],
        "<warning>": _TEXT_COLORS["yellow"],
        "</warning>": _TEXT_COLORS["end_color"],
        "<green>": _TEXT_COLORS["green"],
        "</green>": _TEXT_COLORS["end_color"],
        "<magenta>": _TEXT_COLORS["magenta"],
        "</magenta>": _TEXT_COLORS["end_color"]
    }

    def __init__(self, ts_fmt: str = "%Y-%m-%d %H:%M:%S") -> None:
        super().__init__()

        # Params
        self.ts_fmt = ts_fmt

    def _decorate_str(self, s: str) -> str:
        # Replace colors and decorators
        for k, v in self._TEXT_DECORATOR_TAGS.items():
            s = s.replace(k, v)

        for k, v in self._TEXT_COLOR_TAGS.items():
            s = s.replace(k, v)

        return s

    def print(self, msg: str) -> str:
        print(self._decorate_str(msg))

    def log(self, msg: str) -> str:
        ts = f"[{datetime.now().strftime(self.ts_fmt)}]"
        msg = f"<b><green>{ts}</green></b> {msg}"
        self.print(msg)
