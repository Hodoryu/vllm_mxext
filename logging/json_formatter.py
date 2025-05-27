import json
from json.decoder import JSONDecodeError
from logging import Formatter, LogRecord
from typing import Any, Dict, Literal


class ExceptionAttributeFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        """
        Instead of appending exception info and stack info to the logging message, 'exc_text' and 'exc_stack'
        are kept as attributes of the record for further formatting.
        """
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        s = self.formatMessage(record)
        return s


class JsonFormatter(Formatter):
    """
    Formatter that outputs JSON strings after parsing the LogRecord. Unlike in `logging.Formatter`,
    exception info and stack info are not append to the message. Instead, format placeholders
    `exc_text` and `stack_info` must be used.
    Args:
        fmt (str | None): logging format. It must be a valid JSON string.
            The values of the JSON string can contain placeholders for interpolation with style `style`.
            There is a limitation: values of JSON must be strings. The restriction is needed because
            verification of the JSON structure is difficult if values are allowed to be float/int/boolean.
            Defaults to '{"message": "%(message)s", "exc_info": "%(exc_text)s", "stack_info": "%(stack_info)s"}'.
        datefmt (str | None): time.strftime() format string. If `None`, then the format is "%Y-%m-%dT%H:%M:%S"
        style (Literal["%", "{", "$"]): a format as defined in logging/__init__.py `_STYLES` constant.
            Defaults to  "%"
        validate (bool): If True (the default), incorrect or mismatched fmt and style will raise a ValueError
        defaults (Dict[str, Any]): A dictionary with default values to use in custom fields
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Literal["%", "{", "$"] = '%',
        validate: bool = True,
        *,
        defaults: Dict[str, Any] | None = None,
    ):
        raw_fmt = (
            '{"message": "%(message)s", "exc_info": "%(exc_text)s", "stack_info": "%(stack_info)s"}'
            if fmt is None
            else fmt
        )
        if style == "{":
            raw_fmt = raw_fmt.replace("{{", "{").replace("}}", "}")
        try:
            parsed_fmt = json.loads(raw_fmt)
        except JSONDecodeError as e:
            raise JSONDecodeError(
                f"Cannot parse format of logging used in JsonFormatter. "
                f"The format must be a valid JSON string. "
                f"The faulty logging format:\n{fmt}\n",
                doc=e.doc,
                pos=e.pos,
            )
        self._check_parsed_format(parsed_fmt, fmt)
        self._formatters_by_fields = {
            msg_field: ExceptionAttributeFormatter(
                fmt=msg_fmt, datefmt=datefmt, style=style, validate=validate, defaults=defaults
            )
            for msg_field, msg_fmt in parsed_fmt.items()
        }

    def _check_parsed_format(self, parsed_fmt: Any, orig_fmt: str) -> None:
        if not isinstance(parsed_fmt, dict):
            raise ValueError(
                f"jsonl logging format must be a dictionary, but got {type(parsed_fmt)}. "
                f"The faulty logging format:\n{orig_fmt}\n"
            )
        for key, value in parsed_fmt.items():
            if not isinstance(value, str):
                raise ValueError(
                    f"jsonl logging format values must be strings, but got {type(value)} "
                    f"for field '{key}'. "
                    f"The faulty logging format:\n{orig_fmt}\n"
                )

    def format(self, record: LogRecord) -> str:
        """
        Create a message from `record` according to predefined format `fmt`. The difference with the
        base class method is that exception info and stack info are not appended to the message.
        """
        message_dict = {
            msg_field: formatter.format(record) for msg_field, formatter in self._formatters_by_fields.items()
        }
        return json.dumps(message_dict)
