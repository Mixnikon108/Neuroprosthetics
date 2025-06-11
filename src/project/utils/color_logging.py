# color_logging.py
import logging, colorama
import datetime

RESET = "\033[0m"
DIM   = "\033[2m"
BOLD  = "\033[1m"
FG = {
    "grey":   "\033[38;5;245m",
    "cyan":   "\033[36m",
    "yellow": "\033[33m",
    "red":    "\033[31m",
    "green":  "\033[32m",
}
LEVEL_COLOR = {
    "DEBUG":    FG["cyan"],
    "INFO":     FG["green"],
    "WARNING":  FG["yellow"],
    "ERROR":    FG["red"],
    "CRITICAL": BOLD + FG["red"],
}

class ColorFormatter(logging.Formatter):
    # Le pasamos un datefmt con %f para microsegundos
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def formatTime(self, record, datefmt=None):
        # Usamos datetime para que %f funcione
        dt = datetime.datetime.fromtimestamp(record.created)
        fmt = datefmt or self.datefmt or "%Y-%m-%d %H:%M:%S.%f"
        s = dt.strftime(fmt)
        # s tiene 6 dígitos de microsegundos; truncamos a 3 para ms
        if "%f" in fmt:
            return s[:-3]
        else:
            return s

    def format(self, record):
        # Llama primero a formatTime vía super().format()
        msg = super().format(record)

        # — timestamp en gris tenue —
        msg = msg.replace(
            record.asctime,
            f"{DIM}{record.asctime}{RESET}", 1)

        # — módulo centrado y gris —
        original_name = record.name.ljust(12)
        centered_name = record.name.center(12)
        colored_name  = f"{BOLD}{FG['grey']}{centered_name}{RESET}"
        msg = msg.replace(original_name, colored_name, 1)

        # — nivel right‑justified y coloreado —
        original_level = record.levelname.rjust(8)
        colored_level  = f"{LEVEL_COLOR.get(record.levelname,'')}{original_level}{RESET}"
        msg = msg.replace(original_level, colored_level, 1)

        return msg

def setup_color_logging(level=logging.INFO):
    colorama.just_fix_windows_console()

    LOG_FMT  = "%(asctime)s [ %(name)-12s ] %(levelname)8s| %(message)s"
    # Agregamos .%f para microsegundos (luego los truncamos)
    DATE_FMT = "%Y-%m-%d %H:%M:%S.%f"

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter(LOG_FMT, datefmt=DATE_FMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)
