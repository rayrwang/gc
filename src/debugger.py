
"""
     0                       1100      1500       2000       2500
   0 ┌────────────────────────┬─────────┬─────────────────────┐
     │                        │         │         Col         │
     │                        │         │       Overview      │
     │                        │         │          │          │
     │                        │         │          │          │
     │          Cols          │  Conns  │          │          │
     │          Grid          │   or    │  Activ-  │ Weights  │
     │                        │ Activity│  ations  │          │
     │                        │  Values │          │          │
     │                        │         │          │          │
     │                        │         │          │          │
1100 ├────────────────────────┤         │          │          │
     │     Global Overview    │         │          │          │
1300 └────────────────────────┴─────────┴──────────┴──────────┘
"""

import ast
import math
import os
import signal
import sys
import time

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = ""
import pygame as pg

from .agents import Dir

# Draw on 2500 x 1300 virtual window, then scale to size of real window
W, H = (2500, 1300)

LINE_HEIGHT = 30

dir2pos = {
    Dir.A: "top",
    Dir.E: "bottom",
}


def debugger(PATH, pipes):
    # Suppress keyboard interrupt traceback
    signal.signal(signal.SIGINT, lambda _, __: sys.exit(0))

    Debugger(PATH, pipes).run()


def screen2loc(x, y, width):
    """Convert screen coordinates to col coordinates (loc)"""
    return int(x/width), int(y/width)


def screen2dir(x, y, width):
    """
    top    : Dir.A
    bottom : Dir.E
    """
    x, y = x % width, y % width
    if y <= width/2:
        return Dir.A
    else:
        return Dir.E


def get_color(x):
    if math.isnan(x):  # Cyan
        return (0, 255, 255)
    x = min(max(x, -2), 2)  # Clip to within [-2, 2]
    x = round(x*255/2)  # [-2, 2] -> [-255, 255]
    # Red for negative, white for 0, green for positive
    color = (255-max(0,x), 255+min(0,x), 255-abs(x))
    return color


def parse_loc(name):
    """Parse a col directory name into a loc, or None if it isn't one
    (so stray files like .DS_Store or partial writes are skipped)."""
    if name in ("type", "cfg", "cfg_type"):
        return None
    try:
        loc = ast.literal_eval(name)
    except (ValueError, SyntaxError):
        return None
    return loc if isinstance(loc, tuple) and len(loc) == 2 else None


class Debugger:
    def __init__(self, path, pipes):
        self.path = path
        self.pipes = pipes

        # Calculate size of rendered grid cells according to distance spanned by grid
        x_min, x_max = float("inf"), float("-inf")
        y_min, y_max = float("inf"), float("-inf")
        for name in os.listdir(path):
            loc = parse_loc(name)
            if loc is not None:
                x, y = loc
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y
        dist_spanned = max(x_max - x_min, y_max - y_min)
        self.COL_WIDTH = 1100 // (dist_spanned + 1)

        self.window = pg.Surface((W, H))

        pg.init()
        (desktop_w, desktop_h), = pg.display.get_desktop_sizes()
        self.scale = 0.8 * min(desktop_w/2560, desktop_h/1440)  # Fit to display
        self.true_window = pg.display.set_mode((self.scale*W, self.scale*H), pg.DOUBLEBUF|pg.RESIZABLE)
        pg.display.set_caption(f"debugger - {os.path.basename(path.rstrip('/'))}")

        # Frequently used graphical elements
        # Half size grey highlight for showing conns
        conn_highlight = pg.Surface((self.COL_WIDTH, self.COL_WIDTH/2))
        conn_highlight.fill((200,200,200))
        # Half size blue border for selecting conn
        conn_select = pg.Surface((self.COL_WIDTH, self.COL_WIDTH/2))
        conn_select.fill((255,255,255))
        conn_select.set_colorkey((255,255,255))
        pg.draw.rect(conn_select, (100,149,237), conn_select.get_rect(), width=int(self.COL_WIDTH*0.05))
        self.templates = {
            "conn_highlight": conn_highlight,
            "conn_select": conn_select
        }

        pg.font.init()
        self.fonts = {
            "col": pg.font.SysFont("Helvetica", int(0.3*self.COL_WIDTH)),
            "big": pg.font.SysFont("Helvetica", 48),
            "med_bold": pg.font.SysFont("Helvetica", 36, True),
            "debug": pg.font.SysFont("Helvetica", 24),
            "small": pg.font.SysFont("Helvetica", 12)
        }

        # Cache of which items user selected
        self.gui_state = {
            "loc": None,
            "conn": None,
            "atv": None,
        }

        # Cache of info received from agent
        self.cache = {
            "overview": None,
            "col": None,
            "conn": None,
            "atv": None,
        }

    def draw_col(self, loc, highlight=None, colors=None):
        x, y = loc

        col = pg.Surface((self.COL_WIDTH, self.COL_WIDTH))
        col.fill((255,255,255))
        col.set_colorkey((255,255,255))  # so it's possible to draw different components (background, highlight, border) at separate times

        if colors is not None:
            for i, color in enumerate(colors):
                px_y = i * self.COL_WIDTH / len(colors)
                pg.draw.rect(col, color, (0, px_y, self.COL_WIDTH, self.COL_WIDTH/len(colors)))

        conn_highlight, conn_select = self.templates["conn_highlight"], self.templates["conn_select"]
        if highlight == "top":
            col.blit(conn_highlight, (0, 0))
        elif highlight == "bottom":
            col.blit(conn_highlight, (0, self.COL_WIDTH/2))

        pg.draw.rect(col, (0,0,0), (0, 0, self.COL_WIDTH, self.COL_WIDTH), 1)  # Outer (thin) border
        txt = self.fonts["col"].render(f"{x},{y}", True, (0,0,0))
        txt_rect = txt.get_rect(center=(self.COL_WIDTH/2, self.COL_WIDTH/2))
        col.blit(txt, txt_rect)

        if highlight == "border":
            pg.draw.rect(col, (0, 0, 0), col.get_rect(), width=int(self.COL_WIDTH*0.05))
        elif highlight == "bordertop":
            col.blit(conn_select, (0, 0))
        elif highlight == "borderbottom":
            col.blit(conn_select, (0, self.COL_WIDTH/2))

        self.window.blit(col, (x*self.COL_WIDTH, y*self.COL_WIDTH))

    def get_histogram(self, h, bin_width, has_nan, all_nan, h_e=None):
        """
        Histogram: The histogram displays approximately from -2 to +2 standard deviations,
        with 43 bins in total, where each bin is 1/10 of a std wide.

        For example if the std is 1 (and so each bin is 0.1 wide), the bins are:
        (-inf, -2.05), [-2.05, -1.95), ..., [-0.05, 0.05), ..., [1.95, 2.05), [2.05, inf)

        For activation tensors the bins are fixed at 0.1 wide,
        but for weights it is computed by rounding std/10 to 1eX, 2eX, or 5eX.

        h_e is only used for activation tensors
        """
        # Draw border and markings
        histogram = pg.Surface((100+215, 15+133+15))
        histogram.fill((255,255,255))
        histogram.set_colorkey((255,255,255))
        pg.draw.rect(histogram, (0,0,0), (100+0, 15+0, 215, 133), 1)  # Border

        pg.draw.line(histogram, (100,149,237), (100+7.5, 15+133-10), (100+7.5, 15+133+5), width=2)      # -2 std tick
        txt = self.fonts["small"].render(f"-{20*bin_width:g}", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+7.5, 15+133)))

        pg.draw.line(histogram, (100,149,237), (100+57.5, 15+133-10), (100+57.5, 15+133+5), width=2)    # -1 std tick
        txt = self.fonts["small"].render(f"-{10*bin_width:g}", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+57.5, 15+133)))

        pg.draw.line(histogram, (100,149,237), (100+107.5, 15+133-10), (100+107.5, 15+133+5), width=2)  # 0 tick
        txt = self.fonts["small"].render("0", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+107.5, 15+133)))

        pg.draw.line(histogram, (100,149,237), (100+157.5, 15+133-10), (100+157.5, 15+133+5), width=2)  # +1 std tick
        txt = self.fonts["small"].render(f"{10*bin_width:g}", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+157.5, 15+133)))

        pg.draw.line(histogram, (100,149,237), (100+207.5, 15+133-10), (100+207.5, 15+133+5), width=2)  # +2 std tick
        txt = self.fonts["small"].render(f"{20*bin_width:g}", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+207.5, 15+133)))

        if not all_nan:  # TODO distinguish between h and h_e?
            # Draw histogram h in black
            txt = self.fonts["small"].render("0", True, (0,0,0))
            histogram.blit(txt, txt.get_rect(midright=(100, 15+133)))
            txt = self.fonts["small"].render(f"{int(max(h)):,}", True, (0,0,0))
            histogram.blit(txt, txt.get_rect(midright=(100, 15+0)))
            for i, num in enumerate(h):
                height = round(133*(num/max(h)))
                pg.draw.rect(histogram, (0,0,0), (100+i*5, 15+133-height, 5, height))  # Opaque, black bars

            # Draw histogram h_e in light blue
            if h_e is not None:
                max_height = max(h_e)
                txt = self.fonts["small"].render(f"{int(max_height):,}", True, (100,100,255))
                histogram.blit(txt, txt.get_rect(midright=(100, 15+15)))
                surface = pg.Surface(histogram.get_size(), pg.SRCALPHA)
                surface.fill((255, 255, 255))
                surface.set_colorkey((255, 255, 255))
                for i, num in enumerate(h_e):
                    height_e = round(133*(num/max_height))
                    pg.draw.rect(surface, (200, 220, 255, 180), (100+i*5, 15+133-height_e, 5, height_e))  # Translucent, light blue bars
                histogram.blit(surface, (0, 0))

        # Draw NaN warning
        if has_nan and not all_nan:
            txt = self.fonts["med_bold"].render("Has NaN", True, (255,0,0))
            histogram.blit(txt, txt.get_rect(center=(100+(215/2), 15+(133/2))))
        elif all_nan:
            txt = self.fonts["med_bold"].render("All NaN", True, (255,0,0))
            histogram.blit(txt, txt.get_rect(center=(100+(215/2), 15+(133/2))))
        return histogram

    def draw_static_layout(self):
        """Draw the fixed panel dividers and section headers."""
        self.window.fill((255, 255, 255))

        # Horizontal lines
        pg.draw.line(self.window, (0, 0, 0), (0, 1100), (1100, 1100))

        # Vertical lines
        pg.draw.line(self.window, (0, 0, 0), (1100, 0), (1100, 1300))
        pg.draw.line(self.window, (0, 0, 0), (1500, 0), (1500, 1300))
        pg.draw.line(self.window, (0, 0, 0), (2000, 200), (2000, 1300))

        txt = self.fonts["big"].render("Activations:", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(midbottom=(1750, 220)))
        txt = self.fonts["big"].render("Weights:", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(midbottom=(2250, 220)))

    def draw_cols(self):
        """Draw the grid of cols read off disk."""
        for name in os.listdir(self.path):
            loc = parse_loc(name)
            if loc is not None:
                self.draw_col(loc)

    def draw_overview(self):
        """Draw the global overview panel (bottom left)."""
        _, pipe = self.pipes["overview"]
        # Try to get new info
        info = None
        while pipe.poll():
            new_info = pipe.recv()
            info = new_info
            self.cache["overview"] = new_info

        # If no new info, try to read from cache
        if info is None and self.cache["overview"] is not None:
            info = self.cache["overview"]

        if info is None:
            txt = self.fonts["debug"].render("waiting...", True, (0,0,0))
            self.window.blit(txt, (25, 1125))
        else:
            txt = self.fonts["debug"].render("Active", True, (0,0,0))
            self.window.blit(txt, (25, 1125))

            # Age of information
            age = time.time() - info["timestamp"]
            txt = self.fonts["debug"].render(f"age: {age:.3f}s", True, (0,0,0))
            self.window.blit(txt, (25, 1125+1*LINE_HEIGHT))

            memory = 2 * (2*info["copies"]*info["nrns"] + info["syns"])
            #        ^ 2 bytes per element
            #             ^ current and new versions of activations
            #               ^ "copies" copies of activations for each version
            memory_gb = memory / 1e9
            txt = self.fonts["debug"].render("Memory:", True, (0,0,0))
            self.window.blit(txt, (25, 1125+3*LINE_HEIGHT))
            txt = self.fonts["debug"].render(f"{memory_gb:.2f} GB", True, (0,0,0))
            self.window.blit(txt, (25, 1125+4*LINE_HEIGHT))

            # Total number of activations and weights
            txt = self.fonts["debug"].render("# of:", True, (0,0,0))
            self.window.blit(txt, (200, 1125+0*LINE_HEIGHT))
            txt = self.fonts["debug"].render(f"    activations: {info["nrns"]:,} ({info["copies"]} copies)", True, (0,0,0))
            self.window.blit(txt, (200, 1125+1*LINE_HEIGHT))
            txt = self.fonts["debug"].render(f"    total weights: {info["syns"]:,}", True, (0,0,0))
            self.window.blit(txt, (200, 1125+2*LINE_HEIGHT))
            txt = self.fonts["debug"].render(f"    |-- internal weights: {info["isyns"]:,}", True, (0,0,0))
            self.window.blit(txt, (200, 1125+3*LINE_HEIGHT))
            txt = self.fonts["debug"].render(f"    |-- external weights: {info["esyns"]:,}", True, (0,0,0))
            self.window.blit(txt, (200, 1125+4*LINE_HEIGHT))

            txt = self.fonts["debug"].render("ratios:", True, (0,0,0))
            self.window.blit(txt, (650, 1125+0*LINE_HEIGHT))
            txt = self.fonts["debug"].render(f"{info["syns"]/info["nrns"]:,.2f} to 1", True, (0,0,0))
            self.window.blit(txt, (650, 1125+2*LINE_HEIGHT))
            txt = self.fonts["debug"].render(f"|-- {info["isyns"]/info["nrns"]:,.2f} to 1", True, (0,0,0))
            self.window.blit(txt, (650, 1125+3*LINE_HEIGHT))
            txt = self.fonts["debug"].render(f"|-- {info["esyns"]/info["nrns"]:,.2f} to 1", True, (0,0,0))
            self.window.blit(txt, (650, 1125+4*LINE_HEIGHT))

            txt = self.fonts["debug"].render(f"density: {info["density"]*100:.2f}%", True, (0,0,0))
            self.window.blit(txt, (875, 1125+0*LINE_HEIGHT))

    def handle_events(self):
        """Process the pygame event queue and mouse/keyboard input,
        updating gui_state selection and the (rebound) window scale."""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.pipes["overview"][1].send(None)
                sys.exit(0)
            elif event.type == pg.VIDEORESIZE:
                w_new, h_new = event.size
                if w_new/h_new < W/H:
                    self.scale = w_new / W
                else:
                    self.scale = h_new / H
        buttons = pg.mouse.get_pressed(num_buttons=3)
        screen_x, screen_y = pg.mouse.get_pos()
        screen_x, screen_y = screen_x / self.scale, screen_y / self.scale
        keys = pg.key.get_pressed()
        if buttons[0]:  # Left click
            # Get which col and conn mouse is clicking on
            # Select col
            loc = screen2loc(screen_x, screen_y, self.COL_WIDTH)
            self.gui_state["loc"] = loc
            self.gui_state["conn"] = None
            self.gui_state["atv"] = None
        if keys[pg.K_ESCAPE]:
            self.gui_state["loc"] = None
            self.gui_state["conn"] = None
            self.gui_state["atv"] = None

        if self.gui_state["loc"] and buttons[2]:  # Right click on a selected col
            # Try to stay on same col and select conn or activation layer
            conn_loc = screen2loc(screen_x, screen_y, self.COL_WIDTH)
            conn_dir = screen2dir(screen_x, screen_y, self.COL_WIDTH)
            if os.path.isdir(f"{self.path}/{conn_loc}"):  # Check if conn location is valid
                self.gui_state["atv"] = None
                self.gui_state["conn"] = (conn_loc, conn_dir)
            else:
                self.gui_state["conn"] = None
                if 1500 < screen_x < 2000:  # Try to select layer of activations
                    if 250+0*LINE_HEIGHT <= screen_y < 250+4.5*LINE_HEIGHT:  # nr_1
                        self.gui_state["atv"] = 1
                    elif 250+4.5*LINE_HEIGHT <= screen_y < 250+9.5*LINE_HEIGHT:  # nr_2
                        self.gui_state["atv"] = 2
                    elif 250+9.5*LINE_HEIGHT <= screen_y < 250+14.5*LINE_HEIGHT:  # nr_3
                        self.gui_state["atv"] = 3
                    elif 250+14.5*LINE_HEIGHT <= screen_y < 250+19.5*LINE_HEIGHT:  # nr_4
                        self.gui_state["atv"] = 4
                    elif 250+19.5*LINE_HEIGHT <= screen_y < 250+24.5*LINE_HEIGHT:  # nr_5
                        self.gui_state["atv"] = 5
                    elif 250+24.5*LINE_HEIGHT <= screen_y < 250+29.5*LINE_HEIGHT:  # nr_6
                        self.gui_state["atv"] = 6
                    elif 250+29.5*LINE_HEIGHT <= screen_y < 250+34.5*LINE_HEIGHT:  # nr_7
                        self.gui_state["atv"] = 7
                    else:
                        self.gui_state["atv"] = None
                else:
                    self.gui_state["atv"] = None

    def draw_col_detail(self, loc):
        """Draw the selected col's stats/histograms and highlight its conns.
        Returns the col's loc as confirmed by the agent (or the requested loc
        if no info yet), for use by the conn/atv panels."""
        self.draw_col(loc, "border")

        # Send request
        _, pipe = self.pipes["col"]
        pipe.send(loc)

        # Try to get new information
        info = None
        while pipe.poll():
            new_info = pipe.recv()
            if new_info["loc"] == loc:
                info = new_info
                self.cache["col"] = new_info

        # If no new information, try to read from cache
        if info is None and self.cache["col"] is not None \
                and self.cache["col"]["loc"] == loc:
            info = self.cache["col"]

        if info is None:
            txt = self.fonts["debug"].render("waiting...", True, (0,0,0))
            self.window.blit(txt, (1525, 25+0*LINE_HEIGHT))
            return loc

        # Display debug info
        loc = info["loc"]  # As a way to verify information is coming from correct col
        txt = self.fonts["debug"].render(f"loc: {loc}", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(topleft=(1525, 25)))

        # Age of information
        age = time.time() - info["timestamp"]
        txt = self.fonts["debug"].render(f"age: {age:.3f}s", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(topleft=(1525, 25+LINE_HEIGHT)))

        # Number of activations and weights
        txt = self.fonts["debug"].render(f"activations: {info["nrns"]:,}", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(topleft=(1700, 25+0*LINE_HEIGHT)))
        txt = self.fonts["debug"].render(f"total weights: {info["syns"]:,}", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(topleft=(1700, 25+1*LINE_HEIGHT)))
        txt = self.fonts["debug"].render(f"|-- internal weights: {info["isyns"]:,}", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(topleft=(1700, 25+2*LINE_HEIGHT)))
        txt = self.fonts["debug"].render(f"|-- external weights: {info["esyns"]:,}", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(topleft=(1700, 25+3*LINE_HEIGHT)))

        # Ratios of numbers of weights to activations
        txt = self.fonts["debug"].render(f"{info["syns"]/info["nrns"]:,.2f} to 1", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(topleft=(2075, 25+1*LINE_HEIGHT)))
        txt = self.fonts["debug"].render(f"|-- {info["isyns"]/info["nrns"]:,.2f} to 1", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(topleft=(2075, 25+2*LINE_HEIGHT)))
        txt = self.fonts["debug"].render(f"|-- {info["esyns"]/info["nrns"]:,.2f} to 1", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(topleft=(2075, 25+3*LINE_HEIGHT)))

        # Legend for stats
        txt = self.fonts["debug"].render("name: shape", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(topleft=(2325, 25+0*LINE_HEIGHT)))
        txt = self.fonts["debug"].render("numel", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(topleft=(2325, 25+1*LINE_HEIGHT)))
        txt = self.fonts["debug"].render("density, norm", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(topleft=(2325, 25+2*LINE_HEIGHT)))
        txt = self.fonts["debug"].render("mean, std", True, (0,0,0))
        self.window.blit(txt, txt.get_rect(topleft=(2325, 25+3*LINE_HEIGHT)))
        pg.draw.rect(self.window, (0,0,0), (2310, 15, 175, 135), width=2)

        # Draw activation and weight stats and histograms
        activations = sorted([name for name in info if name.startswith("nr_")])
        txt_line = 0
        for name in activations:
            shape, d, n, m, s, (h, _), has_nan, all_nan = info[name][0]
            _, _, _, _, _, (h_e, _), has_nan_e, all_nan_e = info[name][1]
            txt = f"{name}: {shape}"
            txt = self.fonts["debug"].render(txt, True, (0,0,0))
            self.window.blit(txt, (1525, 250+txt_line*LINE_HEIGHT))
            txt_line += 1
            txt = f"{math.prod(shape):,}"
            txt = self.fonts["debug"].render(txt, True, (0,0,0))
            self.window.blit(txt, (1525, 250+txt_line*LINE_HEIGHT))
            txt_line += 1
            txt = f"{d:.4f}, {n:.4f}"
            txt = self.fonts["debug"].render(txt, True, (0,0,0))
            self.window.blit(txt, (1525, 250+txt_line*LINE_HEIGHT))
            txt_line += 1
            txt = f"{m:.4f}, {s:.4f}"
            txt = self.fonts["debug"].render(txt, True, (0,0,0))
            self.window.blit(txt, (1525, 250+txt_line*LINE_HEIGHT))

            txt_line -= 1
            histogram = self.get_histogram(h, 0.1, has_nan or has_nan_e, all_nan or all_nan_e, h_e=h_e)
            if histogram is not None:
                self.window.blit(histogram, histogram.get_rect(midright=(1975, 250+(txt_line)*LINE_HEIGHT)))
            txt_line += 3

        weights = sorted([name for name in info if name.startswith("is_")])
        txt_line = 0
        for name in weights:
            shape, d, n, m, s, (h, bin_width), has_nan, all_nan = info[name]
            txt = f"{name}: {shape}"
            txt = self.fonts["debug"].render(txt, True, (0,0,0))
            self.window.blit(txt, (2025, 250+txt_line*LINE_HEIGHT))
            txt_line += 1
            txt = f"{math.prod(shape):,}"
            txt = self.fonts["debug"].render(txt, True, (0,0,0))
            self.window.blit(txt, (2025, 250+txt_line*LINE_HEIGHT))
            txt_line += 1
            txt = f"{d:.4f}, {n:.4f}"
            txt = self.fonts["debug"].render(txt, True, (0,0,0))
            self.window.blit(txt, (2025, 250+txt_line*LINE_HEIGHT))
            txt_line += 1
            txt = f"{m:.4f}, {s:.4f}"
            txt = self.fonts["debug"].render(txt, True, (0,0,0))
            self.window.blit(txt, (2025, 250+txt_line*LINE_HEIGHT))

            txt_line -= 1
            histogram = self.get_histogram(h, bin_width, has_nan, all_nan)
            if histogram is not None:
                self.window.blit(histogram, histogram.get_rect(midright=(2475, 250+(txt_line)*LINE_HEIGHT)))
            txt_line += 3

        # Highlight outgoing connections
        for (conn_loc, direction) in info["conns"]:
            self.draw_col(conn_loc, dir2pos[direction])

        return loc

    def draw_conn_detail(self, loc, conn):
        """Draw the selected connection's stats/histogram (middle panel)."""
        conn_loc, conn_dir = conn
        self.draw_col(conn_loc, highlight=f"border{dir2pos[conn_dir]}")

        # Send request
        _, pipe = self.pipes["conn"]
        pipe.send((loc, conn_loc, conn_dir))

        # Try to get new information
        info = None
        while pipe.poll():
            new_info = pipe.recv()
            if new_info is not None and new_info["request"] == (loc, conn_loc, conn_dir):
                info = new_info
                self.cache["conn"] = new_info

        # If no new information, try to read from cache
        if info is None and self.cache["conn"] is not None \
                and self.cache["conn"]["request"] == (loc, conn_loc, conn_dir):
            info = self.cache["conn"]

        if info is None:
            txt = self.fonts["debug"].render("waiting...", True, (0,0,0))
            self.window.blit(txt, (1125, 25))
        elif not info["valid"]:
            txt = self.fonts["debug"].render("conn does not exist", True, (0,0,0))
            self.window.blit(txt, (1125, 25))
        else:
            # Display debug info
            loc, conn_loc, conn_dir = info["request"]
            txt = self.fonts["debug"].render(f"from: {loc}", True, (0,0,0))
            self.window.blit(txt, (1125, 25))
            txt = self.fonts["debug"].render(f"to: {conn_loc}", True, (0,0,0))
            self.window.blit(txt, (1125, 25+LINE_HEIGHT))
            txt = self.fonts["debug"].render(f"direction: {conn_dir}", True, (0,0,0))
            self.window.blit(txt, (1125, 25+2*LINE_HEIGHT))

            txt = self.fonts["debug"].render(f"age: {time.time()-info["timestamp"]:.3f}s", True, (0,0,0))
            self.window.blit(txt, (1125, 25+3*LINE_HEIGHT))

            # Weight values
            txt = self.fonts["debug"].render("name: shape, numel", True, (0,0,0))
            self.window.blit(txt, (1125, 25+5*LINE_HEIGHT))
            txt = self.fonts["debug"].render("density, norm, mean, std", True, (0,0,0))
            self.window.blit(txt, (1125, 25+6*LINE_HEIGHT))

            shape, d, n, m, s, (h, bin_width), has_nan, all_nan = info["stats"]
            txt = f"conn: {shape}, {math.prod(shape):,}"
            txt = self.fonts["debug"].render(txt, True, (0,0,0))
            self.window.blit(txt, (1125, 25+8*LINE_HEIGHT))
            txt = f"{d:.4f}, {n:.4f}, {m:.4f}, {s:.4f}"
            txt = self.fonts["debug"].render(txt, True, (0,0,0))
            self.window.blit(txt, (1125, 25+9*LINE_HEIGHT))

            histogram = self.get_histogram(h, bin_width, has_nan, all_nan)
            if histogram is not None:
                self.window.blit(histogram, histogram.get_rect(midtop=(1250, 25+10*LINE_HEIGHT)))

    def draw_atv(self, loc, atv):
        """Draw the selected activation layer's value grid (middle panel)."""
        # Send request
        _, pipe = self.pipes["atv"]
        request = (loc, atv)
        pipe.send(request)

        # Try to get new information
        info = None
        while pipe.poll():
            new_info = pipe.recv()
            if new_info is not None and new_info["request"] == request:
                info = new_info
                self.cache["atv"] = new_info

        # If no new information, try to read from cache
        if info is None and self.cache["atv"] is not None \
                and self.cache["atv"]["request"] == request:
            info = self.cache["atv"]

        if info is None:
            txt = self.fonts["debug"].render("waiting...", True, (0,0,0))
            self.window.blit(txt, (1125, 25))
        else:
            txt = self.fonts["debug"].render(f"loc: {info["request"][0]}, layer: nr_{info["request"][1]}", True, (0,0,0))
            self.window.blit(txt, (1125, 25))
            txt = self.fonts["debug"].render(f"age: {time.time()-info["timestamp"]:.3f}s", True, (0,0,0))
            self.window.blit(txt, (1125, 55))

            # Draw pointer arrow
            arrow = pg.Surface((30, 20))
            arrow.fill((255, 255, 255))
            arrow.set_colorkey((255, 255, 255))
            pg.draw.polygon(arrow, (0, 181, 226), ((0, 0), (30, 10), (0, 20)))
            self.window.blit(arrow, arrow.get_rect(midright=(1515, 305+150*(info["request"][1]-1))))

            # Draw activations grid
            x = info["x"]
            x_avg = info["x_avg"]
            WIDTH = 600 // (max(128, x.size)**0.5)  # Width of each grid cell
            GRID_WIDTH = 350 // WIDTH  # Number of grid cells across
            for i, (xi, x_avg_i) in enumerate(zip(x, x_avg, strict=True)):
                # Fill from top to bottom, them left to right
                px = i % GRID_WIDTH
                py = i // GRID_WIDTH

                # Current value
                color = get_color(xi)
                pg.draw.rect(self.window, color,
                    (1125 + WIDTH*px, 100 + WIDTH*py, WIDTH, WIDTH))

                # Time average value
                OVERLAY = 0.3
                color = get_color(x_avg_i)
                pg.draw.rect(self.window, color,
                    (1125 + WIDTH*(px + (1-OVERLAY)/2),
                        100 + WIDTH*(py + (1-OVERLAY)/2),
                        OVERLAY*WIDTH, OVERLAY*WIDTH))

            # Draw grid border
            """
            .-----------.
            |           |
            |           |
            |           |
            |           |
            |      .----.
            |      |
            .------.
            """
            top = min(x.size, GRID_WIDTH)
            left = x.size // GRID_WIDTH + (x.size % GRID_WIDTH != 0)
            right = max(x.size // GRID_WIDTH, 1)
            bottom = x.size % GRID_WIDTH
            pg.draw.aaline(self.window, (0, 0, 0), (1125, 100), (1125+WIDTH*top, 100))  # top
            pg.draw.aaline(self.window, (0, 0, 0), (1125, 100), (1125, 100+WIDTH*left))  # left
            pg.draw.aaline(self.window, (0, 0, 0), (1125+WIDTH*top, 100), (1125+WIDTH*top, 100+WIDTH*right))  # right
            pg.draw.aaline(self.window, (0, 0, 0), (1125, 100+WIDTH*left), (1125+WIDTH*bottom, 100+WIDTH*left))  # bottom left portion
            pg.draw.aaline(self.window, (0, 0, 0), (1125+WIDTH*bottom, 100+WIDTH*right), (1125+WIDTH*top, 100+WIDTH*right))  # bottom right portion
            pg.draw.aaline(self.window, (0, 0, 0), (1125+WIDTH*bottom, 100+WIDTH*right), (1125+WIDTH*bottom, 100+WIDTH*left))  # bottom vertical edge

    def draw_detail(self):
        """Draw col/conn/atv detail panels for the currently selected col."""
        if not self.gui_state["loc"]:  # No col selected
            return
        loc = self.gui_state["loc"]
        if os.path.isdir(f"{self.path}/{loc}"):  # Check if col location is valid
            loc = self.draw_col_detail(loc)
            if self.gui_state["conn"] is not None:
                self.draw_conn_detail(loc, self.gui_state["conn"])
            if self.gui_state["atv"] is not None:
                self.draw_atv(loc, self.gui_state["atv"])
        else:  # Selected col does not exist
            self.gui_state["loc"] = None
            self.gui_state["conn"] = None
            self.gui_state["atv"] = None

    def present(self):
        """Scale the virtual window onto the real window and flip."""
        self.true_window.fill((255,255,255))
        self.true_window.blit(pg.transform.smoothscale(self.window, (self.scale*W, self.scale*H)), (0,0))
        pg.display.update()

    def frame(self):
        """One iteration: draw everything, process input, present."""
        self.draw_static_layout()
        self.draw_cols()
        self.draw_overview()
        self.handle_events()
        self.draw_detail()
        self.present()

    def run(self):
        while True:
            self.frame()
