
"""
     0                       1100      1500       2000       2500
   0 .------------------------.---------.----------.----------.
     |                        |         |         Col         |
     |                        |         |       Overview      |
     |                        |         |          |          |
     |                        |         |          |          |
     |          Cols          |  Conns  |          |          |
     |          Grid          |   or    |  Activ-  | Weights  |
     |                        | Activity|  ations  |          |
     |                        |  Values |          |          |
     |                        |         |          |          |
     |                        |         |          |          |
1100 .------------------------.         |          |          |
     |     Global Overview    |         |          |          |
1300 .------------------------.---------.----------.----------.
"""

import pickle
import os
import time
import math
import sys
import warnings

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = ""
with warnings.catch_warnings(action="ignore"):
    # Pygame hasn't been updated since Sep 2024
    import pygame as pg

from .agents import Dir

def debugger(PATH, pipes):
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

    def draw_col(loc, highlight=None, colors=None):
        x, y = loc

        col = pg.Surface((COL_WIDTH, COL_WIDTH))
        col.fill((255,255,255))
        col.set_colorkey((255,255,255))  # so it's possible to draw different components (background, highlight, border) at separate times

        if colors is not None:
            for i, color in enumerate(colors):
                px_y = i * COL_WIDTH / len(colors)
                pg.draw.rect(col, color, (0, px_y, COL_WIDTH, COL_WIDTH/len(colors)))

        h, r = templates["h"], templates["r"]
        if highlight == "top":
            col.blit(h, (0, 0))
        elif highlight == "bottom":
            col.blit(h, (0, COL_WIDTH/2))

        pg.draw.rect(col, (0,0,0), (0, 0, COL_WIDTH, COL_WIDTH), 1)  # Outer (thin) border
        txt = fonts["col"].render(f"{x},{y}", True, (0,0,0))
        txt_rect = txt.get_rect(center=(COL_WIDTH/2, COL_WIDTH/2))
        col.blit(txt, txt_rect)

        if highlight == "border":
            pg.draw.rect(col, (0, 0, 0), col.get_rect(), width=int(COL_WIDTH*0.05))
        elif highlight == "bordertop":
            col.blit(r, (0, 0))
        elif highlight == "borderbottom":
            col.blit(r, (0, COL_WIDTH/2))

        window.blit(col, (x*COL_WIDTH, y*COL_WIDTH))

    def get_histogram(h, is_weight=False, h_e=None):
        assert not (is_weight and h_e)

        histogram = pg.Surface((100+215, 15+133+15))
        histogram.fill((255,255,255))
        histogram.set_colorkey((255,255,255))
        pg.draw.rect(histogram, (0,0,0), (100+0, 15+0, 215, 133), 1)  # Border

        txt = fonts["small"].render("0", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midright=(100, 15+133)))
        if max(h) > 0:  # check not NaN tensor
            txt = fonts["small"].render(f"{int(max(h)):,}", True, (0,0,0))
            histogram.blit(txt, txt.get_rect(midright=(100, 15+0)))
            for i, num in enumerate(h):
                height = round(133*(num/max(h)))
                pg.draw.rect(histogram, (0,0,0), (100+i*5, 15+133-height, 5, height))  # Opaque, black bars
        else:
            return None

        if h_e is not None:
            if max(h_e) > 0:
                max_height = max(h_e)
                txt = fonts["small"].render(f"{int(max_height):,}", True, (100,100,255))
                histogram.blit(txt, txt.get_rect(midright=(100, 15+15)))
                for i, num in enumerate(h_e):
                    height = round(133*(num/max_height))
                    surface = pg.Surface(histogram.get_size(), pg.SRCALPHA)
                    surface.fill((255, 255, 255))
                    surface.set_colorkey((255, 255, 255))
                    height_e = round(133*(num/max_height))
                    pg.draw.rect(surface, (200, 220, 255, 180), (100+i*5, 15+133-height_e, 5, height_e))  # Translucent, light blue bars
                    histogram.blit(surface, (0, 0))
            else:
                return None

        pg.draw.line(histogram, (100,149,237), (100+7.5, 15+133-10), (100+7.5, 15+133+5), width=2)      # -2 tick
        txt = fonts["small"].render("-0.1" if is_weight else "-2", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+7.5, 15+133)))

        pg.draw.line(histogram, (100,149,237), (100+57.5, 15+133-10), (100+57.5, 15+133+5), width=2)    # -1 tick
        txt = fonts["small"].render("-0.05" if is_weight else "-1", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+57.5, 15+133)))

        pg.draw.line(histogram, (100,149,237), (100+107.5, 15+133-10), (100+107.5, 15+133+5), width=2)  # 0 tick
        txt = fonts["small"].render("0", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+107.5, 15+133)))

        pg.draw.line(histogram, (100,149,237), (100+157.5, 15+133-10), (100+157.5, 15+133+5), width=2)  # 1 tick
        txt = fonts["small"].render("0.05" if is_weight else "1", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+157.5, 15+133)))

        pg.draw.line(histogram, (100,149,237), (100+207.5, 15+133-10), (100+207.5, 15+133+5), width=2)  # 2 tick
        txt = fonts["small"].render("0.1" if is_weight else "2", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+207.5, 15+133)))
        return histogram

    def get_color(x):
        x = round(x*255/2)  # [-2, 2] -> [-255, 255]
        color = (255-max(0,x),255+min(0,x),255-abs(x))
        color = [min(max(x, 0), 255) for x in color]
        return tuple(color)

    dir2pos = {
        Dir.A: "top",
        Dir.E: "bottom",
    }

    COL_WIDTH = int(1100 / (1+math.ceil(len(os.listdir(PATH))**0.5)))

    # Draw on 2500 x 1300 virtual window, then scale to size of real window
    W, H = (2500, 1300)
    window = pg.Surface((W, H))

    pg.init()
    (desktop_w, desktop_h), = pg.display.get_desktop_sizes()
    scale = 0.8 * min(desktop_w/2560, desktop_h/1440)  # Fit to display
    true_window = pg.display.set_mode((scale*W, scale*H), pg.DOUBLEBUF|pg.RESIZABLE)
    pg.display.set_caption("debugger")

    # Frequently used graphical elements
    # Half size grey highlight for showing conns
    h = pg.Surface((COL_WIDTH, COL_WIDTH/2))
    h.fill((200,200,200))
    # Half size blue border for selecting conn
    r = pg.Surface((COL_WIDTH, COL_WIDTH/2))
    r.fill((255,255,255))
    r.set_colorkey((255,255,255))
    pg.draw.rect(r, (100,149,237), r.get_rect(), width=int(COL_WIDTH*0.05))
    templates = {
        "h": h,
        "r": r
    }

    pg.font.init()
    fonts = {
        "col": pg.font.SysFont("Helvetica", int(0.3*COL_WIDTH)),
        "big": pg.font.SysFont("Helvetica", 48),
        "debug": pg.font.SysFont("Helvetica", 24),
        "small": pg.font.SysFont("Helvetica", 12)
    }
    LINE_HEIGHT = 30

    # Cache of which item user clicked on
    gui_state = {
        "loc": None,
        "conn": None,
        "atv": None,
    }

    # Cache of info received from agent
    cache = {
        "overview": None,
        "col": None,
        "conn": None,
        "atv": None,
    }

    while True:
        window.fill((255, 255, 255))

        # Horizontal lines
        pg.draw.line(window, (0, 0, 0), (0, 1100), (1100, 1100))

        # Vertical lines
        pg.draw.line(window, (0, 0, 0), (1100, 0), (1100, 1300))
        pg.draw.line(window, (0, 0, 0), (1500, 0), (1500, 1300))
        pg.draw.line(window, (0, 0, 0), (2000, 200), (2000, 1300))

        txt = fonts["big"].render("Activations:", True, (0,0,0))
        window.blit(txt, txt.get_rect(midbottom=(1750, 220)))
        txt = fonts["big"].render("Weights:", True, (0,0,0))
        window.blit(txt, txt.get_rect(midbottom=(2250, 220)))

        # Display cols
        for name in os.listdir(PATH):
            if name != "cfg":
                draw_col(eval(name))

        # Display overview info ###############################################
        if pipes is not None:
            _, pipe = pipes["overview"]
            # Try to get new info
            info = None
            while pipe.poll():
                new_info = pipe.recv()
                info = new_info
                cache["overview"] = new_info

            # If no new info, try to read from cache
            if info is None and cache["overview"] is not None:
                info = cache["overview"]

            if info is None:
                txt = fonts["debug"].render("waiting...", True, (0,0,0))
                window.blit(txt, (25, 1125))
            else:
                txt = fonts["debug"].render("Active", True, (0,0,0))
                window.blit(txt, (25, 1125))

                # Age of information
                age = time.time() - info["timestamp"]
                txt = fonts["debug"].render(f"age: {age:.3f}s", True, (0,0,0))
                window.blit(txt, (25, 1125+1*LINE_HEIGHT))

                memory = 2 * (info["copies"]*info["nrns"] + info["syns"])
                memory_gb = memory / 1e9
                txt = fonts["debug"].render("Memory:", True, (0,0,0))
                window.blit(txt, (25, 1125+3*LINE_HEIGHT))
                txt = fonts["debug"].render(f"{memory_gb:.2f} GB", True, (0,0,0))
                window.blit(txt, (25, 1125+4*LINE_HEIGHT))

                # Total number of activations and weights
                txt = fonts["debug"].render("# of:", True, (0,0,0))
                window.blit(txt, (200, 1125+0*LINE_HEIGHT))
                txt = fonts["debug"].render(f"    activations: {info["nrns"]:,} ({info["copies"]} copies)", True, (0,0,0))
                window.blit(txt, (200, 1125+1*LINE_HEIGHT))
                txt = fonts["debug"].render(f"    internal weights: {info["isyns"]:,}", True, (0,0,0))
                window.blit(txt, (200, 1125+2*LINE_HEIGHT))
                txt = fonts["debug"].render(f"    external weights: {info["esyns"]:,}", True, (0,0,0))
                window.blit(txt, (200, 1125+3*LINE_HEIGHT))
                txt = fonts["debug"].render(f"    total weights: {info["syns"]:,}", True, (0,0,0))
                window.blit(txt, (200, 1125+4*LINE_HEIGHT))

                txt = fonts["debug"].render("ratios:", True, (0,0,0))
                window.blit(txt, (625, 1125+0*LINE_HEIGHT))
                txt = fonts["debug"].render(f"|-- {info["isyns"]/info["nrns"]:,.2f} to 1", True, (0,0,0))
                window.blit(txt, (625, 1125+2*LINE_HEIGHT))
                txt = fonts["debug"].render(f"|-- {info["esyns"]/info["nrns"]:,.2f} to 1", True, (0,0,0))
                window.blit(txt, (625, 1125+3*LINE_HEIGHT))
                txt = fonts["debug"].render(f"|-- {info["syns"]/info["nrns"]:,.2f} to 1", True, (0,0,0))
                window.blit(txt, (625, 1125+4*LINE_HEIGHT))

                txt = fonts["debug"].render(f"density: {info["density"]*100:.2f}%", True, (0,0,0))
                window.blit(txt, (875, 1125+0*LINE_HEIGHT))

        # Handle gui changes ##################################################
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pipes["overview"][1].send(None)
                sys.exit()
            elif event.type == pg.VIDEORESIZE:
                w_new, h_new = event.size
                if w_new/h_new < W/H:
                    scale = w_new / W
                else:
                    scale = h_new / H
        buttons = pg.mouse.get_pressed(num_buttons=3)
        keys = pg.key.get_pressed()
        if buttons[0]:
            # Get which col and conn mouse is clicking on #####################
            screen_x, screen_y = pg.mouse.get_pos()
            screen_x, screen_y = screen_x / scale, screen_y / scale
            if keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT] or buttons[2]:
                # Try stay on same col and select conn or activation layer
                if gui_state["loc"] is not None:
                    loc = gui_state["loc"]
                    conn_loc = screen2loc(screen_x, screen_y, COL_WIDTH)
                    conn_dir = screen2dir(screen_x, screen_y, COL_WIDTH)
                    if os.path.isdir(f"{PATH}/{conn_loc}"):  # Check if conn location is valid
                        gui_state["atv"] = None
                        gui_state["conn"] = (conn_loc, conn_dir)
                    else:
                        gui_state["conn"] = None
                        if 1500 < screen_x < 2000:  # Try to select layer of activations
                            if 250+0*LINE_HEIGHT < screen_y < 250+4*LINE_HEIGHT:  # nr_1
                                gui_state["atv"] = 1
                            elif 250+5*LINE_HEIGHT < screen_y < 250+9*LINE_HEIGHT:  # nr_2
                                gui_state["atv"] = 2
                            elif 250+10*LINE_HEIGHT < screen_y < 250+14*LINE_HEIGHT:  # nr_3
                                gui_state["atv"] = 3
                            elif 250+15*LINE_HEIGHT < screen_y < 250+19*LINE_HEIGHT:  # nr_4
                                gui_state["atv"] = 4
                            elif 250+20*LINE_HEIGHT < screen_y < 250+24*LINE_HEIGHT:  # nr_5
                                gui_state["atv"] = 5
                            elif 250+25*LINE_HEIGHT < screen_y < 250+29*LINE_HEIGHT:  # nr_6
                                gui_state["atv"] = 6
                            elif 250+30*LINE_HEIGHT < screen_y < 250+34*LINE_HEIGHT:  # nr_7
                                gui_state["atv"] = 7
                            else:
                                gui_state["atv"] = None
                        else:
                            gui_state["atv"] = None
                else:  # No col selected so ignore shift
                    loc = screen2loc(screen_x, screen_y, COL_WIDTH)
                    gui_state["loc"] = loc
            else:  # Select col
                loc = screen2loc(screen_x, screen_y, COL_WIDTH)
                gui_state["loc"] = loc
                gui_state["conn"] = None

            # Draw col and conn debug info ####################################
            if os.path.isdir(f"{PATH}/{loc}"):  # check if col location is valid
                # Draw col debug info #########################################
                loc = gui_state["loc"]
                draw_col(loc, "border")
                if pipes is not None:  # Read live
                    _, pipe = pipes["col"]
                    # Send request
                    pipe.send(loc)

                    # Try to get new information
                    info = None
                    while pipe.poll():
                        new_info = pipe.recv()
                        if new_info["loc"] == loc:
                            info = new_info
                            cache["col"] = new_info

                    # If no new information, try to read from cache
                    if info is None and cache["col"] is not None:
                        if cache["col"]["loc"] == loc:
                            info = cache["col"]

                    if info is None:
                        txt = fonts["debug"].render("waiting...", True, (0,0,0))
                        window.blit(txt, (1525, 25+0*LINE_HEIGHT))
                    else:
                        # Display debug info
                        loc = info["loc"]  # As a way to verify information is coming from correct col
                        txt = fonts["debug"].render(f"loc: {loc}", True, (0,0,0))
                        window.blit(txt, txt.get_rect(topleft=(1525, 25)))

                        # Age of information
                        age = time.time() - info["timestamp"]
                        txt = fonts["debug"].render(f"age: {age:.3f}s", True, (0,0,0))
                        window.blit(txt, txt.get_rect(topleft=(1525, 25+LINE_HEIGHT)))

                        # Number of activations and weights
                        txt = fonts["debug"].render(f"activations: {info["nrns"]:,}", True, (0,0,0))
                        window.blit(txt, txt.get_rect(topleft=(1700, 25+0*LINE_HEIGHT)))
                        txt = fonts["debug"].render(f"internal weights: {info["isyns"]:,}", True, (0,0,0))
                        window.blit(txt, txt.get_rect(topleft=(1700, 25+1*LINE_HEIGHT)))
                        txt = fonts["debug"].render(f"external weights: {info["esyns"]:,}", True, (0,0,0))
                        window.blit(txt, txt.get_rect(topleft=(1700, 25+2*LINE_HEIGHT)))
                        txt = fonts["debug"].render(f"total weights: {info["syns"]:,}", True, (0,0,0))
                        window.blit(txt, txt.get_rect(topleft=(1700, 25+3*LINE_HEIGHT)))

                        # Ratios of numbers of weights to activations
                        txt = fonts["debug"].render("ratios:", True, (0,0,0))
                        window.blit(txt, txt.get_rect(topleft=(2050, 25+0*LINE_HEIGHT)))
                        txt = fonts["debug"].render(f"|-- {info["isyns"]/info["nrns"]:,.2f} to 1", True, (0,0,0))
                        window.blit(txt, txt.get_rect(topleft=(2050, 25+1*LINE_HEIGHT)))
                        txt = fonts["debug"].render(f"|-- {info["esyns"]/info["nrns"]:,.2f} to 1", True, (0,0,0))
                        window.blit(txt, txt.get_rect(topleft=(2050, 25+2*LINE_HEIGHT)))
                        txt = fonts["debug"].render(f"|-- {info["syns"]/info["nrns"]:,.2f} to 1", True, (0,0,0))
                        window.blit(txt, txt.get_rect(topleft=(2050, 25+3*LINE_HEIGHT)))

                        # Draw activation and weight stats and histograms
                        txt = fonts["debug"].render("name: shape", True, (0,0,0))
                        window.blit(txt, txt.get_rect(topleft=(2325, 25+0*LINE_HEIGHT)))
                        txt = fonts["debug"].render("numel", True, (0,0,0))
                        window.blit(txt, txt.get_rect(topleft=(2325, 25+1*LINE_HEIGHT)))
                        txt = fonts["debug"].render("density, norm", True, (0,0,0))
                        window.blit(txt, txt.get_rect(topleft=(2325, 25+2*LINE_HEIGHT)))
                        txt = fonts["debug"].render("mean, std", True, (0,0,0))
                        window.blit(txt, txt.get_rect(topleft=(2325, 25+3*LINE_HEIGHT)))
                        pg.draw.rect(window, (0,0,0), (2310, 20, 175, 130), width=2)

                        activations = sorted([name for name in info if name.startswith("nr_")])
                        txt_line = 0
                        for name in activations:
                            shape, d, n, m, s, h = info[name][0]
                            _, _, _, _, _, h_e = info[name][1]
                            txt = f"{name}: {shape}"
                            txt = fonts["debug"].render(txt, True, (0,0,0))
                            window.blit(txt, (1525, 250+txt_line*LINE_HEIGHT))
                            txt_line += 1
                            txt = f"{math.prod(shape):,}"
                            txt = fonts["debug"].render(txt, True, (0,0,0))
                            window.blit(txt, (1525, 250+txt_line*LINE_HEIGHT))
                            txt_line += 1
                            txt = f"{d:.4f}, {n:.4f}"
                            txt = fonts["debug"].render(txt, True, (0,0,0))
                            window.blit(txt, (1525, 250+txt_line*LINE_HEIGHT))
                            txt_line += 1
                            txt = f"{m:.4f}, {s:.4f}"
                            txt = fonts["debug"].render(txt, True, (0,0,0))
                            window.blit(txt, (1525, 250+txt_line*LINE_HEIGHT))

                            txt_line -= 1
                            histogram = get_histogram(h, is_weight=False, h_e=h_e)
                            if histogram is not None:
                                window.blit(histogram, histogram.get_rect(midright=(1975, 250+(txt_line)*LINE_HEIGHT)))
                            txt_line += 3

                        weights = sorted([name for name in info if name.startswith("is_")])
                        txt_line = 0
                        for name in weights:
                            shape, d, n, m, s, h = info[name]
                            txt = f"{name}: {shape}"
                            txt = fonts["debug"].render(txt, True, (0,0,0))
                            window.blit(txt, (2025, 250+txt_line*LINE_HEIGHT))
                            txt_line += 1
                            txt = f"{math.prod(shape):,}"
                            txt = fonts["debug"].render(txt, True, (0,0,0))
                            window.blit(txt, (2025, 250+txt_line*LINE_HEIGHT))
                            txt_line += 1
                            txt = f"{d:.4f}, {n:.4f}"
                            txt = fonts["debug"].render(txt, True, (0,0,0))
                            window.blit(txt, (2025, 250+txt_line*LINE_HEIGHT))
                            txt_line += 1
                            txt = f"{m:.4f}, {s:.4f}"
                            txt = fonts["debug"].render(txt, True, (0,0,0))
                            window.blit(txt, (2025, 250+txt_line*LINE_HEIGHT))

                            txt_line -= 1
                            histogram = get_histogram(h, is_weight=True)
                            if histogram is not None:
                                window.blit(histogram, histogram.get_rect(midright=(2475, 250+(txt_line)*LINE_HEIGHT)))
                            txt_line += 3

                        # Highlight connected
                        for (conn_loc, direction), _ in info["conns"].items():
                            draw_col(conn_loc, dir2pos[direction])

                # Draw conn debug info ######################################################
                if pipes is not None and gui_state["conn"] is not None:  # Read live
                    conn_loc, conn_dir = gui_state["conn"]
                    draw_col(conn_loc, highlight=f"border{dir2pos[conn_dir]}")
                    _, pipe = pipes["conn"]
                    # Send request
                    pipe.send((loc, conn_loc, conn_dir))

                    # Try to get new information
                    info = None
                    while pipe.poll():
                        new_info = pipe.recv()
                        if new_info is not None and new_info["request"] == (loc, conn_loc, conn_dir):
                            info = new_info
                            cache["conn"] = new_info

                    # If no new information, try to read from cache
                    if info is None and cache["conn"] is not None:
                        if cache["conn"]["request"] == (loc, conn_loc, conn_dir):
                            info = cache["conn"]

                    if info is None:
                        txt = fonts["debug"].render("waiting...", True, (0,0,0))
                        window.blit(txt, (1125, 25))
                    elif not info["valid"]:
                        txt = fonts["debug"].render("conn does not exist", True, (0,0,0))
                        window.blit(txt, (1125, 25))
                    else:
                        # Display debug info
                        loc, conn_loc, conn_dir = info["request"]
                        txt = fonts["debug"].render(f"from: {loc}", True, (0,0,0))
                        window.blit(txt, (1125, 25))
                        txt = fonts["debug"].render(f"to: {conn_loc}", True, (0,0,0))
                        window.blit(txt, (1125, 25+LINE_HEIGHT))
                        txt = fonts["debug"].render(f"direction: {conn_dir}", True, (0,0,0))
                        window.blit(txt, (1125, 25+2*LINE_HEIGHT))

                        txt = fonts["debug"].render(f"age: {time.time()-info["timestamp"]:.3f}s", True, (0,0,0))
                        window.blit(txt, (1125, 25+3*LINE_HEIGHT))

                        # Weight values
                        txt = fonts["debug"].render("name: shape, numel", True, (0,0,0))
                        window.blit(txt, (1125, 25+5*LINE_HEIGHT))
                        txt = fonts["debug"].render("density, norm, mean, std", True, (0,0,0))
                        window.blit(txt, (1125, 25+6*LINE_HEIGHT))

                        shape, d, n, m, s, h = info["stats"]
                        txt = f"conn: {shape}, {math.prod(shape):,}"
                        txt = fonts["debug"].render(txt, True, (0,0,0))
                        window.blit(txt, (1125, 25+8*LINE_HEIGHT))
                        txt = f"{d:.4f}, {n:.4f}, {m:.4f}, {s:.4f}"
                        txt = fonts["debug"].render(txt, True, (0,0,0))
                        window.blit(txt, (1125, 25+9*LINE_HEIGHT))

                        histogram = get_histogram(h, is_weight=True)
                        if histogram is not None:
                            window.blit(histogram, histogram.get_rect(midtop=(1250, 25+10*LINE_HEIGHT)))

                # Draw activations
                if pipes is not None and gui_state["atv"] is not None:  # Read live
                    _, pipe = pipes["atv"]
                    request = (loc, gui_state["atv"])
                    # Send request
                    pipe.send(request)

                    # Try to get new information
                    info = None
                    while pipe.poll():
                        new_info = pipe.recv()
                        if new_info is not None and new_info["request"] == request:
                            info = new_info
                            cache["atv"] = new_info

                    # If no new information, try to read from cache
                    if info is None and cache["atv"] is not None:
                        if cache["atv"]["request"] == request:
                            info = cache["atv"]

                    if info is None:
                        txt = fonts["debug"].render("waiting...", True, (0,0,0))
                        window.blit(txt, (1125, 25))
                    else:
                        txt = fonts["debug"].render(f"loc: {info["request"][0]}, layer: nr_{info["request"][1]}", True, (0,0,0))
                        window.blit(txt, (1125, 25))
                        txt = fonts["debug"].render(f"age: {time.time()-info["timestamp"]:.3f}s", True, (0,0,0))
                        window.blit(txt, (1125, 55))

                        # Draw pointer arrow
                        arrow = pg.Surface((30, 20))
                        arrow.fill((255, 255, 255))
                        arrow.set_colorkey((255, 255, 255))
                        pg.draw.polygon(arrow, (0, 181, 226), ((0, 0), (30, 10), (0, 20)))
                        window.blit(arrow, arrow.get_rect(midright=(1515, 305+150*(info["request"][1]-1))))

                        # Draw activations grid
                        x = info["x"]
                        WIDTH = 600 // (max(128, x.size)**0.5)
                        GRID_WIDTH = 350 // WIDTH
                        for i, xi in enumerate(x):
                            # Fill from top to bottom, them left to right
                            px = i % GRID_WIDTH
                            py = i // GRID_WIDTH
                            color = get_color(xi)
                            pg.draw.rect(window, color, (1125+WIDTH*px, 100+WIDTH*py, WIDTH, WIDTH))

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
                        pg.draw.aaline(window, (0, 0, 0), (1125, 100), (1125+WIDTH*top, 100))  # top
                        pg.draw.aaline(window, (0, 0, 0), (1125, 100), (1125, 100+WIDTH*left))  # left
                        pg.draw.aaline(window, (0, 0, 0), (1125+WIDTH*top, 100), (1125+WIDTH*top, 100+WIDTH*right))  # right
                        pg.draw.aaline(window, (0, 0, 0), (1125, 100+WIDTH*left), (1125+WIDTH*bottom, 100+WIDTH*left))  # bottom left portion
                        pg.draw.aaline(window, (0, 0, 0), (1125+WIDTH*bottom, 100+WIDTH*right), (1125+WIDTH*top, 100+WIDTH*right))  # bottom right portion
                        pg.draw.aaline(window, (0, 0, 0), (1125+WIDTH*bottom, 100+WIDTH*right), (1125+WIDTH*bottom, 100+WIDTH*left))  # bottom vertical edge
        else:
            gui_state = {
                "loc": None,
                "conn": None,
                "atv": None,
            }

        true_window.fill((255,255,255))
        true_window.blit(pg.transform.smoothscale(window, (scale*W, scale*H)), (0,0))
        pg.display.update()
