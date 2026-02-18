
import math
import os
import time
import sys
import warnings

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = ""
with warnings.catch_warnings(action="ignore"):
    import pygame as pg


def nrn_debugger(PATH, pipes):
    def screen2loc(x, y, width):  # Convert screen coordinates to col coordinates (loc)
        return int(x/width), int(y/width)

    def draw_col(loc, highlight=None, color=(255,255,255)):
        x, y = loc

        col = pg.Surface((COL_WIDTH, COL_WIDTH))
        col.fill(color)
        col.set_colorkey((255,255,255))  # so it's possible to draw different components (background, highlight, border) at separate times

        h, r = templates["h"], templates["r"]
        if highlight == "highlight":
            col.blit(h, (0, 0))

        pg.draw.rect(col, (0,0,0), (0, 0, COL_WIDTH, COL_WIDTH), 1)  # Outer (thin) border
        txt = fonts["col"].render(f"{x},{y}", True, (0,0,0))
        txt_rect = txt.get_rect(center=(COL_WIDTH/2, COL_WIDTH/2))
        col.blit(txt, txt_rect)

        if highlight == "border":
            pg.draw.rect(col, (0, 0, 0), col.get_rect(), width=int(COL_WIDTH*0.2))
        elif highlight == "borderconn":
            col.blit(r, (0, 0))

        window.blit(col, (x*COL_WIDTH, y*COL_WIDTH))

    def get_color(x):
        # positive: green by reduce r, b; max green at >= 2
        # negative: red by reduce g, b; max red at <= -2
        x = round(x*255/2)  # [-2, 2] -> [-255, 255]
        color = (255-max(0,x),255+min(0,x),255-abs(x))
        color = [min(max(x, 0), 255) for x in color]
        return tuple(color)

    def get_histogram(h):
        histogram = pg.Surface((100+215, 15+133+15))
        histogram.fill((255,255,255))
        histogram.set_colorkey((255,255,255))
        pg.draw.rect(histogram, (0,0,0), (100+0, 15+0, 215, 133), 1)  # Border
        if max(h) > 0:  # not NaN tensor
            txt = fonts["small"].render("0", True, (0,0,0))
            histogram.blit(txt, txt.get_rect(midright=(100, 15+133)))
            txt = fonts["small"].render(f"{int(max(h)):,}", True, (0,0,0))
            histogram.blit(txt, txt.get_rect(midright=(100, 15+0)))
            for i, num in enumerate(h):
                height = round(133*(num/max(h)))
                pg.draw.rect(histogram, (0,0,0), (100+i*5, 15+133-height, 5, height))
        else:
            return None
        pg.draw.line(histogram, (100,149,237), (100+7.5, 15+133-10), (100+7.5, 15+133+5), width=2)      # -2 tick
        txt = fonts["small"].render("-2", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+7.5, 15+133)))

        pg.draw.line(histogram, (100,149,237), (100+57.5, 15+133-10), (100+57.5, 15+133+5), width=2)    # -1 tick
        txt = fonts["small"].render("-1", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+57.5, 15+133)))

        pg.draw.line(histogram, (100,149,237), (100+107.5, 15+133-10), (100+107.5, 15+133+5), width=2)  # 0 tick
        txt = fonts["small"].render("0", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+107.5, 15+133)))

        pg.draw.line(histogram, (100,149,237), (100+157.5, 15+133-10), (100+157.5, 15+133+5), width=2)  # 1 tick
        txt = fonts["small"].render("1", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+157.5, 15+133)))

        pg.draw.line(histogram, (100,149,237), (100+207.5, 15+133-10), (100+207.5, 15+133+5), width=2)  # 2 tick
        txt = fonts["small"].render("2", True, (0,0,0))
        histogram.blit(txt, txt.get_rect(midtop=(100+207.5, 15+133)))
        return histogram

    COL_WIDTH = int(1250 / (1+math.ceil(len(os.listdir(PATH))**0.5)))

    # Draw on 2500 x 1300 virtual window, then scale to size of real window
    W, H = (1300+700+500, 1300)
    window = pg.Surface((W, H))

    pg.init()
    (desktop_w, desktop_h), = pg.display.get_desktop_sizes()
    scale = 0.8 * min(desktop_w/2560, desktop_h/1440)  # Fit to display
    true_window = pg.display.set_mode((scale*W, scale*H), pg.DOUBLEBUF|pg.RESIZABLE)
    pg.display.set_caption("simple debugger")

    pg.font.init()
    fonts = {
        "col": pg.font.SysFont("Helvetica", int(0.3*COL_WIDTH)),
        "large": pg.font.SysFont("Helvetica", 96),
        "regular": pg.font.SysFont("Helvetica", 48),
        "debug": pg.font.SysFont("Helvetica", 24),
        "small": pg.font.SysFont("Helvetica", 12)
    }

    def get_color_bar():
        # Color gradient bar
        color_bar = pg.Surface((500, 150))
        color_bar.fill((255,255,255))
        color_bar.set_colorkey((255,255,255))
        for coord_x in range(50, 450, 1):
            x = (coord_x-50)/(450-50)*4-2  # [50, 450] -> [-2, 2]
            pg.draw.line(color_bar, get_color(x), (coord_x, 0), (coord_x, 80))
        pg.draw.line(color_bar, (0,0,0), (50, 75), (50, 85), width=5)      # -2 tick
        txt = fonts["regular"].render("-2", True, (0,0,0))
        color_bar.blit(txt, txt.get_rect(midtop=(50, 85)))

        pg.draw.line(color_bar, (0,0,0), (150, 75), (150, 85), width=5)      # -1 tick
        txt = fonts["regular"].render("-1", True, (0,0,0))
        color_bar.blit(txt, txt.get_rect(midtop=(150, 85)))

        pg.draw.line(color_bar, (0,0,0), (250, 75), (250, 85), width=5)      # 0 tick
        txt = fonts["regular"].render("0", True, (0,0,0))
        color_bar.blit(txt, txt.get_rect(midtop=(250, 85)))

        pg.draw.line(color_bar, (0,0,0), (350, 75), (350, 85), width=5)      # 1 tick
        txt = fonts["regular"].render("1", True, (0,0,0))
        color_bar.blit(txt, txt.get_rect(midtop=(350, 85)))

        pg.draw.line(color_bar, (0,0,0), (450, 75), (450, 85), width=5)      # 2 tick
        txt = fonts["regular"].render("2", True, (0,0,0))
        color_bar.blit(txt, txt.get_rect(midtop=(450, 85)))
        return color_bar
    color_bar = get_color_bar()

    # Frequently used visual elements
    h = pg.Surface((COL_WIDTH, COL_WIDTH))  # Highlight outgoing connections (blue border)
    h.fill((255,255,255))
    pg.draw.rect(h, (65,105,225), h.get_rect(), width=int(COL_WIDTH*0.2))
    r = pg.Surface((COL_WIDTH, COL_WIDTH))  # Select a conn (orange border)
    r.fill((255,255,255))
    r.set_colorkey((255,255,255))
    pg.draw.rect(r, (255,165,0), r.get_rect(), width=int(COL_WIDTH*0.2))
    templates = {
        "h": h,
        "r": r
    }

    # For visually showing activations and weights values
    square = {
        "small": pg.Surface((100, 100)),
        "big": pg. Surface((200, 200))
    }

    # cache of debugger gui state e.g. which item user clicked on
    gui_state = {
        "loc": None,
        "conn": None
    }

    # cache of info received from agent
    cache = {
        "overview": None,
        "nrn": None
    }

    while True:
        window.fill((255, 255, 255))
        pg.draw.line(window, (0, 0, 0), (1300, 0), (1300, 1300))
        pg.draw.line(window, (0, 0, 0), (2000, 0), (2000, 1300))
        pg.draw.line(window, (0, 0, 0), (1300+700, 900), (1300+700+500, 900))

        # Display overview of levels of activity ##############################
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
                txt = fonts["regular"].render("waiting...", True, (0,0,0))
                window.blit(txt, (2050, 50))
            else:
                txt = fonts["regular"].render("Active", True, (0,0,0))
                window.blit(txt, (2050, 50))

                # Age of information
                age = time.time() - info["timestamp"]
                txt = fonts["regular"].render(f"age: {age:.3f}s", True, (0,0,0))
                window.blit(txt, (2050, 110))

                # Color gradient bar
                window.blit(color_bar, (2000, 170))

                # Draw activations stats and histogram
                (shape,), d, n, m, s, h = info["nrn_stats"]
                txt = fonts["debug"].render(f"# of activations: {shape:,}", True, (0,0,0))
                window.blit(txt, (2050, 320))
                txt = fonts["debug"].render("density, norm, mean, std", True, (0,0,0))
                window.blit(txt, (2050, 350))
                txt = fonts["debug"].render(f"{d:.3f}, {n:.3f}, {m:.3f}, {s:.3f}", True, (0,0,0))
                window.blit(txt, (2050, 380))
                histogram = get_histogram(h)
                window.blit(histogram, histogram.get_rect(center=(2150, 490)))

                # Draw weights stats and histogram
                (shape,), d, n, m, s, h = info["syn_stats"]
                txt = fonts["debug"].render(f"# of weights: {shape:,}", True, (0,0,0))
                window.blit(txt, (2050, 620))
                txt = fonts["debug"].render("density, norm, mean, std", True, (0,0,0))
                window.blit(txt, (2050, 650))
                txt = fonts["debug"].render(f"{d:.3f}, {n:.3f}, {m:.3f}, {s:.3f}", True, (0,0,0))
                window.blit(txt, (2050, 680))
                histogram = get_histogram(h)
                window.blit(histogram, histogram.get_rect(center=(2150, 790)))

                # Display cols
                for name in os.listdir(PATH):
                    if name != "cfg":
                        x = info[eval(name)]
                        draw_col(eval(name), color=get_color(x))

        # Handle col and conn debug ###########################################
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
            if keys[pg.K_LSHIFT] or buttons[2]:  # try stay on same col and select conn
                if gui_state["loc"] is not None:  # select conn
                    loc = gui_state["loc"]
                    conn_loc = screen2loc(screen_x, screen_y, COL_WIDTH)
                    if os.path.isdir(f"{PATH}/{conn_loc}"):  # check if conn location is valid
                        gui_state["conn"] = conn_loc
                    else:
                        gui_state["conn"] = None
                else:  # no col selected so ignore shift
                    loc = screen2loc(screen_x, screen_y, COL_WIDTH)
                    gui_state["loc"] = loc
            else:  # select col
                loc = screen2loc(screen_x, screen_y, COL_WIDTH)
                gui_state["loc"] = loc
                gui_state["conn"] = None

            # Draw col and conn debug info ####################################
            if os.path.isdir(f"{PATH}/{loc}"):  # Check if col location is valid
                # Draw col debug info #########################################
                loc = gui_state["loc"]
                draw_col(loc, "border")
                if pipes is not None:  # Read live
                    _, pipe = pipes["nrn"]
                    # Send request
                    pipe.send(loc)

                    # Try to get new information
                    info = None
                    while pipe.poll():
                        new_info = pipe.recv()
                        if new_info["loc"] == loc:
                            info = new_info
                            cache["nrn"] = new_info

                    # If no new information, try to read from cache
                    if info is None and cache["nrn"] is not None:
                        if cache["nrn"]["loc"] == loc:
                            info = cache["nrn"]

                    if info is None:
                        txt = fonts["large"].render("waiting...", True, (0,0,0))
                        window.blit(txt, (1350, 50))
                    else:
                        # Display debug info
                        loc = info["loc"]  # As a way to verify information is coming from correct col
                        txt = fonts["large"].render(f"loc: {loc}", True, (0,0,0))
                        window.blit(txt, (1350, 50))

                        # Age of information
                        age = time.time() - info["timestamp"]
                        txt = fonts["large"].render(f"age: {age:.3f}s", True, (0,0,0))
                        window.blit(txt, (1350, 170))

                        x = cache["overview"][loc]
                        txt = fonts["large"].render(f"{x:.3f}", True, (0,0,0))
                        window.blit(txt, (1350, 410))

                        sq = square["big"]
                        sq.fill(get_color(x))
                        window.blit(sq, sq.get_rect(midleft=(1750, 470)))

                        # Highlight connected
                        for conn_loc, _ in info["conns"].items():
                            draw_col(conn_loc, "highlight")

                # Draw conn debug info ########################################
                if pipes is not None and gui_state["conn"] is not None:  # Read live
                    conn_loc = gui_state["conn"]
                    draw_col(conn_loc, highlight="borderconn")

                    conn = cache["nrn"]["conns"].get(conn_loc)
                    if conn is not None:
                        # Display debug info
                        txt = fonts["regular"].render(f"from: {gui_state["loc"]}", True, (0,0,0))
                        window.blit(txt, (2050, 950))
                        txt = fonts["regular"].render(f"to: {gui_state["conn"]}", True, (0,0,0))
                        window.blit(txt, (2050, 1010))

                        # Weight value
                        txt = fonts["regular"].render(f"{conn:.3f}", True, (0,0,0))
                        window.blit(txt, (2050, 1130))

                        sq = square["small"]
                        sq.fill(get_color(conn))
                        window.blit(sq, sq.get_rect(midleft=(2300, 1160)))
                    else:
                        txt = fonts["regular"].render("conn does not exist", True, (0,0,0))
                        window.blit(txt, (2050, 950))
        else:
            gui_state = {
                "loc": None,
                "conn": None
            }

        true_window.fill((255,255,255))
        true_window.blit(pg.transform.smoothscale(window, (scale*W, scale*H)), (0,0))
        pg.display.update()
