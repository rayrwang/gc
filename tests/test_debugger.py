
import os

# Headless: must be set before pygame creates a display (respects an explicit
# user override, e.g. running with a real driver locally)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import multiprocessing
import time

import numpy as np
import pygame as pg
import pytest

# isort: off
from src.agents import Dir
from src.debugger import (
    ATV_STATS, GRID, H, LINE_HEIGHT, MIDDLE, OVERVIEW, STATS_BLOCK_LINES,
    STATS_TOP, W, WEIGHT_STATS, Debugger, get_color, parse_loc, screen2dir,
    screen2loc,
)

LOCS = [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]


# Pure helpers ################################################################
def test_screen2loc():
    """Grid cells are COL_WIDTH squares indexed from the top left."""
    assert screen2loc(0, 0, 100) == (0, 0)
    assert screen2loc(99, 99, 100) == (0, 0)
    assert screen2loc(100, 250, 100) == (1, 2)


def test_screen2dir():
    """Top half of a cell selects Dir.A, bottom half Dir.E (in-cell coords)."""
    assert screen2dir(150, 120, 100) == Dir.A     # y%100=20, top half
    assert screen2dir(150, 180, 100) == Dir.E     # y%100=80, bottom half
    assert screen2dir(0, 50, 100) == Dir.A        # boundary y == width/2 is top


def test_get_color():
    """Red negative, white zero, green positive, clipped at |2|; NaN is cyan."""
    assert get_color(0) == (255, 255, 255)
    assert get_color(2) == (0, 255, 0)
    assert get_color(-2) == (255, 0, 0)
    assert get_color(99) == get_color(2)          # clipped
    assert get_color(float("nan")) == (0, 255, 255)


def test_parse_loc():
    """Col dir names parse to locs; agt metadata and stray files are skipped."""
    assert parse_loc("(1, 2)") == (1, 2)
    assert parse_loc("type") is None
    assert parse_loc("cfg") is None
    assert parse_loc("cfg_type") is None
    assert parse_loc(".DS_Store") is None
    assert parse_loc("(1, 2, 3)") is None         # not a 2-tuple
    assert parse_loc("5") is None


def test_layout_panels_tile_the_canvas():
    """The layout table's panels must tile the virtual canvas: adjacent edges
    coincide (the drawn dividers and the click zones both depend on this)."""
    assert GRID.bottom == OVERVIEW.top
    assert GRID.right == OVERVIEW.right == MIDDLE.left
    assert MIDDLE.right == ATV_STATS.left
    assert ATV_STATS.right == WEIGHT_STATS.left
    assert WEIGHT_STATS.right == W
    assert OVERVIEW.bottom == MIDDLE.bottom == ATV_STATS.bottom == H


# Debugger fixture ############################################################
def make_stats(shape=(1024,), has_nan=False, all_nan=False):
    h = [(i * 7) % 60 for i in range(43)]
    return (shape, 0.11, 42.84, 0.0003, 1.34, (h, 0.1), has_nan, all_nan)


def col_info(loc):
    return {
        "timestamp": time.time(), "loc": loc,
        "nrns": 4224, "isyns": 3_407_872, "esyns": 10_485_760,
        "syns": 13_893_632,
        "nr_1": [make_stats(), make_stats(), make_stats(), make_stats()],
        "is_1_2": make_stats(shape=(1024, 1024)),
        "conns": {((2, 1), Dir.A): None, ((1, 2), Dir.E): None},
    }


@pytest.fixture
def dbg(tmp_path):
    for loc in LOCS:
        os.mkdir(tmp_path / str(loc))
    pipes = {name: multiprocessing.Pipe() for name in ("overview", "col", "conn", "atv")}
    d = Debugger(str(tmp_path), pipes)
    yield d
    for a, b in pipes.values():
        a.close()
        b.close()


def region_has_content(window, rect):
    """True if any pixel inside rect differs from the white background."""
    arr = pg.surfarray.array3d(window)  # (W, H, 3)
    region = arr[rect.left:rect.right, rect.top:rect.bottom]
    return bool((region != 255).any())


# _drain semantics ############################################################
def test_drain_newest_wins_and_caches(dbg):
    send = dbg.pipes["overview"][0]
    send.send({"timestamp": 1})
    send.send({"timestamp": 2})
    assert dbg._drain("overview")["timestamp"] == 2   # newest of the backlog
    assert dbg.cache["overview"]["timestamp"] == 2
    assert dbg._drain("overview")["timestamp"] == 2   # empty pipe -> cache


def test_drain_predicate_filters_and_gates_cache(dbg):
    send = dbg.pipes["col"][0]
    send.send(col_info((9, 9)))                       # for another col
    assert dbg._drain("col", lambda i: i["loc"] == (1, 1)) is None
    send.send(col_info((1, 1)))
    assert dbg._drain("col", lambda i: i["loc"] == (1, 1))["loc"] == (1, 1)
    # cache holds (1, 1); asking for a different col must not serve it
    assert dbg._drain("col", lambda i: i["loc"] == (2, 1)) is None


# Hit-testing through the real event path ####################################
def press(monkeypatch, pos_virtual, scale, buttons=(False, False, False), esc=False):
    monkeypatch.setattr(pg.mouse, "get_pos",
        lambda: (pos_virtual[0] * scale, pos_virtual[1] * scale))
    monkeypatch.setattr(pg.mouse, "get_pressed", lambda num_buttons=3: buttons)

    class Keys:
        def __getitem__(self, key):
            return esc and key == pg.K_ESCAPE
    monkeypatch.setattr(pg.key, "get_pressed", lambda: Keys())


def test_click_selects_col(dbg, monkeypatch):
    w = dbg.COL_WIDTH
    press(monkeypatch, (1.5 * w, 1.5 * w), dbg.scale, buttons=(True, False, False))
    dbg.handle_events()
    assert dbg.gui_state["loc"] == (1, 1)


def test_right_click_selects_conn_by_half(dbg, monkeypatch):
    dbg.gui_state["loc"] = (1, 1)
    w = dbg.COL_WIDTH
    press(monkeypatch, (2.5 * w, 1.25 * w), dbg.scale, buttons=(False, False, True))
    dbg.handle_events()
    assert dbg.gui_state["conn"] == ((2, 1), Dir.A)   # top half -> Dir.A
    press(monkeypatch, (2.5 * w, 1.75 * w), dbg.scale, buttons=(False, False, True))
    dbg.handle_events()
    assert dbg.gui_state["conn"] == ((2, 1), Dir.E)   # bottom half -> Dir.E


def test_right_click_stats_band_selects_layer(dbg, monkeypatch):
    dbg.gui_state["loc"] = (1, 1)
    # Center of stats block 2's band, x inside the activations panel (and not
    # over any col dir)
    y = STATS_TOP + (STATS_BLOCK_LINES * 1.5 - 0.5) * LINE_HEIGHT
    press(monkeypatch, (ATV_STATS.centerx, y), dbg.scale, buttons=(False, False, True))
    dbg.handle_events()
    assert dbg.gui_state["atv"] == 2
    # Below the last band: no layer
    press(monkeypatch, (ATV_STATS.centerx, STATS_TOP + 40 * LINE_HEIGHT), dbg.scale,
        buttons=(False, False, True))
    dbg.handle_events()
    assert dbg.gui_state["atv"] is None


def test_escape_clears_selection(dbg, monkeypatch):
    dbg.gui_state.update(loc=(1, 1), conn=((2, 1), Dir.A), atv=1)
    press(monkeypatch, (0, 0), dbg.scale, esc=True)
    dbg.handle_events()
    assert dbg.gui_state == {"loc": None, "conn": None, "atv": None}


def test_quit_signals_agent_and_exits(dbg, monkeypatch):
    """Closing the window must send None on the overview pipe (the agent's
    exit signal) before the process dies."""
    press(monkeypatch, (0, 0), dbg.scale)
    pg.event.post(pg.event.Event(pg.QUIT))
    with pytest.raises(SystemExit):
        dbg.handle_events()
    assert dbg.pipes["overview"][0].poll()
    assert dbg.pipes["overview"][0].recv() is None


# Per-state render smoke ######################################################
# Not pixel-exact (that would break on any intentional visual change); asserts
# each state draws SOMETHING into its panel region without raising.
def test_frame_idle_renders(dbg, monkeypatch):
    press(monkeypatch, (0, 0), dbg.scale)
    dbg.pipes["overview"][0].send({
        "timestamp": time.time(), "nrns": 844_820, "copies": 3,
        "isyns": 681_574_400, "esyns": 1_458_667_520,
        "syns": 2_140_241_920, "density": 0.1113,
    })
    dbg.frame()
    assert region_has_content(dbg.window, GRID)       # cols drawn
    assert region_has_content(dbg.window, OVERVIEW)   # overview stats drawn


def test_frame_col_state_renders_stats(dbg, monkeypatch):
    press(monkeypatch, (0, 0), dbg.scale)
    dbg.gui_state["loc"] = (1, 1)
    dbg.pipes["col"][0].send(col_info((1, 1)))
    dbg.frame()
    assert region_has_content(dbg.window, pg.Rect(ATV_STATS.left, STATS_TOP, 500, 300))
    assert region_has_content(dbg.window, pg.Rect(WEIGHT_STATS.left, STATS_TOP, 500, 300))


def test_frame_atv_state_renders_value_grid(dbg, monkeypatch):
    press(monkeypatch, (0, 0), dbg.scale)
    dbg.gui_state["loc"] = (1, 1)
    dbg.gui_state["atv"] = 1
    dbg.pipes["col"][0].send(col_info((1, 1)))
    x = np.linspace(-3, 3, 128).astype(np.float32)
    x[5] = np.nan                                     # exercises the NaN color
    dbg.pipes["atv"][0].send({
        "timestamp": time.time(), "request": ((1, 1), 1),
        "x": x, "x_avg": (x * 0.5).astype(np.float32),
    })
    dbg.frame()
    assert region_has_content(dbg.window, pg.Rect(MIDDLE.left, 100, 400, 600))


def test_frame_stale_selection_self_clears(dbg, monkeypatch):
    """Selecting a col that does not exist resets the selection (old behavior:
    draw_detail validates against the save dir)."""
    press(monkeypatch, (0, 0), dbg.scale)
    dbg.gui_state["loc"] = (9, 9)
    dbg.frame()
    assert dbg.gui_state["loc"] is None


def test_histogram_nan_states(dbg):
    """Has-NaN and all-NaN histograms render (the all-NaN path must not divide
    by max(h)=0)."""
    surf = dbg.get_histogram([0] * 43, 0.1, True, True)
    assert surf is not None
    surf = dbg.get_histogram([1] * 43, 0.1, True, False, h_e=[2] * 43)
    assert surf is not None


def test_stats_bands_match_block_geometry():
    """The click band for layer i must contain the y-rows where block i's text
    is drawn (rows 0-3 of the block) -- the drift the layout table exists to
    prevent."""
    for i in range(1, 8):
        block_top = STATS_TOP + STATS_BLOCK_LINES * (i - 1) * LINE_HEIGHT
        rows = [block_top + k * LINE_HEIGHT for k in range(4)]
        lo = STATS_TOP if i == 1 \
            else STATS_TOP + (STATS_BLOCK_LINES * (i - 1) - 0.5) * LINE_HEIGHT
        hi = STATS_TOP + (STATS_BLOCK_LINES * i - 0.5) * LINE_HEIGHT
        assert all(lo <= y < hi for y in rows), f"band {i} misses its block"
