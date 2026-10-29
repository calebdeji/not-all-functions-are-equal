"""
generate_temporal_sheet.py

Generates an Excel workbook from the temporal evaluation results
(evaluate_20260716_112610.log):
  - Accuracy sheet: per-year (2019, 2020, 2021) + combined, 1-shot and 5-shot
  - Resources sheet: shared resource data for these models

Output: results/table_temporal.xlsx
"""

import os
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from src.core.utils import Utils

OUT_DIR = os.path.join(Utils.get_project_root(), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Raw data — HiGraph temporal split, 5-way, 100 episodes per year
# (acc_mean, ci) per year and combined
# ---------------------------------------------------------------------------

# Structure: model -> {year: (mean, ci), 'combined': (mean, ci)}
#            two entries per model: '1shot' and '5shot'

ACC_TEMPORAL = {
    'sdl_complexity': {
        '1shot': {'2019': (63.31, 2.76), '2020': (71.80, 2.97), '2021': (62.92, 2.40), 'combined': (66.91, 2.71)},
        '5shot': {'2019': (73.57, 2.37), '2020': (80.56, 2.63), '2021': (75.95, 1.89), 'combined': (80.89, 2.01)},
    },
    'sdl_degree': {
        '1shot': {'2019': (63.25, 2.80), '2020': (71.75, 2.96), '2021': (62.93, 2.40), 'combined': (66.88, 2.71)},
        '5shot': {'2019': (73.57, 2.37), '2020': (80.59, 2.61), '2021': (75.92, 1.88), 'combined': (80.89, 2.01)},
    },
    'self_attention': {
        '1shot': {'2019': (62.84, 2.71), '2020': (69.99, 2.88), '2021': (59.68, 2.53), 'combined': (63.09, 2.69)},
        '5shot': {'2019': (73.53, 2.40), '2020': (79.09, 2.46), '2021': (73.13, 1.87), 'combined': (78.91, 2.11)},
    },
    'fcgat': {
        '1shot': {'2019': (60.83, 2.75), '2020': (63.83, 2.42), '2021': (57.24, 2.26), 'combined': (62.45, 2.44)},
        '5shot': {'2019': (73.91, 2.29), '2020': (75.96, 2.41), '2021': (70.45, 1.92), 'combined': (79.04, 2.03)},
    },
    'famcf': {
        '1shot': {'2019': (65.29, 2.45), '2020': (74.37, 2.20), '2021': (69.64, 2.89), 'combined': (70.84, 2.61)},
        '5shot': {'2019': (72.28, 2.25), '2020': (81.85, 1.82), '2021': (76.90, 2.09), 'combined': (81.71, 2.18)},
    },
    'androfim': {
        '1shot': {'2019': (55.51, 2.63), '2020': (68.24, 2.08), '2021': (55.97, 2.64), 'combined': (59.16, 2.36)},
        '5shot': {'2019': (68.04, 2.36), '2020': (78.68, 1.87), '2021': (67.34, 2.33), 'combined': (74.46, 2.38)},
    },
    'pretrained_backbone': {
        '1shot': {'2019': (57.85, 2.75), '2020': (63.91, 2.23), '2021': (58.12, 2.58), 'combined': (60.68, 3.04)},
        '5shot': {'2019': (66.69, 2.29), '2020': (70.69, 2.75), '2021': (68.13, 1.84), 'combined': (72.88, 2.67)},
    },
}

DISPLAY_NAMES = {
    'sdl_complexity':      'SDL-Complexity (Ours)',
    'sdl_degree':          'SDL-Degree (Ours)',
    'self_attention':      'Self-Attention',
    'fcgat':               'FCGAT',
    'famcf':               'FAMCF',
    'androfim':            'AndroFIM',
    'pretrained_backbone': 'GraphCL Backbone',
}

PARAMS = {
    'sdl_complexity':      (1_584_898,   2),
    'sdl_degree':          (1_584_898,   2),
    'self_attention':      (1_880_577,   295_681),
    'fcgat':               (532_264,     532_264),
    'famcf':               (86_336,      86_336),
    'androfim':            (7_996_800,   7_996_800),
    'pretrained_backbone': (1_584_896,   1_584_896),
}

# wall_mean, wall_std, gpu_mean, gpu_std, peak_mem_avg, peak_mem_max (ms / MB)
RESOURCES = {
    'sdl_complexity':      (29.51,      8.30,  29.56,      8.30,  1075.5,  2064.5),
    'sdl_degree':          (29.63,      8.34,  29.68,      8.34,  1075.5,  2064.5),
    'self_attention':      (31.16,     11.34,  31.21,     11.34,  1075.5,  2064.5),
    'fcgat':               (14.04,      3.45,  14.08,      3.45,   613.5,  1086.6),
    'famcf':               ( 2.69,      0.63,   2.73,      0.63,   192.0,   218.4),
    'androfim':            (62778.25, 43427.31, 62777.93, 43427.02, 9306.2, 16180.2),
    'pretrained_backbone': (40.16,     17.58,  40.20,     17.58,  1394.5,  3916.0),
}

# Table row order
ROWS = [
    'SDL Variants (Ours)',
    'sdl_complexity',
    'sdl_degree',
    'Other Baselines',
    'self_attention',
    'fcgat',
    'famcf',
    'pretrained_backbone',
    'androfim',
]

YEARS = ['2019', '2020', '2021', 'combined']

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

HEADER_FILL = PatternFill('solid', fgColor='1F4E79')
SUBHDR_FILL = PatternFill('solid', fgColor='2E75B6')
SHOT_FILL   = PatternFill('solid', fgColor='2F5496')   # mid blue for 1-shot/5-shot bands
OUR_FILL    = PatternFill('solid', fgColor='E2EFDA')
ALT_FILL    = PatternFill('solid', fgColor='F2F2F2')
WHITE_FILL  = PatternFill('solid', fgColor='FFFFFF')

HEADER_FONT = Font(name='Calibri', bold=True, color='FFFFFF', size=11)
SHOT_FONT   = Font(name='Calibri', bold=True, color='FFFFFF', size=10)
SUBHDR_FONT = Font(name='Calibri', bold=True, color='FFFFFF', size=10)
BODY_FONT   = Font(name='Calibri', size=10)

CENTER = Alignment(horizontal='center', vertical='center', wrap_text=True)
LEFT   = Alignment(horizontal='left',   vertical='center')
RIGHT  = Alignment(horizontal='right',  vertical='center')

THIN = Side(style='thin',   color='BFBFBF')
MED  = Side(style='medium', color='595959')

def _border():
    return Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

def _hcell(ws, row, col, value, fill=None, font=None, align=CENTER):
    c = ws.cell(row=row, column=col, value=value)
    c.fill      = fill or HEADER_FILL
    c.font      = font or HEADER_FONT
    c.alignment = align
    c.border    = _border()
    return c

def _bcell(ws, row, col, value, fill=None, align=CENTER, fmt=None):
    c = ws.cell(row=row, column=col, value=value)
    c.fill      = fill or WHITE_FILL
    c.font      = BODY_FONT
    c.alignment = align
    c.border    = _border()
    if fmt:
        c.number_format = fmt
    return c


# ---------------------------------------------------------------------------
# Accuracy sheet
# ---------------------------------------------------------------------------
# Column layout (1-indexed):
#   1        : Model
#   2-3      : 2019  1-shot (acc, ±CI)
#   4-5      : 2020  1-shot
#   6-7      : 2021  1-shot
#   8-9      : Combined 1-shot
#   10-11    : 2019  5-shot
#   12-13    : 2020  5-shot
#   14-15    : 2021  5-shot
#   16-17    : Combined 5-shot

def build_accuracy_sheet(ws):
    ws.title = 'Accuracy'
    TITLE = 'Temporal Evaluation — HiGraph (5-way, 100 episodes/year, train 2012-2018, test 2019-2021)'
    NOTE  = 'SDL rows highlighted in green.  All families in test years evaluated (open-set split).  Combined = pooled across 2019-2021.'

    # Row 1: title
    ws.merge_cells('A1:Q1')
    c = ws['A1']
    c.value     = TITLE
    c.font      = Font(name='Calibri', bold=True, size=13, color='1F4E79')
    c.alignment = LEFT

    # Row 2: note
    ws.merge_cells('A2:Q2')
    c = ws['A2']
    c.value     = NOTE
    c.font      = Font(name='Calibri', italic=True, size=9, color='595959')
    c.alignment = LEFT

    # Row 3: 1-shot / 5-shot band headers
    ws.merge_cells(start_row=3, start_column=2, end_row=3, end_column=9)
    _hcell(ws, 3, 2, '1-Shot Accuracy (%)', fill=SHOT_FILL, font=SHOT_FONT)
    ws.merge_cells(start_row=3, start_column=10, end_row=3, end_column=17)
    _hcell(ws, 3, 10, '5-Shot Accuracy (%)', fill=SHOT_FILL, font=SHOT_FONT)
    _hcell(ws, 3, 1, '')  # blank corner

    # Row 4: column sub-headers
    _hcell(ws, 4, 1, 'Model')
    for base_col, label in [(2, '2019'), (4, '2020'), (6, '2021'), (8, 'Combined'),
                             (10, '2019'), (12, '2020'), (14, '2021'), (16, 'Combined')]:
        _hcell(ws, 4, base_col,     label)
        _hcell(ws, 4, base_col + 1, '±95% CI')

    data_row = 5
    alt = False
    for item in ROWS:
        if item not in ACC_TEMPORAL:
            # Section header
            ws.merge_cells(start_row=data_row, start_column=1, end_row=data_row, end_column=17)
            c = ws.cell(row=data_row, column=1, value=item)
            c.fill      = SUBHDR_FILL
            c.font      = SUBHDR_FONT
            c.alignment = LEFT
            c.border    = _border()
            data_row += 1
            alt = False
            continue

        key  = item
        name = DISPLAY_NAMES.get(key, key)
        is_ours = key.startswith('sdl_')
        fill = OUR_FILL if is_ours else (ALT_FILL if alt else WHITE_FILL)
        alt  = not alt

        _bcell(ws, data_row, 1, name, fill=fill, align=LEFT)

        shot_data = ACC_TEMPORAL[key]
        for shot, base_col in [('1shot', 2), ('5shot', 10)]:
            for yi, year in enumerate(YEARS):
                mean, ci = shot_data[shot][year]
                col = base_col + yi * 2
                _bcell(ws, data_row, col,     mean, fill=fill, fmt='0.00')
                _bcell(ws, data_row, col + 1, ci,   fill=fill, fmt='0.00')

        data_row += 1

    # Column widths
    ws.column_dimensions['A'].width = 28
    for col in range(2, 18):
        ws.column_dimensions[get_column_letter(col)].width = 9

    ws.freeze_panes = 'B5'
    ws.row_dimensions[1].height = 22
    ws.row_dimensions[2].height = 14
    ws.row_dimensions[3].height = 20
    ws.row_dimensions[4].height = 28


# ---------------------------------------------------------------------------
# Resource sheet
# ---------------------------------------------------------------------------

def build_resource_sheet(ws):
    ws.title = 'Resources'

    ws.merge_cells('A1:I1')
    c = ws['A1']
    c.value     = 'Resource Usage (shared across splits — from HiGraph no-overlap run)'
    c.font      = Font(name='Calibri', bold=True, size=13, color='1F4E79')
    c.alignment = LEFT

    headers = ['Model', 'Total Params', 'Head Params',
               'Wall Time (ms)', 'Wall ±Std',
               'GPU Time (ms)',  'GPU ±Std',
               'Peak Mem Avg (MB)', 'Peak Mem Max (MB)']
    for ci, h in enumerate(headers, 1):
        _hcell(ws, 3, ci, h)

    row_idx = 4
    alt = False
    for item in ROWS:
        if item not in PARAMS:
            ws.merge_cells(start_row=row_idx, start_column=1, end_row=row_idx, end_column=9)
            c = ws.cell(row=row_idx, column=1, value=item)
            c.fill      = SUBHDR_FILL
            c.font      = SUBHDR_FONT
            c.alignment = LEFT
            c.border    = _border()
            row_idx += 1
            alt = False
            continue

        key  = item
        name = DISPLAY_NAMES.get(key, key)
        is_ours = key.startswith('sdl_')
        fill = OUR_FILL if is_ours else (ALT_FILL if alt else WHITE_FILL)
        alt  = not alt

        tp, hp = PARAMS[key]
        wm, ws_, gm, gs_, pm, pmax = RESOURCES[key]

        vals  = [name, tp, hp, wm, ws_, gm, gs_, pm, pmax]
        fmts  = [None, '#,##0', '#,##0', '0.00', '0.00', '0.00', '0.00', '0.0', '0.0']
        aligns = [LEFT] + [RIGHT] * 8
        for ci, (v, fm, al) in enumerate(zip(vals, fmts, aligns), 1):
            _bcell(ws, row_idx, ci, v, fill=fill, align=al, fmt=fm)

        row_idx += 1

    ws.column_dimensions['A'].width = 28
    for col, w in [('B', 14), ('C', 13), ('D', 14), ('E', 10),
                   ('F', 14), ('G', 10), ('H', 18), ('I', 18)]:
        ws.column_dimensions[col].width = w

    ws.freeze_panes = 'B4'
    ws.row_dimensions[1].height = 22
    ws.row_dimensions[3].height = 32


# ---------------------------------------------------------------------------

def main():
    wb = Workbook()
    ws_acc = wb.active
    ws_res = wb.create_sheet()

    build_accuracy_sheet(ws_acc)
    build_resource_sheet(ws_res)

    path = os.path.join(OUT_DIR, 'table_temporal.xlsx')
    wb.save(path)
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
