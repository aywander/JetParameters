import matplotlib.pyplot as plt
from collections import deque
from cycler import cycler


def adjust_rcParams(style='seaborn-v0_8', use_kpfonts=False, dark_mode=False, grid=True):

    plt.style.use(style)

    if dark_mode:
        fc = 'white'
        fc_i = 'black'
        fc_n = (1, 1, 1, 0.3)
        # TODO: Make the colorpalette below a bit lighter
        tableau10_colors = deque(['006BA4', 'FF800E', 'ABABAB', '595959', '5F9ED1', 'C85200', '898989', 'A2C8EC', 'FFBC79',
                            'CFCFCF'])

    else:
        fc = 'black'
        fc_i = 'white'
        fc_n = (0, 0, 0, 0.1)
        tableau10_colors = deque(['006BA4', 'FF800E', 'ABABAB', '595959', '5F9ED1', 'C85200', '898989', 'A2C8EC', 'FFBC79',
                            'CFCFCF'])

    tableau10_colors.rotate(0)

    plt.rcParams['axes.prop_cycle'] = cycler(color=['#' + s for s in tableau10_colors])

    plt.rcParams.update({
        'text.color': fc,
        'axes.labelcolor': fc,
        'xtick.major.size': 3.5 if not grid else 0.,
        'ytick.major.size': 3.5 if not grid else 0.,
        'xtick.major.width': 0.6 if not grid else 0.,
        'ytick.major.width': 0.6 if not grid else 0.,
        'xtick.minor.width': 0.4 if not grid else 0.,
        'ytick.minor.width': 0.4 if not grid else 0.,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.spines.left': not grid,
        'axes.spines.bottom': not grid,
        'axes.spines.top': not grid,
        'axes.spines.right':not grid,
        'axes.linewidth': 0.8 if not grid else 0.,
        'axes.edgecolor': fc,
        'xtick.color': fc,
        'ytick.color': fc,
        'grid.color': fc_i,
        # 'axes.axisbelow': False,
        'grid.alpha': 0.5,
        'axes.facecolor': fc_n,
        'axes.grid' : grid,
        # 'axes.grid.which': 'both',
        # 'axes.grid.axis': 'both',
        'figure.facecolor': (0, 0, 0, 0),
        # 'figure.edgecolor': 'black',
        'savefig.facecolor': (0, 0, 0, 0),
        'figure.dpi': 300
    })

    if use_kpfonts:
        plt.rcParams.update({
            'font.family': 'serif',
            'text.usetex': True,
            'text.latex.preamble': [
                r'\usepackage{amsmath}',
                r'\usepackage{amssymb}',
                r'\usepackage{siunitx}',
                r'\usepackage[notextcomp]{kpfonts}',
            ],
        })

    else:
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': 'Times New Roman',
            'font.sans-serif': 'Times New Roman',
            'mathtext.fontset': 'cm',
        })

