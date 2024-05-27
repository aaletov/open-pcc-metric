import matplotlib.axes
import matplotlib.axis
import matplotlib.figure
import matplotlib.ticker
import pandas as pd
import numpy as np
import numpy.polynomial.hermite as herm
import scipy.optimize as spo
import pwlf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import dataclasses
import enum

class Scale(enum.Enum):
    LINEAR = 1
    LOG = 2


@dataclasses.dataclass
class Metric:
    plot_label: str
    scale: Scale


metrics_dict = {
    # "algs": "",
    # "datasets": "",
    # "rate": "",
    # "pc_file": "",
    # "encT": "",
    # "decT": "",
    "bpp": Metric("Битрейт (bpp)", scale=Scale.LINEAR),
    "y_psnr": Metric("Y-PSNR (Дб)", scale=Scale.LINEAR),
    "cb_psnr": Metric("Cb-PSNR (Дб)", scale=Scale.LINEAR),
    "cr_psnr": Metric("Cr-PSNR (Дб)", scale=Scale.LINEAR),
    "cd_p2pt": Metric("Расстояние Чамфера", scale=Scale.LOG),
    "cdpsnr_p2pt": Metric("CD-PSNR (Дб)", scale=Scale.LINEAR),
    "h_p2pt": Metric("Метрика Хаусдорфа", scale=Scale.LOG),
    "cd_p2pl": Metric("Расстояние Чамфера (проектив.)", scale=Scale.LOG),
    "cdpsnr_p2pl": Metric("CD-PSNR (проектив.) (Дб)", scale=Scale.LINEAR),
    "h_p2pl": Metric("Метрика Хаусдорфа (проектив.)", scale=Scale.LOG),
    "y_cpsnr": Metric("Y-PSNR (Дб)", scale=Scale.LINEAR),
    "u_cpsnr": Metric("Cb-PSNR (Дб)", scale=Scale.LINEAR),
    "v_cpsnr": Metric("Cr-PSNR (Дб)", scale=Scale.LINEAR),
}

def approx(
    df: pd.DataFrame,
    alg: str,
    variable: str,
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
) -> None:
    alg_df = df[df["algs"] == alg].sort_values(by="bpp")

    bpp = alg_df["bpp"].to_numpy(dtype="float64")
    vals = alg_df[variable].to_numpy(dtype="float64")

    metric = metrics_dict[variable]

    approx_vals = []
    if metric.scale == Scale.LINEAR:
        fun = lambda t, a, b: a * np.log10(t) + b
        coefs = spo.curve_fit(fun,  bpp, vals)[0]
        approx_vals = np.array([fun(t, coefs[0], coefs[1]) for t in bpp])
    elif metric.scale == Scale.LOG:
        fun = lambda t, a, b: 1 / (a * t + b)
        coefs = spo.curve_fit(fun,  bpp, vals)[0]
        approx_vals = np.array([fun(t, coefs[0], coefs[1]) for t in bpp])

    ax.plot(bpp, approx_vals, label=alg)
    ax.scatter(bpp, vals)

    if metric.scale == Scale.LINEAR:
        get_max = lambda x: 5 * ((x // 5) + 1)
        ax.set_xlim(0.0, get_max(df["bpp"].max()))
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=5))
        ax.set_ylim(0.0, get_max(df[variable].max()))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=5))
    elif metric.scale == Scale.LOG:
        get_max = lambda x: 5 * ((x // 5) + 1)
        ax.set_xlim(0.0, get_max(df["bpp"].max()))
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=5))
        ax.set_ylim(1e-2, 1e+5)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatter())
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]))

    ax.xaxis.set_label_text("Битрейт (bpp)")
    ax.yaxis.set_label_text(metric.plot_label)

    ax.grid(True)
    ax.legend()


if __name__ == "__main__":
    df = pd.read_csv("summary.csv")
    df["algs"].replace(to_replace={"GPCC": "TMC13"}, inplace=True)

    algs = df["algs"].unique()

    for metric in metrics_dict.keys():
        fig, ax = plt.subplots()
        for alg in algs:
            approx(df, alg, metric, fig, ax)
        fig.savefig(f"approx_{metric}.png")

