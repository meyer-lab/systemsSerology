#!/usr/bin/env python3
import sys
import logging
import time
import matplotlib
from syserol.figures.common import overlayCartoon

matplotlib.use("AGG")

fdir = "./output/"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if __name__ == "__main__":
    nameOut = "figure" + sys.argv[1]

    start = time.time()

    exec("from syserol.figures." + nameOut + " import makeFigure")
    ff = makeFigure()
    ff.savefig(fdir + nameOut + ".svg", dpi=ff.dpi, bbox_inches="tight", pad_inches=0)

    if sys.argv[1] == '6':
        # Overlay Figure 6a cartoon
        overlayCartoon('./output/figure6.svg','./manuscript/images/figure6a.svg',
                       90, 10, scalee=0.4, scale_x=0.5, scale_y=0.5)

    logging.info("%s is done after %s seconds.", nameOut, time.time() - start)
