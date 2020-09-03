#!/usr/bin/env python3
import sys
import logging
import time
import matplotlib

matplotlib.use("AGG")

fdir = "./"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if __name__ == "__main__":
    for ii in range(1, 7):
        nameOut = "figure" + str(ii)

        start = time.time()

        exec("from syserol.figures." + nameOut + " import makeFigure")
        ff = makeFigure()
        ff.savefig(fdir + nameOut + ".svg", dpi=ff.dpi, bbox_inches="tight", pad_inches=0)

        logging.info("%s is done after %s seconds.", nameOut, time.time() - start)
