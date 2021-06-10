#!/usr/bin/env python

# imports:
import sys
import os
import numpy as np
import pandas as pd
import tools
import pylab as matlib
import matplotlib.pyplot as plt
import argparse

from pdastro import pdastroclass
from test2 import test2class


# class:
class testclass(test2class):
    def __init__(self):
        test2class.__init__(self)

        # define vars
        self.testvar = 0
        self.testvar2 = 'sjajdjdjdj'

    def defineOptions(self, parser=None, usage=None, conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)

        parser.add_argument('run_loadLightCurve', default=False)
        parser.add_argument()

        return parser

    def printvars(self, x=False):
        # print(self.testvar)
        # print(self.testvar2)

        # sys.exit(0)

        if x is True:
            print('x has been passed as True')
        elif x is False:
            print('x has been set to False, the default')
        else:
            print('x has been passed as ' + x)

    def function(self):
        print('hello')

    def makeArray(self):
        print(np.array([1, 2, 3, 4, 5]))
        print(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))

        print(np.full(4, 1))
        print(np.full((3, 4), 1))

        print(np.zeros((2, 3), dtype=int))

    def makeDataFrame(self):
        df = pd.DataFrame(data={'col1': ['a', 'b', 'c'], 'col2': [3, 4, 5]})
        df = pd.DataFrame(np.array([['a', 3], ['b', 4], ['c', 5]]), columns=['column1', 'column2'])
        print(df)
        print(df.loc[[0, 1]])
        print(df.values)

    def coordinateConverter(self, ra, dec):
        raDegrees = tools.RaInDeg(ra)
        decDegrees = tools.DecInDeg(dec)
        return raDegrees, decDegrees

    def loadLightCurve(self):
        example_lc = pdastroclass()
        example_lc.load_spacesep('example_lc.txt', delim_whitespace=True)
        print(example_lc.t)

        indices_without_nans = example_lc.ix_remove_null(colnames=['m', 'dm', 'uJy', 'duJy'])
        print(example_lc.t.loc[indices_without_nans])

        indices_without_nans_limits = example_lc.ix_inrange(colnames=['MJD'], lowlim=57357, uplim=57359,
                                                            indices=indices_without_nans)
        print(example_lc.t.loc[indices_without_nans_limits])

        indicesLimits = example_lc.ix_inrange(colnames=['MJD'], lowlim=57357, uplim=57358, indices=indices_without_nans)

        example_lc.write(filename='example_lc_modified.txt', indices=indices_without_nans)

        plt.figure(0)

        sp, plotx, dplot = self.dataPlot(example_lc.t['MJD'], example_lc.t['uJy'], dy=example_lc.t['duJy'])
        # plt.show()

        matlib.setp(plotx, ms=5, color='cyan', marker='s', mfc='white')
        plt.title("Example Light Curve")
        plt.xlabel('MJD')
        plt.ylabel('uJy')
        plt.xlim(57356, 57357)
        plt.ylim(-200, 200)
        plt.axhline(linewidth=1, color='k')
        plt.savefig('ExamplePlot.png', dpi=100)

        plt.figure(1)
        sp, ploty, dplot = self.dataPlot(example_lc.t.loc[indicesLimits]['MJD'], example_lc.t.loc[indicesLimits]['uJy'], dy=example_lc.t.loc[indicesLimits]['duJy'])
        plt.show()



    def dataPlot(self, x, y, dx=None, dy=None, sp=None, label=None, fmt='bo', ecolor='k', elinewidth=None, barsabove=False,
                 capsize=1, logx=False, logy=False):
        if sp == None:
            sp = matlib.subplot(111)
        if dx is None and dy is None:
            if logy:
                if logx:
                    plot, = sp.loglog(x, y, fmt)
                else:
                    plot, = sp.semilogy(x, y, fmt)
            elif logx:
                plot, = sp.semilogx(x, y, fmt)
            else:
                if barsabove:
                    plot, dplot, dummy = sp.errorbar(x, y, label=label, fmt=fmt, capsize=capsize, barsabove=barsabove)
                else:
                    plot, = sp.plot(x, y, fmt)
            return sp, plot, None
        else:
            if logy:
                sp.set_yscale("log", nonposx='clip')
            if logx:
                sp.set_xscale("log", nonposx='clip')
            plot, dplot, dummy = sp.errorbar(x, y, xerr=dx, yerr=dy, label=label, fmt=fmt, ecolor=ecolor,
                                             elinewidth=elinewidth, capsize=capsize, barsabove=barsabove)
            return sp, plot, dplot

    def newLightCurve(self):
        new_lc = pdastroclass()
        new_lc.load_spacesep('2019vxm.o.light.txt', delim_whitespace=True)

        indices_above_100 = new_lc.ix_inrange(colnames=['duJy'], lowlim=100, uplim=None)
        indices_below_100 = new_lc.ix_inrange(colnames=['duJy'], lowlim=None, uplim=100)

        plt.figure(0)

        sp, plot1, dplot = self.dataPlot(new_lc.t.loc[indices_above_100, 'MJD'], new_lc.t.loc[indices_above_100, 'uJy'], dy=new_lc.t.loc[indices_above_100, 'duJy'])
        matlib.setp(plot1, ms=5, color='cyan', marker='s', mfc='white')
        plt.title("Second Light Curve")
        plt.xlabel('MJD')
        plt.ylabel('uJy')
        plt.axhline(linewidth=1, color='k')
        # plt.xlim(58000, 58500)
        # plt.ylim(-2500, 10000)

        sp, plot2, dplot = self.dataPlot(new_lc.t.loc[indices_below_100, 'MJD'], new_lc.t.loc[indices_below_100, 'uJy'], dy=new_lc.t.loc[indices_below_100, 'duJy'])
        matlib.setp(plot2, ms=5, color='cyan', marker='s')

        plt.show()

        # plt.savefig('ExamplePlot.png', dpi=150)


# basically this is like java main
if __name__ == '__main__':
    # make instance of class
    blaa = testclass()

    blaa.newLightCurve()
    # blaa.makeDataFrame()
    #
    # parser = blaa.defineOptions()
    # args = parser.parse_args()
    # RA = '22:37:36.23'
    # Dec = '+35:00:07.68'
    # print(RA, Dec)
    # RA, Dec = blaa.coordinateConverter(RA, Dec)
    # print(RA, Dec)

    # if args.run_loadLightCurve is True:
    #     blaa.loadLightCurve()

    # blaa.makeArray()

    # blaa.function()
    # print('x is default:')
    # blaa.printvars()
    # print('x is True:')
    # blaa.printvars(x=True)
    # print('x is a random str:')
    # blaa.printvars(x='randomkdkdkdkdk')
    # blaa.randomfunction()
