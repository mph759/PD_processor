'''
Processing and plotting of synchrotron data
Possible to modify to apply to any 2D data file

Written by Jack Binns and Michael Hassett
Last updated on 2023-08-02
'''

import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import pandas as pd


class PDDatset:

    def __init__(self,
                 dataset_root: str,
                 target_dataset: str,
                 target_subtag: str = '',
                 conversion_factor=None,
                 x_dim=f'2$\Theta$ / $^\circ$',
                 extension: str = '.xye'):
        '''

        :param dataset_root: Primary directory to where general data to be ingested is stored
        :param target_dataset: Folder inside the primary directory where the data is stored
        :param target_subtag: Identifying string in file name of the files you want to plot. Can be an empty string.
        :param extension: File extension for which you are searching for
        '''
        self.dataset_root = dataset_root
        self.target_dataset = target_dataset
        self.target_subtag = target_subtag
        self.ext = extension
        self.conversion_factor = conversion_factor
        self.x_dim = x_dim
        self.manifest = []
        self.__get_manifest__()
        # print(self.manifest)

    def __get_manifest__(self) -> list[Path]:
        '''
        Find all matching files with the extension '.xye' in the directory.
        :return: A list of all found files as Path objects
        '''
        raw_list = sorted(Path(f'{self.dataset_root}\\{self.target_dataset}').glob(f'*{self.target_subtag}*{self.ext}'))
        # print(Path(f'{self.dataset_root}\\{self.target_dataset}'))
        # print(raw_list)
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', str(key))]
        [print(indx, x) for indx, x in enumerate(sorted(raw_list, key=alphanum_key))]
        self.manifest = sorted(raw_list, key=alphanum_key)
        return sorted(raw_list, key=alphanum_key)

    def inspect_pattern(self, pattern_index: int):
        '''
        Plot a single pattern
        :param pattern_index: index in the manifest for the single pattern to be plotted
        :return:
        '''
        xye_array = np.loadtxt(self.manifest[pattern_index])
        # plt.figure()
        plt.xlabel(self.x_dim)
        plt.ylabel('Intensity')
        # plt.plot(xye_array[:, 0], np.log10(xye_array[:, 1]), linewidth=0.5)
        plt.plot(xye_array[:, 0], xye_array[:, 1], linewidth=0.2)

    def generate_waterfall(self,
                           step: int = 1000,
                           type: str = "normal",
                           save: bool = False,
                           save_dir: object = Path.cwd(),
                           save_filename: str = 'foo'):
        '''
        Generate a plot of all patterns in the manifest on a single figure, with a spacing step between each plot
        :param step: The step amount for the gap between line plots
        :param type: Type of treatment to data before plotting. "normal" will plot as-is, "log" will plot log10 of the data
        :param save: Boolean of whether you want to save the file (True), or plot directly (False)
        :param save_dir: Directory to which you will save the figure
        :param save_filename: File name for the figure
        :return:
        '''
        for indx, pattern in enumerate(self.manifest):
            try:
                xye_array = np.loadtxt(pattern)
                if self.conversion_factor is not None:
                    xye_array[:, 0] = np.array(map(self.conversion_factor, xye_array[:, 0]))
            except:
                xye_array = np.loadtxt(pattern, skiprows=1)
            xye_array[:, 1] += (indx + 1) * step
            if type == "normal":
                plt.plot(xye_array[:, 0], xye_array[:, 1], linewidth=1)
            if type == "log":
                plt.plot(xye_array[:, 0], np.log10(xye_array[:, 1]), linewidth=1)
            # else:
            # raise ValueError("type must be 'normal' or 'log'")
            plt.title(f'{pattern}')
        plt.legend()
        plt.xlabel(self.x_dim)
        plt.ylabel('Intensity')
        plt.ylim(0, np.median(xye_array[:, 1]) + 4 * step)
        plt.gca().get_yaxis().set_ticks([])
        if save:
            plt.savefig(f'{save_dir}/{save_filename}')
        else:
            plt.show()

    def generate_heatmap(self, temp_range: list[float],
                         lim: int = 20000,
                         save: bool = False,
                         save_dir=Path.cwd(),
                         save_filename: str = 'foo'):
        """
        Parameters
        ----------
        :lim: The x range of the data in pixels
        """
        wf_array = np.zeros((lim, len(self.manifest)))
        print(wf_array.shape)
        # wf_array = np.loadtxt(self.manifest[0])[:lim]
        # wf_int_slice = wf_array[:, 1]
        # plt.plot(wf_int_slice[:])
        # plt.show()
        print(f'wf_array.shape {wf_array.shape}')
        for indx, pattern in enumerate(self.manifest):
            xye_array = np.loadtxt(pattern)[:lim]
            if self.conversion_factor is not None:
                xye_array[:, 0] = np.array(map(self.conversion_factor, xye_array[:, 0]))
            xye_int_slice = xye_array[:, 1]
            wf_array[:, indx] = xye_int_slice
        temp_range.sort()
        shape = [0, xye_array[lim - 1, 0], temp_range[0], temp_range[1]]
        plt.figure()
        plt.imshow(np.transpose(wf_array), aspect=0.1, extent=shape, origin='lower', vmax=50000)
        plt.xlabel(self.x_dim)
        plt.ylabel('Run No.')
        plt.tight_layout
        if save:
            plt.savefig(f'{save_dir}/{save_filename}.jpg', bbox_inches="tight")
        else:
            plt.show()

    def ensemble_csv_convert(self):
        xye_list = Path(f'{self.dataset_root}\\').glob(f'**\\*.xye')
        print(f'{self.dataset_root}**\\*.xye')
        print(xye_list)
        for ex in xye_list:
            arr = np.loadtxt(ex)
            print(arr.shape)
            new_root = ex[:-3] + 'csv'
            print(new_root)
            np.savetxt(new_root, arr, delimiter=',')

    def inspect_multi_pattern(self,
                              title: str,
                              x_range: list[float] = [],
                              twotheta: bool = True):
        '''
        Method for plotting all single diffraction patterns from the manifest directory.

        :param title: Title to be printed on plot
        :param x_range: Limit the x axis to the set range. Does account for unit/variable
        :param twotheta: If True, it will print using the label 2theta, and convert the data according using the wavelength. If false, will convert to q
        :return:
        '''
        for indx, pattern in enumerate(self.manifest):
            xye_array = np.loadtxt(pattern)
            if not twotheta:
                global wavelength
                try:
                    xye_array[:, 0] = ((4 * np.pi) / wavelength) * np.sin(np.deg2rad(xye_array[:, 0]) / 2)
                except:
                    raise ValueError("wavelength must be given as an int or float")
            plt.plot(xye_array[:, 0], xye_array[:, 1], linewidth=1)
            plt.title(f'{title}')
            if twotheta:
                plt.xlabel('2$\\theta$')
            else:
                plt.xlabel('q')
            plt.ylabel('Intensity')
            if not x_range == []:
                plt.xlim(x_range)
            plt.gca().get_yaxis().set_ticks([])
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    print('Powder Diffraction Processing')
    dataset_root = (
        r'C:\Users\Michael_X13\OneDrive - RMIT University\Research\2021 - Honours Research Project\Synchrotron Beamtime\Data')  # The main directory; leave the last folder off
    target_dataset = [r'DEAF\DEAFwool2\VT',
                      r'EAA\EAA\VT',
                      r'EAF\EAF_2',
                      r'EAG\EAGwool2\CRYST',
                      r'EAL\EALwool\CRYST',
                      r'EAN\EAN\VT2',
                      r'EATFA\EATFA\VT',
                      r'TEAF\TEAFwool\CRYST']  # The subdirectory where the exact data is
    target_subtag = 'p12_ns'  # Text in the listed file. Can be left blank
    file_ext = '.xye'  # File extension for files to be searched for

    wavelength = 0.826278E-10  # wavelength in metres. Only needed if converting from 2theta to q
    # convert_2theta_to_q = lambda array: ((4 * np.pi) / wavelength) * np.sin(np.deg2rad(array) / 2)

    for target in target_dataset:
        filename = target.split("\\")[0]
        dset = PDDatset(dataset_root, target, target_subtag)
        # , conversion_factor=convert_2theta_to_q, x_dim='q', extension=file_ext)
        dset.generate_waterfall(step=10000, type="normal",
                                save=True,
                                save_dir=r'C:\Users\Michael_X13\OneDrive - RMIT University\Research\2021 - Honours Research Project\Paper\Figures\Waterfall plots\Normal',
                                save_filename=f'{filename}_normalwaterfall')

        dset.generate_heatmap(temp_range=[120, 310], lim=10000,
                              save=True,
                              save_dir=r'C:\Users\Michael_X13\OneDrive - RMIT University\Research\2021 - Honours Research Project\Paper\Figures\Waterfall plots\Colour map',
                              save_filename=f'{filename}_2Dwaterfall')

    # dset.generate_waterfall(step=5000, type="normal")
    # dset.ensemble_csv_convert()
    # dset.inspect_pattern(0)
    # dset.inspect_pattern(30)
    # dset.inspect_pattern(35)
    # dset.inspect_pattern(32)
    # dset.inspect_pattern(30)
    # dset.inspect_pattern(1)
    # dset.inspect_pattern(4)

    # [dset.inspect_pattern(q) for q in range(25,32)]
    # plt.show()
    # dset.generate_waterfall(step=5000)
    # dset.generate_heatmap([259,327], lim=2250)

    # dset.inspect_multi_pattern(target_dataset, [0,40])

    """
    Compare
    """
    # # dataset_root2 = f'E:\\RMIT\\PIL-pxrd\\Greaves_16233\\'
    # dataset_root2 = f'E:\\RMIT\\PIL-pxrd\\Greaves_16598\\frames\\'
    #
    # target_dataset2 = 'EAA'
    # target_subtag2 = ''                  # Can be left blank
    # dset2 = PDDatset(dataset_root2, target_dataset2, target_subtag2)
    # dset2.inspect_pattern(0)
    # # dset2.inspect_pattern(1)
    # # dset2.generate_waterfall(step=500)

    #
    # dataset_root3 = f'E:\\RMIT\\PIL-pxrd\\Greaves_16233\\'
    # dataset_root3 = f'E:\\RMIT\\PIL-pxrd\\Greaves_16598\\frames\\'
    # #
    # target_dataset3 = 'EAN10_2'
    # target_subtag3 = ''                  # Can be left blank
    # #
    # dset3 = PDDatset(dataset_root3, target_dataset3, target_subtag3)
    # dset3.inspect_pattern(0)
    # # dset3.generate_waterfall(step=500)
    # #
    # plt.show()
