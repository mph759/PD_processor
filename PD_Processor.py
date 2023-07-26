import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path


class PDDatset:

    def __init__(self, dataset_root, target_dataset, target_subtag):
        self.dataset_root = dataset_root
        self.target_dataset = target_dataset
        self.target_subtag = target_subtag
        self.manifest = []
        self.get_manifest()
        print(self.manifest)

    def get_manifest(self):
        '''
        Find all matching files with the extention '.xye' in the directory.
        :return: A manifest of all found files
        '''
        raw_list = sorted(Path(self.dataset_root + '\\' + self.target_dataset).glob(f'*{self.target_subtag}*.xye'))
        # print(Path(self.dataset_root + '\\' + self.target_dataset))
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
        plt.xlabel('2$\\theta$')
        plt.ylabel('Intensity')
        # plt.plot(xye_array[:, 0], np.log10(xye_array[:, 1]), linewidth=0.5)
        plt.plot(xye_array[:, 0], xye_array[:, 1], linewidth=0.2)

    def generate_waterfall(self, step: int = 1000):
        '''
        Generate a plot of all patterns in the manifest on a single figure, with a spacing step between each plot
        :param step: The step amount for the gap between line plots
        :return:
        '''
        for indx, pattern in enumerate(self.manifest):
            xye_array = np.loadtxt(pattern)
            xye_array[:, 1] += (indx + 1) * step
            # plt.plot(xye_array[:, 0], np.log10(xye_array[:, 1]), linewidth=1, label=f'{indx}')
            plt.plot(xye_array[:, 0], xye_array[:, 1], linewidth=1)
            plt.title(f'{pattern}')
        plt.legend()
        plt.xlabel('2$\\theta$')
        plt.ylabel('Intensity')
        plt.gca().get_yaxis().set_ticks([])
        plt.show()

    def generate_heatmap(self, temp_range: list, lim: int = 20000, ):
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
            xye_int_slice = xye_array[:, 1]
            wf_array[:, indx] = xye_int_slice
        temp_range.sort()
        shape = [0, xye_array[lim - 1, 0], temp_range[0], temp_range[1]]
        plt.figure()
        plt.imshow(np.transpose(wf_array), aspect=0.1, extent=shape, origin='lower')
        plt.xlabel('2$\Theta$ / $^\circ$')
        plt.ylabel('Run No.')
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

    def inspect_multi_pattern(self, title: str, x_range: list, twotheta: bool = True):
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
                xye_array[:, 0] = ((4 * np.pi) / wavelength) * np.sin(np.deg2rad(xye_array[:, 0]) / 2)
            plt.plot(xye_array[:, 0], xye_array[:, 1], linewidth=1)
            plt.title(f'{title}')
            if twotheta:
                plt.xlabel('2$\\theta$')
            else:
                plt.xlabel('q')
            plt.ylabel('Intensity')
            plt.xlim(x_range)
            plt.gca().get_yaxis().set_ticks([])
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    print('Powder Diffraction Processing')
    dataset_root = r'C:/Users/Michael_X13/OneDrive - RMIT University/Beamtime/18855_PD_Bryant/Data/sorted/DESs/'

    target_dataset = 'ChClEG_1to20_VT1'     # the name in the listed file. Can use * as a wildcard
    target_subtag = 'p12_ns'                # Can be left blank
    wavelength = 0.826278E-10               # wavelength in metres. Only needed if converting from 2theta to q
    dset = PDDatset(dataset_root, target_dataset, target_subtag)
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
    dset.generate_waterfall(step=5000)
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
