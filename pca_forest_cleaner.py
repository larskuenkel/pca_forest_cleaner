#!/usr/bin/env python

# Tool to remove RFI from pulsar archives using principal component analysis and the isolation forsest algorithm.
# Originally written by Lars Kuenkel

import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import psrchive
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest



def parse_arguments():
    parser = argparse.ArgumentParser(description='Commands for the cleaner')
    parser.add_argument('archive', nargs='+', help='The chosen archives')
    parser.add_argument('-n', '--components', type=int, default=1024, help='Number of pca components.')
    parser.add_argument('-e', '--estimators', type=int, default=100, help='Number of tree estimators.')
    parser.add_argument('-s', '--samples', type=float, default=1.0, help='Fraction of samples that trains each estimators.\
        If 1.0 all samples are used')
    parser.add_argument('-z', '--print_zap', action='store_true', help='Creates a plot that shows which profiles get zapped.')
    parser.add_argument('-v', '--verbose', action='store_false', help='Reduces the verbosity of the program.')
    parser.add_argument('-o', '--output', type=str, default='',
        help="Name of the output file. If set to 'std' the pattern NAME.FREQ.MJD.ar will be used.")
    args = parser.parse_args()
    return args


def main(args):
    for arch in args.archive:
        if args.verbose:
            print "Input archive: %s" % arch
        ar = psrchive.Archive_load(arch)
        if args.output == '':
            orig_name = str(ar).split(':', 1)[1].strip()
            o_name = orig_name + '_cleaned.ar'
        else:
            if args.output == 'std':
                mjd = (float(ar.start_time().strtempo()) + float(ar.end_time().strtempo())) / 2.0
                name = ar.get_source()
                cent_freq = ar.get_centre_frequency()
                o_name = "%s.%.3f.%f.ar" % (name, cent_freq, mjd)
            else:
                o_name = args.output
        ar = clean(ar, args, arch)
        ar.unload(o_name)
        print "Cleaned archive: %s" % o_name


def clean(ar, args, arch):
    """Cleans the archive and returns the cleaned copy.
    """
    ar_name = ar.get_filename().split()[-1]

    # Create copy of archive that is used to grab the profiles
    patient = ar.clone()
    patient.pscrunch()
    patient.remove_baseline()
    patient.dedisperse()
    # Grab the profiles after dedispersing them
    data = patient.get_data()[:, 0, :, :]
    profile_number = data[:, :, 0].size

    if args.verbose:
        print ("PCA parameters: n_components: %s" % args.components)
        print ("IsolationForest parameters: n_estimators: %s max_samples: %s" % (args.estimators, args.samples))

    orig_shape = np.shape(data)
    # Reshape the profiles for pca computation
    data = np.reshape(data, (-1, orig_shape[2]))

    # Compute the pca
    pca = PCA(n_components=args.components, svd_solver="full")
    data_pca = pca.fit_transform(data)

    # Compute the anomaly scores of the isolation forest algorithm
    # The random_state creates a reproducible result but this may not be the best solution in the future
    clf = IsolationForest(n_estimators=args.estimators, max_samples=args.samples, n_jobs=2, random_state=1)

    clf.fit(data_pca)

    anomaly_factors = clf.decision_function(data_pca)
    anomaly_factors_reshape = np.reshape(anomaly_factors, orig_shape[0:2])
    # plt.plot(np.sort(anomaly_factors))
    # plt.show()

    snrs = []
    split_values = []
    rfi_fracs = []
    # Cycle through different rfi fractions and find the best snr
    for rfi_frac in np.linspace(0, 40, num=100):
        split_value = np.percentile(anomaly_factors, rfi_frac)
        test_profile = np.sum(data[anomaly_factors > split_value, :], axis=0)
        profile_object = psrchive.Profile(orig_shape[2])
        profile_object.get_amps()[:] = test_profile
        test_snr = profile_object.snr()
        snrs.append(test_snr)
        split_values.append(split_value)
        rfi_fracs.append(rfi_frac)
        # print test_snr

    best_index = np.argmax(snrs)
    best_snr = snrs[best_index]
    best_frac = rfi_fracs[best_index]
    best_split_value = split_values[best_index]

    if args.verbose:
        print "Best SNR: %.1f RFI fraction: %.2f" % (best_snr, best_frac * 0.01)

    # Set the weights in the archive
    set_weights_archive(ar, anomaly_factors_reshape, best_split_value)

    # Create plot that shows zapped( red) and unzapped( blue) profiles if needed
    if args.print_zap:
        plt.imshow(anomaly_factors_reshape, vmin=best_split_value, vmax=best_split_value + 0.0001, aspect='auto',
                interpolation='nearest', cmap=cm.coolwarm)
        plt.gca().invert_yaxis()
        plt.savefig("%s_%s_%s_%s.png" % (ar_name, args.components, args.estimators, args.samples), bbox_inches='tight')

    # Create log that contains the used parameters
    with open("clean.log", "a") as myfile:
        myfile.write("\n %s: Cleaned %s with %s"
        % (datetime.datetime.now(), ar_name, args))
    return ar


def set_weights_archive(archive, anomaly_values, split_value):
    """Apply the weights to an archive according to the classfication result.
    """
    for (isub, ichan) in np.argwhere(anomaly_values < split_value):
        integ = archive.get_Integration(int(isub))
        integ.set_weight(int(ichan), 0.0)


if __name__=="__main__":
    args = parse_arguments()
    main(args)
