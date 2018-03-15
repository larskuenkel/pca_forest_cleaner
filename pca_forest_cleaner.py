#!/usr/bin/env python

# Tool to remove RFI from pulsar archives using principal component analysis and the isolation forest algorithm.
# Originally written by Lars Kuenkel

from __future__ import print_function

import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import psrchive
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy.signal import medfilt
# import time



def parse_arguments():
    parser = argparse.ArgumentParser(description='Commands for the cleaner')
    parser.add_argument('archive', nargs='+', help='The chosen archives')
    parser.add_argument('-n', '--components', type=int, default=256, help='Number of pca components.')
    parser.add_argument('-e', '--estimators', type=int, default=100, help='Number of tree estimators.')
    parser.add_argument('-s', '--samples', type=float, default=1.0, help='Fraction of samples that trains each estimators.\
        If 1.0 all samples are used.')
    parser.add_argument('-x', '--max_features', type=float, default=1.0, help='Fraction of features that are used in each estimator.')
    parser.add_argument('-f', '--features', action='store_true', help='Add additional feature to the pca features (std, mean, ptp).')
    parser.add_argument('-p', '--partition', type=int, default=[1], nargs='+',
        help='Partitions the profiles into a number of parts for the calculation of the -f parameters\
                                                                Multiple Values can be used. Default: 1.')
    parser.add_argument('-m', '--metrics', type=str, default=['std', 'fft', 'mean', 'ptp'], nargs='+', help='Choose which additional features are computed by -f.\
        Available: mean, fft, std, ptp.')
    parser.add_argument('-a', '--additional', type=float, default=0.0, help='Remove up to -a percent more profiles than at the point with highest SNR.\
                                Will lower SNR but might remove RFI profiles.')
    parser.add_argument('-d', '--disable_pca', action='store_true', help='Do not use pca features, ony use additional features.')
    parser.add_argument('-c', '--contamination_plot', action='store_true', help='Show plot that shows SNR at different contamination levels.')
    parser.add_argument('-z', '--print_zap', action='store_true', help='Creates a plot that shows which profiles get zapped.')
    parser.add_argument('-b', '--bandpass', action='store_true', help='Use bandpass correction.')
    parser.add_argument('-q', '--quiet', action='store_true', help='Reduces the verbosity of the program.')
    parser.add_argument('-o', '--output', type=str, default='',
        help="Name of the output file. If set to 'std' the pattern NAME.FREQ.MJD.ar will be used.")
    parser.add_argument('-r', '--order', action='store_true', help='Use the computed features in the pca calculation.')
    parser.add_argument('-w', '--weight', action='store_true', help='Do not use profiles which already have weight 0.')
    args = parser.parse_args()
    return args


def main(args):
    for arch in args.archive:
        if not args.quiet:
            print ("Input archive: %s" % arch)
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
        print ("Cleaned archive: %s" % o_name)


def clean(ar, args, arch):
    """Cleans the archive and returns the cleaned copy.
    """
    ar_name = ar.get_filename().split()[-1]
    # Create copy of archive that is used to grab the profiles
    if args.bandpass:
        patient = calibrate_bandpass(ar)
    else:
        patient = ar.clone()
        patient.pscrunch()
        patient.remove_baseline()
    # Grab the profiles after dedispersing them
    patient.dedisperse()
    data = patient.get_data()[:, 0, :, :]
    profile_number = data[:, :, 0].size
    pca_components = min(args.components, data.shape[2])

    if not args.quiet:
        print ("Number of Profiles: %s" % profile_number)
        if not args.disable_pca:
            print ("PCA parameters: n_components: %s" % pca_components)
        print ("IsolationForest parameters: n_estimators: %s max_samples: %s max_features: %s" % (args.estimators, args.samples, args.max_features))

    orig_shape = np.shape(data)
    # Reshape the profiles for pca computation


    data = np.reshape(data, (-1, orig_shape[2]))


    # Delete 
    if args.weight:
        orig_weights = ar.get_weights().flatten()
        known_rfi = np.where(orig_weights == 0)
        known_non_rfi = np.where(orig_weights != 0)
        data = np.delete(data, known_rfi, axis=0)

    # Compute additional features if wanted
    if args.features or args.disable_pca:
        array_feat = compute_metrics(data)

    if args.order:
        data = np.concatenate((data, array_feat), axis =1)

    # Compute the pca
    if not args.disable_pca:
        pca = PCA(n_components=pca_components, svd_solver="full")
        data_pca = pca.fit_transform(data)
        data_features = data_pca
        if args.features and not args.order:
            data_features = np.concatenate((data_features, array_feat), axis =1)
    else:
        data_features = array_feat

    print ("All features: %s" % (data_features.shape[1]))

    # Compute the anomaly scores of the isolation forest algorithm
    # The random_state creates a reproducible result but this may not be the best solution in the future
    clf = IsolationForest(n_estimators=args.estimators, max_samples=args.samples, max_features=args.max_features, n_jobs=2, random_state=1)

    clf.fit(data_features)

    anomaly_factors = clf.decision_function(data_features)

    # Introduce 
    if args.weight:
        dummy_anomaly = np.zeros(orig_weights.shape)
        dummy_anomaly[known_non_rfi] = anomaly_factors
        dummy_anomaly[known_rfi] = np.inf
        anomaly_factors_reshape = np.reshape(dummy_anomaly, orig_shape[0:2])
    else:
        anomaly_factors_reshape = np.reshape(anomaly_factors, orig_shape[0:2])

    snrs = []
    split_values = []
    rfi_fracs = []
    # Cycle through different rfi fractions and find the best snr

    min_frac = 0
    max_frac = 60
    num_frac = 120

    for rfi_frac in np.linspace(min_frac, max_frac, num=num_frac):
        split_value = np.percentile(anomaly_factors, rfi_frac)
        test_profile = np.sum(data[anomaly_factors >= split_value, :orig_shape[2]], axis=0)
        profile_object = psrchive.Profile(orig_shape[2])
        profile_object.get_amps()[:] = test_profile
        test_snr = profile_object.snr()
        snrs.append(test_snr)
        split_values.append(split_value)
        rfi_fracs.append(rfi_frac)
        # print test_snr

    best_index = int(np.argmax(snrs) + args.additional*num_frac/max_frac)
    best_snr = snrs[best_index]
    best_frac = rfi_fracs[best_index]
    best_split_value = split_values[best_index]

    if not args.quiet:
        print ("Best SNR: %.1f RFI fraction: %.4f" % (best_snr, best_frac * 0.01))


    # Show snr evolution for different split values
    if args.contamination_plot:
        x_vals_a = np.linspace(0, 100, num= len(anomaly_factors))
        x_vals_b = np.linspace(np.min(rfi_fracs), np.max(rfi_fracs), num= len(snrs))
        plt.plot(x_vals_a, np.sort(anomaly_factors)/np.max(anomaly_factors))
        plt.plot(x_vals_b, snrs/np.max(snrs))
        plt.show()


    # Set the weights in the archive
    set_weights_archive(ar, anomaly_factors_reshape, best_split_value)

    # Create plot that shows zapped( red) and unzapped( blue) profiles if needed
    if args.print_zap:
        plt.imshow(anomaly_factors_reshape.T, vmin=best_split_value - 0.0001, vmax=best_split_value, aspect='auto',
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


def calibrate_bandpass(ar):
    # Calibrates the bandpass and grabs data. Based on:
    # https://github.com/sosl/public_codes/blob/master/pulsar/bandpass/bandpass_correction.py
    arch = ar.clone()
    arch.pscrunch()
    arch.tscrunch()
    subint = arch.get_Integration(0)
    (bl_mean, bl_var) = subint.baseline_stats()
    bl_mean = bl_mean.squeeze()
    bl_var = bl_var.squeeze()
    non_zeroes = np.where(bl_mean != 0.0)
    min_freq = arch.get_Profile(0, 0, 0).get_centre_frequency()
    max_freq = arch.get_Profile(0, 0, ar.get_nchan()-1).get_centre_frequency()
    smoothed = medfilt(bl_mean, 21)
    # plt.plot(bl_mean)
    # plt.plot(smoothed)
    # plt.show()
    # fig1 = plt.plot(freqs[non_zeroes],bl_mean[non_zeroes])
    # xlab = plt.xlabel('frequency [MHz]')
    # ylab = plt.ylabel('power [arbitrary]')
    # plt.savefig(args.ar+"_bandpass.png")
    # plt.clf()
    arch = ar.clone()
    arch.remove_baseline()
    arch.pscrunch()
    bl_mean_avg = np.average(bl_mean[non_zeroes])
    for isub in range(arch.get_nsubint()):
        for ipol in range(arch.get_npol()):
            for ichan in range(arch.get_nchan()):
                prof = arch.get_Profile(isub, ipol, ichan) 
                if ichan in non_zeroes[0]:
                    prof.scale(bl_mean_avg / smoothed[ichan])
                # else:
                #     prof.set_weight(0.0)
    return arch


def compute_metrics(data):
    #  Compute the various metrics of the profiles
    # array_feat = np.array([]).reshape(data.shape[0],0)
    feat = []
    for parts in args.partition:
        one_part_size = data.shape[1] / float(parts)
        for i in range(parts):
            data_part = data[:, int(i * one_part_size):int((i + 1) * one_part_size)]
            if 'std' in args.metrics:
                array_std = np.log(np.std(data_part, axis=1))
                feat.append(array_std)
            if 'mean' in args.metrics:
                # array_mean = np.log(np.abs(np.mean(data_part, axis=1)))
                # array_mean = np.mean(data_part, axis=1)
                array_mean = np.tanh(np.mean(data_part, axis=1))
                feat.append(array_mean)
            if 'ptp' in args.metrics:
                array_ptp = np.log(np.ptp(data_part, axis=1))
                feat.append(array_ptp)
            if 'fft' in args.metrics:
                array_fft = np.log(np.max(np.abs(np.fft.rfft(data_part - np.expand_dims(np.mean(data_part, axis=1),axis=1), axis=1)),axis=1))
                feat.append(array_fft)
            if 'med' in args.metrics:
                array_med = np.max(np.abs(data_part - medfilt(data_part, kernel_size=(1,5))),axis=1)
                feat.append(array_med)
    array_feat = np.asarray(feat).T
    # for i in range(array_feat.shape[1]):
    #     plt.imshow(np.reshape(array_feat[:,i].T, orig_shape[0:2]), aspect='auto')
    #     plt.colorbar()
    #     plt.savefig("metric_%s.png"%i)
    #     plt.close('all')
    return array_feat


if __name__=="__main__":
    args = parse_arguments()
    main(args)
