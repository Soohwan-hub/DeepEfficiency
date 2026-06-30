#!/usr/bin/python
# coding: utf-8
# Date: 2026-01-02

import os
import pandas as pd
import numpy as np
from os.path import join
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import gaussian_kde


def load_prediction(model_dir, training_dir, model_name, language_model) :
    prediction_path = join(model_dir, training_dir, model_name, language_model, "test_data_predictions.csv")
    model_predictions = pd.read_csv(prediction_path)
    experimental_values = model_predictions["y_test"]
    predicted_values = model_predictions["y_pred"]
    # # Calculate evaluation metrics
    # r2 = r2_score(experimental_values, predicted_values)
    # pearson_corr, _ = pearsonr(experimental_values, predicted_values)
    # print("Language model:", language_model)
    # print("R2:", r2)
    # print("PCC:", pearson_corr)
    return experimental_values, predicted_values, len(experimental_values)

def plot(model_dir, training_dir, model_name, language_model) :
    experimental_values, predicted_values, summary_data = load_prediction(model_dir, training_dir, model_name, language_model)

    r2 = r2_score(experimental_values, predicted_values)
    pearson_corr, _ = pearsonr(experimental_values, predicted_values)
    print("R2:", r2)
    print("PCC:", pearson_corr)
    # plt.figure(figsize=(2.0,1.8))
    plt.figure(figsize=(2.2,1.7))

    # To solve the 'Helvetica' font cannot be used in PDF file
    # https://stackoverflow.com/questions/59845568/the-pdf-backend-does-not-currently-support-the-selected-font
    # rc('text', usetex=True) 
    rc('font',**{'family':'serif','serif':['Arial']})
    plt.rcParams['pdf.fonttype'] = 42

    plt.axes([0.12,0.12,0.83,0.83])

    plt.tick_params(direction='in')
    plt.tick_params(which='major',length=1.5)
    plt.tick_params(which='major',width=0.4)

    # http://showteeth.tech/posts/24328.html
    # https://stackoverflow.com/questions/49662964/density-scatter-plot-for-huge-dataset-in-matplotlib
    values_vstack = np.vstack([experimental_values,predicted_values])
    experimental_predicted = gaussian_kde(values_vstack)(values_vstack)

    # ax = plt.scatter(x = experimental_values, y = predicted_values, c=experimental_predicted, s=3, edgecolor=[])
    ax = plt.scatter(x = experimental_values, y = predicted_values, c=experimental_predicted, cmap="viridis", s=1.5, edgecolor=[], alpha=0.8)  # "Spectral_r", "turbo", "viridis"
    # https://stackoverflow.com/questions/53935805/specify-range-of-colors-for-density-plot-in-matplotlib
    cbar = plt.colorbar(ax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Density', size=7)

    plt.text(-9.6, 9.4, 'R$^2$ = %.2f' % r2, fontweight ="normal", fontsize=7)
    plt.text(-9.6, 7.5, 'PCC = %.2f' % pearson_corr, fontweight ="normal", fontsize=7)
    # plt.text(-6.2, 4.1, 'N = %s' % summary_data, fontweight ="normal", fontsize=7)

    plt.rcParams['font.family'] = 'Arial'

    plt.xlabel(
        r'Experimental $\log_{10}(k_{\mathrm{cat}}/K_{\mathrm{m}})$ (mM$^{-1}$ s$^{-1}$)',
        fontdict={'weight': 'normal', 'fontname': 'Arial', 'size': 7}
    )

    plt.ylabel(
        r'Predicted $\log_{10}(k_{\mathrm{cat}}/K_{\mathrm{m}})$ (mM$^{-1}$ s$^{-1}$)',
        fontdict={'weight': 'normal', 'fontname': 'Arial', 'size': 7}
    )

    plt.xlim(-12, 12)
    plt.ylim(-12, 12)
    plt.xticks([-12, -8, -4, 0, 4, 8, 12])
    plt.yticks([-12, -8, -4, 0, 4, 8, 12])

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)

    plt.savefig("../figures/Fig2b_test_KCAT_KM.pdf", dpi=400, bbox_inches='tight')

def main() :
    model_dir = "../model"
    training_dir = "KCAT_KM"
    model_name = "XGBoost"
    protein_model = "ESM2_3B"
    substrate_model = "ChemBERTa-77M-MTR"
    language_model = protein_model + "&" + substrate_model
    plot(model_dir, training_dir, model_name, language_model)


if __name__ == '__main__' :
    main()


