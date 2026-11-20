import numpy as np
import matplotlib.pyplot as plt

class HealthAnalyzer:
    """"Analyserar data från hälsostudien."""

    def __init__(self, df):
        """"Sparar datan som ska anlyseras."""
        self.df = df

    def describe(self, columns):
        """"Returnerar medel, median, min och max för valda kolumner."""
        return self.df[columns].agg(["mean", "median", "min", "max"]).round(2)

    def disease_share(self):
        """"Returnerar andelen deltagare som har sjukdomen"""
        return self.df["disease"].mean()

    def confidence_interval(self, column):
        """"Returnerar ett 95 % konfidensintervall för medelvärdet med normalapproximation."""
        medel = self.df[column].mean()
        se = self.df[column].std() / np.sqrt(len(self.df))
        return medel - 1.96 * se, medel + 1.96 * se

    def plot_histogram(self, column, title, xlabel):
        """"Ritar ett histogram för en kolumn."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(self.df[column], bins=30, color="steelblue", edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Antal deltagare")
        plt.show()