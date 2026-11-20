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


    def linear_regression(self, x_column, y_column):
        """"Anpassar en linjär regression y = a + b * x med minsta kvadratmetoden Returnerar intercept (a) och lutning (b)."""

        x = self.df[x_column].to_numpy()
        y = self.df[y_column].to_numpy()
        X = np.column_stack([np.ones(len(x)), x])
        koefficienter = np.linalg.lstsq(X, y, rcond=None)[0]
        return koefficienter[0], koefficienter[1]

    def plot_regression(self, x_column, y_column, xlabel, ylabel):
        """""Ritar ett spridningsdiagram med regressionslinjen"""
        a, b = self.linear_regression(x_column, y_column)
        x = self.df[x_column].sort_values()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(self.df[x_column], self.df[y_column], alpha=0.4, color="steelblue", label="Deltagare")
        ax.plot(x, a + b * x, color="red", label="Regressionslinje")
        ax.set_title(f"{ylabel} och {xlabel}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.show()


    def multiple_regression(self, x_columns, y_column):
        """Anpassar en multipel linjär regression med minsta kvadratmetoden.

        x_columns är en lista med förklarande variabler.
        Returnerar koefficienterna och förklaringsgraden R².
        """
        X = self.df[x_columns].to_numpy()
        X = np.column_stack([np.ones(len(X)), X])
        y = self.df[y_column].to_numpy()

        koefficienter = np.linalg.lstsq(X, y, rcond=None)[0]

        prediktion = X @ koefficienter
        r2 = 1 - np.sum((y - prediktion) ** 2) / np.sum((y - y.mean()) ** 2)

        namn = ["intercept"] + x_columns
        return dict(zip(namn, koefficienter)), r2