import numpy as np
import openpyxl

def load_data(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)

    def read_sheet(sheet_name):
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] is None or row[1] is None:
                continue
            rows.append([float(row[0]), float(row[1])])
        arr = np.array(rows, dtype=float)
        return arr[:, 0], arr[:, 1]

    x_train, y_train = read_sheet("Training Data")
    x_test, y_test = read_sheet("Test Data")
    return x_train, y_train, x_test, y_test


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def predict(theta, x):
    return theta[0] + theta[1] * x