import sys

import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QTableWidgetItem
from Giao_dien_phan_mem import Ui_MainWindow
import ANN_GA_pred_strength_conc, ANN_GA_pred_Sn_2, Mix_conc_design_2
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from keras import backend
import tensorflow as tf
import pickle
import numpy as np

class MainWindows:
    def __init__(self):
        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_win)

        self.material_params = {"gaC": 2.62, "g0C": 1.42, "Mdl": 2.6,
                           "gaD": 2.68, "g0D": 1.42, "Dmax": 20, "rD": 0.47,
                           "gaNa2SiO3": 1.45, "SiO2_ttl": 0.267, "Na2O_ttl": 0.0984, "H2O_ttl": 0.6346,
                           "gaNaOH": 2.13, "do_tinh_khiet": 0.99,
                           "gaXLC": 2.85, "do_min": "S1",
                           "gaTB": 2.24, "Loai_TB": "F",
                           "C_price": 181.5, "D_price": 242.0, "XLC_price": 500, "TB_price": 100,
                           "Na2SiO3_price": 4000, "NaOH_price": 13200, "N_price": 5, "Ms": 1.2}
        self.n1 = 32
        self.n2 = 16
        self.n3 = 32
        self.n4 = 16
        self.mix_final = []

        # Modul-1
        self.uic.pushButton_3.clicked.connect(self.Vat_lieu)
        # Modul_2
        self.uic.pushButton_6.clicked.connect(self.GA_ANN_du_doan_R)
        self.uic.pushButton_12.clicked.connect(self.Display_1)
        self.uic.pushButton_13.clicked.connect(self.Du_doan_R)
        # Modul_3
        self.uic.pushButton_7.clicked.connect(self.GA_ANN_du_doan_Sn)
        self.uic.pushButton_15.clicked.connect(self.Display_2)
        self.uic.pushButton_14.clicked.connect(self.Du_doan_Sn)
        # Modul_4
        self.uic.pushButton.clicked.connect(self.Tinh_cap_phoi)
        # Modul_5
        self.uic.pushButton_2.clicked.connect(self.Toi_uu_cap_phoi)

    def Vat_lieu(self):
        self.material_params["gaC"] = float(self.uic.doubleSpinBox_5.text()[:4].replace(",","."))
        self.material_params["g0C"] = float(self.uic.doubleSpinBox_6.text()[:4].replace(",", "."))
        self.material_params["Mdl"] = float(self.uic.doubleSpinBox_7.text()[:3].replace(",", "."))

        self.material_params["gaD"] = float(self.uic.doubleSpinBox_39.text()[:4].replace(",", "."))
        self.material_params["g0D"] = float(self.uic.doubleSpinBox_40.text()[:4].replace(",", "."))
        self.material_params["Dmax"] = float(self.uic.doubleSpinBox_41.text()[:2])
        self.material_params["rD"] = (self.material_params["gaD"]-self.material_params["g0D"])/self.material_params["gaD"]

        self.material_params["gaNa2SiO3"] = float(self.uic.doubleSpinBox_35.text()[:4].replace(",", "."))
        self.material_params["SiO2_ttl"] = float(self.uic.doubleSpinBox_36.text()[:5].replace(",", "."))/100
        self.material_params["Na2O_ttl"] = float(self.uic.doubleSpinBox_37.text()[:4].replace(",", "."))/100
        self.material_params["H2O_ttl"] = 1 - self.material_params["SiO2_ttl"] - self.material_params["Na2O_ttl"]

        self.material_params["gaNaOH"] = float(self.uic.doubleSpinBox_32.text()[:4].replace(",", "."))
        self.material_params["do_tinh_khiet"] = float(self.uic.doubleSpinBox_33.text()[:4].replace(",", "."))

        self.material_params["gaXLC"] = float(self.uic.doubleSpinBox_45.text()[:4].replace(",", "."))
        self.material_params["do_min"] = self.uic.comboBox_4.currentText()
        self.material_params["gaTB"] = float(self.uic.doubleSpinBox_43.text()[:4].replace(",", "."))
        self.material_params["Loai_TB"] = self.uic.comboBox_3.currentText()

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Đã lưu thành công!")
        msg.setWindowTitle("Vật liệu đầu vào")
        msg.exec_()
        return self.material_params

    def GA_ANN_du_doan_R(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Chương trình chạy mất vài phút. Xin vui lòng đợi!")
        msg.setWindowTitle("GA-ANN dự đoán R")
        msg.exec_()

        self.n1 = int(self.uic.spinBox_14.text())
        self.n2 = int(self.uic.spinBox_15.text())

        ANN_GA_pred_strength_conc.main(self.n1, self.n2)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Đã chạy thành công!")
        msg.setWindowTitle("GA-ANN dự đoán R")
        msg.exec_()

    def Display_1(self):
        if self.uic.radioButton_13.isChecked():
            path = "C:/Users/Admin/PycharmProjects/pythonProject/BTL_CN_phan_mem/Save_png/4.png"
        if self.uic.radioButton_14.isChecked():
            path = "C:/Users/Admin/PycharmProjects/pythonProject/BTL_CN_phan_mem/Save_png/3.png"
        if self.uic.radioButton_15.isChecked():
            path = "C:/Users/Admin/PycharmProjects/pythonProject/BTL_CN_phan_mem/Save_png/5.png"
        self.uic.label_2.setPixmap(QPixmap(path))

    def rmse(self, y_true, y_pred):
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
    def mae(self, y_true, y_pred):
        return backend.mean(backend.abs(y_pred - y_true), axis=-1)
    def ANN(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(self.n1, input_dim=6, use_bias=True, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(self.n2, use_bias=True, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(1))
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.005, rho=0.9)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=[self.mae, self.rmse])
        return model
    def Du_doan_R(self):
        C_weight = float(self.uic.doubleSpinBox_68.text()[:6].replace(",", "."))
        D_weight = float(self.uic.doubleSpinBox_69.text()[:6].replace(",", "."))
        XLC_weight = float(self.uic.doubleSpinBox_70.text()[:5].replace(",", "."))
        TB_weight = float(self.uic.doubleSpinBox_113.text()[:5].replace(",", "."))
        Na2SiO3_weight = float(self.uic.doubleSpinBox_114.text()[:5].replace(",", "."))
        NaOH_weight = float(self.uic.doubleSpinBox_115.text()[:4].replace(",", "."))
        N_weight = float(self.uic.doubleSpinBox_116.text()[:5].replace(",", "."))
        Tuoi = float(self.uic.doubleSpinBox_117.text()[:2].replace(",", "."))

        Mtx = TB_weight + XLC_weight
        XLC = XLC_weight/Mtx *100
        Na2O = (Na2SiO3_weight*self.material_params["Na2O_ttl"] + NaOH_weight*62/81)/Mtx *100
        N_TX = N_weight/Mtx
        Kd = (TB_weight/self.material_params["gaTB"]+XLC_weight/self.material_params["gaXLC"]+N_weight+C_weight/self.material_params["gaC"])/\
             (self.material_params["rD"]*D_weight/self.material_params["g0D"])

        R = self.pred_R(Mtx,XLC,Na2O,N_TX,Kd,Tuoi)
        R_pred = str(R)[:5] + " MPa"
        self.uic.textEdit_9.setText(R_pred)

    def GA_ANN_du_doan_Sn(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Chương trình chạy mất vài phút. Xin vui lòng đợi!")
        msg.setWindowTitle("GA-ANN dự đoán Sn")
        msg.exec_()

        self.n3 = int(self.uic.spinBox_18.text())
        self.n4 = int(self.uic.spinBox_19.text())

        ANN_GA_pred_Sn_2.main(self.n3, self.n4)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Đã chạy thành công!")
        msg.setWindowTitle("GA-ANN dự đoán SnR")
        msg.exec_()

    def Display_2(self):
        if self.uic.radioButton_16.isChecked():
            path = "C:/Users/Admin/PycharmProjects/pythonProject/BTL_CN_phan_mem/Save_png/9.png"
        if self.uic.radioButton_18.isChecked():
            path = "C:/Users/Admin/PycharmProjects/pythonProject/BTL_CN_phan_mem/Save_png/10.png"
        if self.uic.radioButton_17.isChecked():
            path = "C:/Users/Admin/PycharmProjects/pythonProject/BTL_CN_phan_mem/Save_png/8.png"
        self.uic.label_3.setPixmap(QPixmap(path))

    def ANN_Sn(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(self.n3, input_dim=5, use_bias=True, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(self.n4, use_bias=True, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(1))
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.005, rho=0.9)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=[self.mae, self.rmse])
        return model

    def Du_doan_Sn(self):
        C_weight = float(self.uic.doubleSpinBox_71.text()[:6].replace(",", "."))
        D_weight = float(self.uic.doubleSpinBox_72.text()[:6].replace(",", "."))
        XLC_weight = float(self.uic.doubleSpinBox_73.text()[:5].replace(",", "."))
        TB_weight = float(self.uic.doubleSpinBox_118.text()[:5].replace(",", "."))
        Na2SiO3_weight = float(self.uic.doubleSpinBox_119.text()[:5].replace(",", "."))
        NaOH_weight = float(self.uic.doubleSpinBox_120.text()[:4].replace(",", "."))
        N_weight = float(self.uic.doubleSpinBox_121.text()[:5].replace(",", "."))

        Mtx = TB_weight + XLC_weight
        XLC = XLC_weight/Mtx *100
        Na2O = (Na2SiO3_weight*self.material_params["Na2O_ttl"] + NaOH_weight*62/81)/Mtx *100
        N_TX = N_weight/Mtx
        Kd = (TB_weight/self.material_params["gaTB"]+XLC_weight/self.material_params["gaXLC"]+N_weight+C_weight/self.material_params["gaC"])/\
             (self.material_params["rD"]*D_weight/self.material_params["g0D"])

        Sn = self.pred_Sn(Mtx,XLC,Na2O,N_TX,Kd)
        Sn_pred = str(Sn)[:4] + " cm"
        self.uic.textEdit_10.setText(Sn_pred)

    def pred_Sn(self, Mtx,XLC,Na2O,N_TX,Kd):
        model_ANN_Sn = self.ANN_Sn()
        model_ANN_Sn.load_weights('Save_Model/checkpoint_ANN_GA_Sn.hdf5')
        Scaler_Sn = pickle.load(open('Save_Model/scaler_ANN_GA_Sn.pkl', 'rb'))
        Xi = Scaler_Sn.transform(np.array((Mtx,XLC,Na2O,N_TX, Kd), dtype=float).reshape(-1,5))
        Sn = model_ANN_Sn(Xi).numpy()[0,0]
        return Sn

    def pred_R(self, Mtx,XLC,Na2O,N_TX,Kd,tuoi=28):
        model_ANN_GA = self.ANN()
        model_ANN_GA.load_weights('Save_Model/checkpoint_ANN_GA.hdf5')
        Scaler_ANN_GA = pickle.load(open('Save_Model/scaler_ANN_GA.pkl', 'rb'))
        Xi = Scaler_ANN_GA.transform(np.array((Mtx,XLC,Na2O,N_TX, Kd, tuoi), dtype=float).reshape(-1,6))
        R = model_ANN_GA(Xi).numpy()[0,0]
        return R

    def get_initial_Na2O_N_TX_XLC(self, R28):
        result=[]
        XLC_params = np.arange(40,61,10)              #7 gia tri
        Na2O_params = np.arange(4.0, 6.1, 0.5)    #21 gia tri => result gom 7*21=147 gia tri
        for XLC in XLC_params:
            for Na2O in Na2O_params:
                # From R28 = f(XLC, Na2O, N_TX) => a*N_TX^2 + b*N_TX + c = 0
                a = -159.665
                b = 31.2176 +18.7442*Na2O -1.81025*XLC
                c = - R28 -55.3333 +20.6822*Na2O +1.93365*XLC +0.0968158*Na2O*XLC -2.42788*(Na2O)**2 -0.0116982*(XLC)**2
                delta = b**2 - 4*a*c
                if delta<0:
                    break
                else:
                    N_TX = float((-b - np.sqrt(delta))/(2 * a))
                    result.append([XLC, Na2O, N_TX])
        return result
    def Mix_result(self, Mtx,XLC,Na2O,N_TX,Kd):
        Ms = self.material_params["Ms"]
        Na2O_ttl = self.material_params["Na2O_ttl"]
        SiO2_ttl = self.material_params["SiO2_ttl"]
        H2O_ttl = self.material_params["H2O_ttl"]
        gaNaOH = self.material_params["gaNaOH"]
        gaNa2SiO3 = self.material_params["gaNa2SiO3"]
        gaC = self.material_params["gaC"]
        g0C = self.material_params["g0C"]
        gaD = self.material_params["gaD"]
        g0D = self.material_params["g0D"]
        rD = self.material_params["rD"]
        gaTB = self.material_params["gaTB"]
        gaXLC = self.material_params["gaXLC"]




        XLC = XLC/100
        Na2O = Na2O/100
        CKD_weight = Mtx/(1-Na2O*(1+Ms))
        XLC_weight = XLC*Mtx
        TB_weight = Mtx - XLC_weight
        NaOH_weight = 1.29*Na2O*CKD_weight*(1-Ms*Na2O_ttl/SiO2_ttl)
        Na2SiO3_weight = Ms*CKD_weight*Na2O/SiO2_ttl
        N = Mtx * N_TX
        N_weight = N - Na2SiO3_weight * H2O_ttl
        D_weight = 1000/(Kd*rD/g0D + 1/gaD)
        C_weight = (1000 - (TB_weight/gaTB + XLC_weight/gaXLC + NaOH_weight/gaNaOH + Na2SiO3_weight/gaNa2SiO3
                        + N_weight/1.0 + D_weight/gaD))*gaC

        mix = {"Cát":C_weight, "Đá dăm": D_weight, "Tro bay": TB_weight,
           "Xỉ lò cao":XLC_weight, "NaOH":NaOH_weight, "dd Na2SiO3":Na2SiO3_weight,
           "Nước thêm":N_weight}
        return mix
    def Tinh_cap_phoi(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Chương trình chạy mất vài phút. Xin vui lòng đợi!")
        msg.setWindowTitle("Tính toán cấp phối")
        msg.exec_()

        R28 = float(self.uic.doubleSpinBox_77.text()[:3].replace(",", "."))
        Sn = float(self.uic.doubleSpinBox_78.text()[:3].replace(",", "."))
        initial_params = self.get_initial_Na2O_N_TX_XLC(R28)
        for x in initial_params:
            print(x)
        print('=============================')
        result = []
        for x in initial_params:
            XLC = x[0]
            Na2O = x[1]
            N_TX = x[2]
            N_params = np.arange(150, 181, 5)  # 7 gia tri
            Kd_params = np.arange(1.50, 1.76, 0.25)  # 6 gia tri => result gom 147*7*6 = 6174 gia tri
            for N in N_params:
                for Kd in Kd_params:
                    Mtx = N / N_TX
                    if 250 <= Mtx <= 450:
                        Sn_pred = self.pred_Sn(Mtx, XLC, Na2O, N_TX, Kd)
                        R28_pred = self.pred_R(Mtx, XLC, Na2O, N_TX, Kd)
                        result.append([Sn_pred, R28_pred, Mtx, XLC, Na2O, N_TX, Kd])
        self.mix_final = []
        for x in result:
            Sn_pred = x[0]
            R28_pred = x[1]
            err_Sn = (Sn_pred - Sn) / Sn * 100
            err_R28 = (R28_pred - R28) / R28 * 100
            if (-20< err_Sn < 20) and (0 < err_R28 < 5):
                print(x)
                Mtx, XLC, Na2O, N_TX, Kd = x[2], x[3], x[4], x[5], x[6]
                self.mix_final.append(self.Mix_result(Mtx, XLC, Na2O, N_TX, Kd))

        df = pd.DataFrame(self.mix_final)
        df.fillna('', inplace=True)
        self.uic.tableWidget.setRowCount(df.shape[0])
        self.uic.tableWidget.setColumnCount(df.shape[1])
        self.uic.tableWidget.setHorizontalHeaderLabels(df.columns)

        for row in df.iterrows():
            values = row[1]
            for col_index, value in enumerate(values):
                if isinstance(value, (float, int)):
                    value = '{0:0,.0f}'.format(value)
                tableItem = QTableWidgetItem(str(value))
                self.uic.tableWidget.setItem(row[0], col_index, tableItem)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Đã chạy xong!")
        msg.setWindowTitle("Tính toán cấp phối")
        msg.exec_()
        return self.mix_final

    def Toi_uu_cap_phoi(self):
        C_price = float(self.uic.doubleSpinBox_8.text()[:6])/(self.material_params["g0C"]*1000)
        D_price = float(self.uic.doubleSpinBox_9.text()[:6])/(self.material_params["g0D"]*1000)
        XLC_price = float(self.uic.doubleSpinBox_11.text()[:3])
        TB_price = float(self.uic.doubleSpinBox_10.text()[:3])
        Na2SiO3_price = float(self.uic.doubleSpinBox_12.text()[:4])
        NaOH_price = float(self.uic.doubleSpinBox_13.text()[:5])
        N_price = 5

        result = []

        for m in self.mix_final:
            C_weight = float(m["Cát"])
            D_weight = float(m["Đá dăm"])
            TB_weight = float(m["Tro bay"])
            XLC_weight = float(m["Xỉ lò cao"])
            NaOH_weight = float(m["NaOH"])
            Na2SiO3_weight = float(m["dd Na2SiO3"])
            N_weight = float(m["Nước thêm"])

            cost = (C_weight * C_price + D_weight * D_price + TB_weight * TB_price +
                    XLC_weight * XLC_price + NaOH_weight * NaOH_price + Na2SiO3_weight * Na2SiO3_price +
                    N_weight * N_price)
            result.append({"Chi phí":cost,"Cát":C_weight, "Đá dăm": D_weight, "Tro bay": TB_weight,
                           "Xỉ lò cao":XLC_weight, "NaOH":NaOH_weight, "dd Na2SiO3":Na2SiO3_weight,
                           "Nước thêm":N_weight})


        df = pd.DataFrame(result)
        df.fillna('', inplace=True)
        self.uic.tableWidget_2.setRowCount(df.shape[0])
        self.uic.tableWidget_2.setColumnCount(df.shape[1])
        self.uic.tableWidget_2.setHorizontalHeaderLabels(df.columns)

        for row in df.iterrows():
            values = row[1]
            for col_index, value in enumerate(values):
                if isinstance(value, (float, int)):
                    value = '{0:0,.0f}'.format(value)
                tableItem = QTableWidgetItem(str(value))
                self.uic.tableWidget_2.setItem(row[0], col_index, tableItem)

    def show(self):
        self.main_win.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Windows")
    main_win = MainWindows()
    main_win.show()
    sys.exit(app.exec_())