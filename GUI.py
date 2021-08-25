'''
负责生成GUI界面，以及数据读取，模型调用等操作
'''
import os
import subprocess
import sys
import h5py
import open3d as o3d
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QComboBox, QGraphicsWidget, QLabel, QFileDialog
from PyQt5.QtGui import QIcon

import predict

#open3d测试
# points = np.random.rand(10000, 3)
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(points)
# pic =  o3d.visualization.draw_geometries([point_cloud])

INPUT = np.empty((10000,3))

'''
main frame
'''
class frame(QWidget):
    def __init__(self):
        super(frame, self).__init__()
        self.initUI()

    def initUI(self):
        #最外层设置
        self.setGeometry(300,200,1280,720)
        self.setWindowTitle('GUI_PARTNET')


        #两个显示框
        self.grah_input = QWidget(self  )
        self.grah_input.setGeometry(40,100,400,400)

        self.grah_output = QWidget(self)
        self.grah_output.setGeometry(850, 100, 400, 400)


        #按钮
        self.btn_input = QPushButton('选择输入点云',self)
        self.btn_input.setGeometry(150, 530, 130, 50)
        self.btn_input.clicked.connect(self.show_in_pts)

        self.btn_predict = QPushButton('predict',self)
        self.btn_predict.setGeometry(1000,530,130,50)
        self.btn_predict.clicked.connect(self.show_out_pts)


        #下拉栏
        self.comb_algorithm = QComboBox(self)
        self.comb_algorithm.setGeometry(590,200,150,36)
        self.comb_algorithm.addItems(['ins_seg','sem_seg'])

        self.comb_catgory = QComboBox(self)
        self.comb_catgory.setGeometry(590, 300, 150, 36)
        self.comb_catgory.addItems(['Bag','Bed','Bottle','Bowl','Chair','Clock','Dishwasher','Display','Door','Earphone','Faucet','Hat','Keyboard','Knife','Lamp','Laptop','Microwave','Mug','Refrigerator','Scissors','StorageFurniture','Table','TrashCan','Vase'])

        self.comb_level = QComboBox(self)
        self.comb_level.setGeometry(590, 400, 150, 36)
        self.comb_level.addItems(['1', '2','3'])


        #标签
        self.label1 = QLabel(self)
        self.label1.setText('算法：')
        self.label1.setGeometry(530, 200, 60, 36)

        self.label2 = QLabel(self)
        self.label2.setText('类别：')
        self.label2.setGeometry(530, 300, 60, 36)

        self.label3 = QLabel(self)
        self.label3.setText('层级：')
        self.label3.setGeometry(530, 400, 60, 36)


        #输入点云展示部分
        self.w = gl.GLViewWidget(self.grah_input)
        self.w.setGeometry(0,0,400,400)
        self.w.opts['distance'] = 3 # 初始视角高度
        self.w.show()
        # w.setWindowTitle('pyqtgraph example: GLLinePlotItem')

        #输出点云展示部分
        self.out = gl.GLViewWidget(self.grah_output)
        self.out.setGeometry(0,0,400,400)
        self.out.opts['distance'] = 3 # 初始视角高度
        self.out.show()


        #这部分是在画网格
        # gx = gl.GLGridItem()
        # gx.rotate(90, 0, 1, 0)
        # gx.translate(-10, 0, 0)
        # w.addItem(gx)
        # gy = gl.GLGridItem()
        # gy.rotate(90, 1, 0, 0)
        # gy.translate(0, -10, 0)
        # w.addItem(gy)
        # gz = gl.GLGridItem()
        # gz.translate(0, 0, -10)
        # w.addItem(gz)


        #渲染点
        # n = 51
        # y = np.linspace(-10, 10, n)
        # x = np.linspace(-10, 10, 100)
        # for i in range(n):
        #     yi = np.array([y[i]] * 100)
        #     d = (x ** 2 + yi ** 2) ** 0.5
        #     z = 10 * np.cos(d) / (d + 1)
        #     pts = np.vstack([x, yi, z]).transpose()
        #     plt = gl.GLLinePlotItem(pos=pts, color=pg.glColor((i, n * 1.3)), width=(i + 1) / 10., antialias=True)
        #     w.addItem(plt)

        self.show()

    def show_in_pts(self):
        self.w.clear()
        self.w.show()
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', 'point_cloud files(*.h5 , *.txt)')
        print(openfile_name)
        openfile_name = openfile_name[0]
        with h5py.File(openfile_name,'r') as f:
            print(f.keys())
            ptc = f['pts'][0]
            global INPUT
            INPUT = ptc #作为算法输入
            n = ptc.shape[0]
            pos = np.empty((n, 3))  # 存放点的位置，为53 * 3的向量，感觉说是矩阵更合适
            size = np.empty((n))  # 存放点的大小
            color = np.empty((n, 4))  # 存放点的颜色
            for i,point in enumerate(ptc):
                x = point[0]
                y = point[1]
                z = point[2]
                pos[i] = (x, y, z)
                size[i] = 0.01
                color[i] = (1.0, 0.0, 0.0, 1)
            sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
            self.w.addItem(sp1)
            self.w.show()



    def show_out_pts(self):
        #调用预测前
        self.out.clear()
        self.out.show()
        algorithm = self.comb_algorithm.currentText()
        catgory =  self.comb_catgory.currentText()
        level = self.comb_level.currentText()
        seg_net,ins_net,max_indexs = predict.pre(INPUT,algorithm,catgory,level)
        # 渲染变量初始化
        n = INPUT.shape[0]
        pos = np.empty((n, 3))  # 存放点的位置，为53 * 3的向量，感觉说是矩阵更合适
        size = np.empty((n))  # 存放点的大小
        color = np.empty((n, 4))  # 存放点的颜色
        # 为per part着色
        colors_all = np.empty((len(max_indexs), 4))
        print(len(max_indexs))
        for i in range(len(max_indexs)):
            colors_all[i][0] = np.random.rand()
            colors_all[i][1] = np.random.rand()
            colors_all[i][2] = np.random.rand()
            colors_all[i][3] = 1
            print(colors_all[i])
        #实例分割可视化
        if algorithm == 'ins_seg':
            for i, p in enumerate(INPUT):
                x = p[0]
                y = p[1]
                z = p[2]
                pos[i] = (x, y, z)
                size[i] = 0.01
                if ins_net[i] in max_indexs:
                    index = max_indexs.index(ins_net[i])
                    color[i] = colors_all[index]
                else:
                    color[i] = (1.0, 1.0, 1.0, 1)
            sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
            self.out.addItem(sp1)
        #语义分割可视化
        elif algorithm == 'sem_seg':
            for i, p in enumerate(INPUT):
                x = p[0]
                y = p[1]
                z = p[2]
                pos[i] = (x, y, z)
                size[i] = 0.01
                color[i] = colors_all[seg_net[i]]
            sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
            self.out.addItem(sp1)
        self.out.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myframe = frame()
    sys.exit(app.exec_())