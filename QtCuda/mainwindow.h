#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector_types.h>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QLabel>
#include <QtWidgets/QCheckBox>
#include <QApplication>
#include <QDesktopWidget>
#include <QColorDialog>
#include <QMessageBox>
#include <QTextStream>
#include "ui_mainwindow.h"
#include "def.h"

#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QAreaSeries>
#include <QtCharts/QScatterSeries>
QT_CHARTS_USE_NAMESPACE

typedef float(*Pointer)[4];
extern "C" float4* get_tf_array();
extern "C" float* get_relative_visibility_histogram();

extern "C" void apply_blending_operation();
extern "C" void reset_transfer_function();

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QWidget *parent = Q_NULLPTR);

	void update_color(QColor c)
	{
		if (color_array)
		{
			float *p = *color_array;
			p[0] = (float)c.redF();
			p[1] = (float)c.greenF();
			p[2] = (float)c.blueF();
			p[3] = (float)c.alphaF();
			std::cout << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << std::endl;
		}
		color = c;
		auto c2 = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue());
		QPalette sample_palette;
		sample_palette.setColor(QPalette::Button, color);
		sample_palette.setColor(QPalette::Base, color);
		//sample_palette.setColor(QPalette::ButtonText, c2);
		sample_palette.setColor(QPalette::Text, c2);
		ui.pushButton->setPalette(sample_palette);
		ui.lineEdit->setPalette(sample_palette);
		ui.lineEdit->setText(color.name());
	}

	void set_pointers(Pointer picked_color, bool *alpha, bool *color)
	{
		color_array = picked_color;
		apply_alpha = alpha;
		apply_color = color;
	}

private slots:
    void on_pushButton_clicked();

    void on_checkBox_clicked();

    void on_checkBox_2_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_pushButton_4_clicked();

private:
	Ui::MainWindowClass ui;
	QColor color;
	Pointer color_array = NULL;
	bool *apply_alpha = NULL;
	bool *apply_color = NULL;
	//QGraphicsScene scene;
	QChartView chartView;
	QChartView chartView2;
};
