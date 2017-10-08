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
#include <QFileDialog>
#include <QDebug>
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
extern "C" void apply_tf_editing();
extern "C" void reset_transfer_function();
extern "C" void add_volume_to_list_for_update_from_vector(std::vector<std::string> filelist);
extern "C" void load_mhd_header(std::string filename);

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

	void set_pointers(Pointer picked_color, bool *alpha, bool *color, bool *time_varying_tf, bool *time_varying_tf_reset)
	{
		color_array = picked_color;
		apply_alpha = alpha;
		apply_color = color;
		apply_time_varying_tf_editing = time_varying_tf;
		apply_time_varying_tf_reset = time_varying_tf_reset;
	}

private slots:
    void on_pushButton_clicked();

    void on_checkBox_clicked();

    void on_checkBox_2_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_pushButton_4_clicked();

    void on_checkBox_3_clicked();

    void on_checkBox_4_clicked();

    void on_action_About_triggered();

    void on_action_Exit_triggered();

    void on_action_Open_triggered();

    void on_actionOpen_Files_triggered();

private:
	Ui::MainWindowClass ui;
	QColor color;
	Pointer color_array = NULL;
	bool *apply_alpha = NULL;
	bool *apply_color = NULL;
	bool *apply_time_varying_tf_editing = NULL;
	bool *apply_time_varying_tf_reset = NULL;
	//QGraphicsScene scene;
	QChartView chartView;
	QChartView chartView2;
};
