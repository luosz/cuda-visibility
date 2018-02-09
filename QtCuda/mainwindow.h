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
#include <QTimer>
#include <QPixmap>
#include <QLabel>
#include <helper_math.h>
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
extern "C" float* get_global_visibility_histogram();
extern "C" float* get_local_visibility_histogram();
extern "C" float* get_tf_component0();
extern "C" float* get_tf_component1();
extern "C" float* get_tf_component2();
extern "C" float* get_tf_component3();
extern "C" void apply_tf_editing();
extern "C" void apply_temporal_tf_editing();
extern "C" void reset_transfer_function();
extern "C" void add_volume_to_list_for_update_from_vector(std::vector<std::string> filelist);
extern "C" void load_mhd_file(std::string filename);
extern "C" void save_view(const char *file);
extern "C" void load_view(const char *file);
extern "C" void openTransferFunctionFromVoreenXML(const char *filename);
extern "C" void bind_tf_texture();
extern "C" void save_tf_array_to_voreen_XML(const char *filename);
extern "C" int get_screenshot_id();
extern "C" int get_next_screenshot_id(int id);

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

	void set_pointers(Pointer picked_color, bool *alpha, bool *color, bool *time_varying_tf_editing, bool *time_varying_tf_reset, bool *time_varying_vws_optimization, bool *temporal_visibility)
	{
		color_array = picked_color;
		apply_alpha = alpha;
		apply_color = color;
		apply_time_varying_tf_editing = time_varying_tf_editing;
		apply_time_varying_tf_reset = time_varying_tf_reset;
		apply_time_varying_vws_optimization = time_varying_vws_optimization;
		calc_temporal_visibility = temporal_visibility;
	}

	void delay_show_transfer_function(int msec=10)
	{
		QTimer::singleShot(msec, this, SLOT(show_transfer_function()));
	}

	void update_checkbox()
	{
		if (calc_temporal_visibility)
		{
			ui.checkBox_6->setChecked(*calc_temporal_visibility);
		}
		if (apply_time_varying_vws_optimization)
		{
			ui.checkBox_5->setChecked(*apply_time_varying_vws_optimization);
		}
		if (apply_time_varying_tf_reset)
		{
			ui.checkBox_4->setChecked(*apply_time_varying_tf_reset);
		}
		if (apply_time_varying_tf_editing)
		{
			ui.checkBox_3->setChecked(*apply_time_varying_tf_editing);
		}
	}

	void update_screenshots_later(int msec = 10)
	{
		QTimer::singleShot(msec, this, SLOT(update_screenshots()));
	}

	bool is_save_renderings_checked()
	{
		return ui.checkBox_7->isChecked();
	}

	void calculate_visibility_without_editing_tf()
	{
		ui.checkBox->setChecked(false);
		ui.checkBox_2->setChecked(false);
		apply_tf_editing();
		delay_show_transfer_function();
	}

	void draw_transfer_function_component(float tf_component[], QChartView &chartView)
	{
		const qreal N = D_BIN_COUNT - 1;
		auto p_tf = get_tf_array();
		auto chart_tf = chartView.chart();
		chart_tf->removeAllSeries();
		chart_tf->legend()->hide();
		auto line_width = get_line_width(chart_tf->size().width());
		for (int i = 0; i < D_BIN_COUNT; i++)
		{
			auto c = QColor::fromRgbF((qreal)p_tf[i].x, (qreal)p_tf[i].y, (qreal)p_tf[i].z);
			auto line = new QLineSeries();
			line->append(i / N, 0);
			line->append(i / N, (qreal)tf_component[i]);
			//line->setColor(c);
			QPen pen(c);
			pen.setWidth(line_width);
			line->setPen(pen);
			chart_tf->addSeries(line);
		}
		chart_tf->createDefaultAxes();
		//chart_tf->setTitle(title);
		chartView.setRenderHint(QPainter::Antialiasing);
	}

	void delay_add_transfer_function_component(float tf_component[], QChartView &chartView, int msec = 10)
	{
		// Use a lambda expression with a capture list for the Qt slot with arguments
		QTimer::singleShot(msec, this, [this, tf_component, &chartView]() {add_transfer_function_component(tf_component, chartView); });
	}

	inline qreal get_line_width(qreal chart_width)
	{
		return chart_width / D_BIN_COUNT + 1. / 6.;
	}

	void draw_histogram(float histogram[], QChartView &chartView)
	{
		const qreal N = D_BIN_COUNT - 1;
		//auto p = get_global_visibility_histogram();
		auto chart = chartView.chart();
		chart->removeAllSeries();
		chart->legend()->hide();
		auto line_width = get_line_width(chart->size().width());
		for (int i = 0; i < D_BIN_COUNT; i++)
		{
			auto c = QColor::fromRgbF(0.5, 0.5, 0.5);
			auto line = new QLineSeries();
			line->append(i / N, 0);
			line->append(i / N, (qreal)histogram[i]);
			//line->setColor(c);
			QPen pen(c);
			pen.setWidth(line_width);
			line->setPen(pen);
			chart->addSeries(line);
		}
		chart->createDefaultAxes();
		//chart->setTitle(title);
		chartView.setRenderHint(QPainter::Antialiasing);
	}

	void clear_transfer_function_components()
	{
		auto tf0 = get_tf_component0();
		auto tf1 = get_tf_component1();
		auto tf2 = get_tf_component2();
		memset(tf0, 0, sizeof(float)*D_BIN_COUNT);
		memset(tf1, 0, sizeof(float)*D_BIN_COUNT);
		memset(tf2, 0, sizeof(float)*D_BIN_COUNT);
		draw_transfer_function_component(tf0, chartView_features[0]);
		draw_transfer_function_component(tf1, chartView_features[1]);
		draw_transfer_function_component(tf2, chartView_features[2]);
	}

private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_pushButton_4_clicked();

    void on_checkBox_3_clicked();

    void on_checkBox_4_clicked();

    void on_action_About_triggered();

    void on_action_Exit_triggered();

    void on_action_Open_triggered();

    void on_actionOpen_Files_triggered();

    void on_actionOpen_transfer_function_triggered();

    void on_actionSave_transfer_function_as_triggered();

    void on_actionLoad_view_and_region_triggered();

    void on_actionSave_view_and_region_as_triggered();

    void on_checkBox_5_clicked();

	void show_transfer_function()
	{
		auto p_tf = get_tf_array();
		const qreal N = D_BIN_COUNT - 1;

		auto chart_tf = chartView_tf.chart();
		chart_tf->removeAllSeries();
		chart_tf->legend()->hide();
		auto line_width = get_line_width(chart_tf->size().width());
		for (int i = 0; i < D_BIN_COUNT; i++)
		{
			auto c = QColor::fromRgbF((qreal)p_tf[i].x, (qreal)p_tf[i].y, (qreal)p_tf[i].z);
			auto line = new QLineSeries();
			line->append(i / N, (qreal)p_tf[i].w);
			line->append(i / N, 0);
			//line->setColor(c);
			QPen pen(c);
			pen.setWidth(line_width);
			line->setPen(pen);
			chart_tf->addSeries(line);
		}
		chart_tf->createDefaultAxes();
		//chart_tf->setTitle("Transfer function");
		chartView_tf.setRenderHint(QPainter::Antialiasing);

		draw_histogram(get_relative_visibility_histogram(), chartView_relative);
		draw_histogram(get_global_visibility_histogram(), chartView_global);
		draw_histogram(get_local_visibility_histogram(), chartView_local);

		//auto p4 = get_relative_visibility_histogram();
		//auto chart4 = chartView_relative.chart();
		//chart4->removeAllSeries();
		//chart4->legend()->hide();
		//for (int i = 0; i < D_BIN_COUNT; i++)
		//{
		//	auto c = QColor::fromRgbF(0.5, 0.5, 0.5);
		//	auto line = new QLineSeries();
		//	line->append(i / N, 0);
		//	line->append(i / N, (qreal)p4[i]);
		//	//line->setColor(c);
		//	QPen pen(c);
		//	pen.setWidth(line_width);
		//	line->setPen(pen);
		//	chart4->addSeries(line);
		//}
		//chart4->createDefaultAxes();
		//chart4->setTitle("Relative visibility histogram");
		//chartView_relative.setRenderHint(QPainter::Antialiasing);

		//auto p = get_global_visibility_histogram();
		//auto chart = chartView_global.chart();
		//chart->removeAllSeries();
		//chart->legend()->hide();
		//for (int i = 0; i < D_BIN_COUNT; i++)
		//{
		//	auto c = QColor::fromRgbF(0.5, 0.5, 0.5);
		//	auto line = new QLineSeries();
		//	line->append(i / N, 0);
		//	line->append(i / N, (qreal)p[i]);
		//	//line->setColor(c);
		//	QPen pen(c);
		//	pen.setWidth(line_width);
		//	line->setPen(pen);
		//	chart->addSeries(line);
		//}
		//chart->createDefaultAxes();
		//chart->setTitle("Global visibility histogram");
		//chartView_global.setRenderHint(QPainter::Antialiasing);

		//auto p2 = get_local_visibility_histogram();
		//auto chart2 = chartView_local.chart();
		//chart2->removeAllSeries();
		//chart2->legend()->hide();
		//for (int i = 0; i < D_BIN_COUNT; i++)
		//{
		//	auto c = QColor::fromRgbF(0.5, 0.5, 0.5);
		//	auto line = new QLineSeries();
		//	line->append(i / N, 0);
		//	line->append(i / N, (qreal)p2[i]);
		//	//line->setColor(c);
		//	QPen pen(c);
		//	pen.setWidth(line_width);
		//	line->setPen(pen);
		//	chart2->addSeries(line);
		//}
		//chart2->createDefaultAxes();
		//chart2->setTitle("Local visibility histogram");
		//chartView_local.setRenderHint(QPainter::Antialiasing);
	}

	void add_transfer_function_component(float tf_component[], QChartView &chartView)
	{
		//const qreal N = D_BIN_COUNT - 1;
		//auto p_tf = get_tf_array();
		//auto tf_component = get_tf_component1();
		auto histogram = get_relative_visibility_histogram();
		for (int i = 0; i < D_BIN_COUNT; i++)
		{
			tf_component[i] = histogram[i] > 0 ? histogram[i] : 0;
		}

		draw_transfer_function_component(tf_component, chartView);

		//auto chart_tf = chartView.chart();
		//chart_tf->removeAllSeries();
		//chart_tf->legend()->hide();
		//auto line_width = get_line_width(chart_tf->size().width());
		//for (int i = 0; i < D_BIN_COUNT; i++)
		//{
		//	auto c = QColor::fromRgbF((qreal)p_tf[i].x, (qreal)p_tf[i].y, (qreal)p_tf[i].z);
		//	auto line = new QLineSeries();
		//	line->append(i / N, 0);
		//	line->append(i / N, (qreal)tf_component[i]);
		//	//line->setColor(c);
		//	QPen pen(c);
		//	pen.setWidth(line_width);
		//	line->setPen(pen);
		//	chart_tf->addSeries(line);
		//}
		//chart_tf->createDefaultAxes();
		//chart_tf->setTitle(title);
		//chartView.setRenderHint(QPainter::Antialiasing);
	}

	void update_screenshots()
	{
		int n = get_next_screenshot_id(get_screenshot_id());
		int n2 = get_next_screenshot_id(n);
		int n3 = get_next_screenshot_id(n2);
		int n4 = get_next_screenshot_id(n3);
		char str[_MAX_PATH];
		sprintf(str, "~screenshot_%d.ppm", n);
		QPixmap p(str);
		sprintf(str, "~screenshot_%d.ppm", n2);
		QPixmap p2(str);
		sprintf(str, "~screenshot_%d.ppm", n3);
		QPixmap p3(str);
		sprintf(str, "~screenshot_%d.ppm", n4);
		QPixmap p4(str);
		ui.label->setPixmap(p);
		ui.label_2->setPixmap(p2);
		ui.label_3->setPixmap(p3);
		ui.label_4->setPixmap(p4);
	}

	float4 build_color(float4 colors[], float v0, float v1, float v2)
	{
		float t = v0 + v1 + v2;
		float w = t < 0 ? 0 : (t > 1 ? 1 : t);
		float4 ans;
		if (v0 > v1 && v0 > v2)
		{
			ans = colors[0];
		}
		else
		{
			if (v1 > v0 && v1 > v2)
			{
				ans = colors[1];
			}
			else
			{
				ans = colors[2];
			}
		}
		ans.w = w;
		return ans;
	}

    void on_checkBox_6_clicked();

    void on_pushButton_5_clicked();

    void on_pushButton_6_clicked();

    void on_pushButton_7_clicked();

    void on_pushButton_8_clicked();

    void on_pushButton_9_clicked();

    void on_pushButton_10_clicked();

    void on_checkBox_stateChanged(int arg1);

    void on_checkBox_2_stateChanged(int arg1);

    void on_pushButton_11_clicked();

    void on_pushButton_12_clicked();

    void on_pushButton_13_clicked();

    void on_pushButton_14_clicked();

private:
	Ui::MainWindowClass ui;
	QColor color;
	Pointer color_array = NULL;
	bool *apply_alpha = NULL;
	bool *apply_color = NULL;
	bool *apply_time_varying_tf_editing = NULL;
	bool *apply_time_varying_tf_reset = NULL;
	bool *apply_time_varying_vws_optimization = NULL;
	bool *calc_temporal_visibility = NULL;
	QChartView chartView_tf;
	QChartView chartView_relative;
	QChartView chartView_global;
	QChartView chartView_local;

	//QChartView chartView_feature0;
	//QChartView chartView_feature1;
	//QChartView chartView_feature2;
	QChartView chartView_sum;
	QChartView chartView_features[D_MAX_TF_COMPONENTS];
};
