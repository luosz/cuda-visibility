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
#include <QInputDialog>
#include <helper_math.h>
#include "ui_mainwindow.h"
#include "def.h"

// include cereal for serialization
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>
#include "serialize.h"

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
extern "C" float* get_tf_component(int i);
extern "C" float* get_tf_component_sum();
extern "C" int get_region_size();
extern "C" void set_region_size(int value);

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
		auto c2 = QColor(255 - c.red(), 255 - c.green(), 255 - c.blue());
		QPalette sample_palette;

		// for pushButton
		sample_palette.setColor(QPalette::Button, c);
		sample_palette.setColor(QPalette::ButtonText, c2);

		//// for lineEdit
		//sample_palette.setColor(QPalette::Base, c);
		//sample_palette.setColor(QPalette::Text, c2);

		ui.toolButton_4->setPalette(sample_palette);
		ui.toolButton_4->setText(c.name());
	}

	void set_button_color(QAbstractButton &button, const QColor &c)
	{
		auto c2 = QColor(255 - c.red(), 255 - c.green(), 255 - c.blue());
		QPalette sample_palette;
		sample_palette.setColor(QPalette::Button, c);
		sample_palette.setColor(QPalette::ButtonText, c2);
		button.setPalette(sample_palette);
		button.setText(c.name());
	}

	void set_button_color_dialog(QAbstractButton &button)
	{
		auto c = QColorDialog::getColor(button.palette().color(QPalette::Button));
		if (c.isValid())
		{
			set_button_color(button, c);
		}
	}

	float4 get_button_color(const QAbstractButton &button)
	{
		auto c = button.palette().color(QPalette::Button);
		return make_float4(c.redF(), c.greenF(), c.blueF(), c.alphaF());
	}

	QColor float4_to_QColor(float4 c)
	{
		//return QColor::fromRgbF(c.x, c.y, c.z, c.w);
		return QColor::fromRgbF(c.x, c.y, c.z);
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
		delay_draw_transfer_function_and_visibility_histograms();
	}

	void draw_transfer_function_component_in_lines(const float tf_component[], const float4 tf[], QChartView &chartView)
	{
		const qreal N = D_BIN_COUNT - 1;
		auto chart = chartView.chart();
		chart->removeAllSeries();
		chart->legend()->hide();
		auto line_width = get_line_width(chart->size().width());
		for (int i = 0; i < D_BIN_COUNT; i++)
		{
			auto c = float4_to_QColor(tf[i]);
			auto line = new QLineSeries();
			qreal intensity = i / N;
			line->append(intensity, 0);
			line->append(intensity, (qreal)tf_component[i]);
			//line->setColor(c);
			QPen pen(c);
			pen.setWidth(line_width);
			line->setPen(pen);
			chart->addSeries(line);
		}
		chart->createDefaultAxes();
		chartView.setRenderHint(QPainter::Antialiasing);
	}

	void draw_transfer_function_component_in_area(const float tf_component[], const float4 tf[], QChartView &chartView)
	{
		const qreal N = D_BIN_COUNT - 1;
		auto chart_tf = chartView.chart();
		chart_tf->removeAllSeries();
		chart_tf->legend()->hide();
		QLineSeries *series0 = new QLineSeries();
		QLineSeries *series1 = new QLineSeries();
		QLinearGradient gradient(QPointF(0, 0), QPointF(1, 0));
		for (int i = 0; i < D_BIN_COUNT; i++)
		{
			auto c = float4_to_QColor(tf[i]);
			qreal intensity = i / N;
			*series0 << QPointF(intensity, 0);
			*series1 << QPointF(intensity, (qreal)tf_component[i]);
			gradient.setColorAt(intensity, c);
		}
		QAreaSeries *series = new QAreaSeries(series0, series1);
		gradient.setCoordinateMode(QGradient::ObjectBoundingMode);
		series->setBrush(gradient);
		chart_tf->addSeries(series);
		chart_tf->createDefaultAxes();
		chartView.setRenderHint(QPainter::Antialiasing);
	}

	void draw_transfer_function_component(const float tf_component[], const float4 tf[], QChartView &chartView)
	{
		if (ui.action_Smooth_transfer_functions->isChecked())
		{
			draw_transfer_function_component_in_area(tf_component, tf, chartView);
		} 
		else
		{
			draw_transfer_function_component_in_lines(tf_component, tf, chartView);
		}
	}

	qreal get_line_width(qreal chart_width)
	{
		return chart_width / D_BIN_COUNT;
	}

	void draw_histogram(const float histogram[], QChartView &chartView)
	{
		const qreal N = D_BIN_COUNT - 1;
		auto chart = chartView.chart();
		chart->removeAllSeries();
		chart->legend()->hide();
		auto line_width = get_line_width(chart->size().width());
		for (int i = 0; i < D_BIN_COUNT; i++)
		{
			auto c = QColor::fromRgbF(0.5, 0.5, 0.5);
			auto line = new QLineSeries();
			qreal intensity = i / N;
			line->append(intensity, 0);
			line->append(intensity, (qreal)histogram[i]);
			//line->setColor(c);
			QPen pen(c);
			pen.setWidth(line_width);
			line->setPen(pen);
			chart->addSeries(line);
		}
		chart->createDefaultAxes();
		chartView.setRenderHint(QPainter::Antialiasing);
	}

	void draw_transfer_function_in_area(const float4 tf[], QChartView &chartView)
	{
		const qreal N = D_BIN_COUNT - 1;
		auto chart = chartView.chart();
		chart->removeAllSeries();
		chart->legend()->hide();
		QLineSeries *series0 = new QLineSeries();
		QLineSeries *series1 = new QLineSeries();
		QLinearGradient gradient(QPointF(0, 0), QPointF(1, 0));
		for (int i = 0; i < D_BIN_COUNT; i++)
		{
			qreal intensity = i / N;
			auto c = float4_to_QColor(tf[i]);
			*series0 << QPointF(intensity, 0);
			*series1 << QPointF(intensity, (qreal)tf[i].w);
			gradient.setColorAt(intensity, c);
		}
		QAreaSeries *series = new QAreaSeries(series0, series1);
		gradient.setCoordinateMode(QGradient::ObjectBoundingMode);
		series->setBrush(gradient);
		chart->addSeries(series);
		chart->createDefaultAxes();
		chartView.setRenderHint(QPainter::Antialiasing);
	}

	void draw_transfer_function_in_lines(const float4 tf[], QChartView &chartView)
	{
		const qreal N = D_BIN_COUNT - 1;
		auto chart = chartView.chart();
		chart->removeAllSeries();
		chart->legend()->hide();
		auto line_width = get_line_width(chart->size().width());
		for (int i = 0; i < D_BIN_COUNT; i++)
		{
			auto c = float4_to_QColor(tf[i]);
			auto line = new QLineSeries();
			qreal intensity = i / N;
			line->append(intensity, 0);
			line->append(intensity, (qreal)tf[i].w);
			//line->setColor(c);
			QPen pen(c);
			pen.setWidth(line_width);
			line->setPen(pen);
			chart->addSeries(line);
		}
		chart->createDefaultAxes();
		chartView.setRenderHint(QPainter::Antialiasing);
	}

	void draw_transfer_function(const float4 tf[], QChartView &chartView)
	{
		if (ui.action_Smooth_transfer_functions->isChecked())
		{
			draw_transfer_function_in_area(tf, chartView);
		}
		else
		{
			draw_transfer_function_in_lines(tf, chartView);
		}
	}

	void clear_transfer_function_components()
	{
		for (int i = 0; i < tf_component_number; i++)
		{
			memset(get_tf_component(i), 0, sizeof(float)*D_BIN_COUNT);
		}
		draw_all_transfer_function_components();
	}

	void draw_all_transfer_function_components()
	{
		auto tf = get_tf_array();
		for (int i = 0; i < tf_component_number; i++)
		{
			draw_transfer_function_component(get_tf_component(i), tf, chartView_features[i]);
		}
	}

	void update_all_transfer_functions_and_histograms()
	{
		draw_transfer_functions();
		draw_all_histograms();
		draw_all_transfer_function_components();
	}

	void delay_draw_transfer_function_and_visibility_histograms(int msec = 10)
	{
		QTimer::singleShot(msec, this, SLOT(draw_transfer_functions_and_histograms()));
	}

	void delay_add_transfer_function_component(float tf_component[], QChartView &chartView, int msec = 50)
	{
		// Use a lambda expression with a capture list for the Qt slot with arguments
		QTimer::singleShot(msec, this, [this, tf_component, &chartView]() {add_transfer_function_component(tf_component, chartView); });
	}

	void delay_set_button_color_to_component_peak_color(QAbstractButton &button, const float tf_component[], const float4 tf[], int msec = 90)
	{
		// Use a lambda expression with a capture list for the Qt slot with arguments
		QTimer::singleShot(msec, this, [this, tf_component, tf, &button]() {set_button_color_to_component_peak_color(button, tf_component, tf); });
	}

	QString enter_chart_title(int i = 0)
	{
		if (i >= 0 && i < D_MAX_TF_COMPONENTS)
		{
			bool ok;
			QString text = QInputDialog::getText(this, tr("Enter Chart Title"),
				tr("Chart title:"), QLineEdit::Normal,
				chartView_features[i].chart()->title(), &ok);
			if (ok && !text.isEmpty())
			{
				//qDebug() << text;
				chartView_features[i].chart()->setTitle(text);
			}
			return chartView_features[i].chart()->title();
		}
		else
		{
			return "";
		}
	}

	void save_chart_to_image(QChartView &chartview)
	{
		//qDebug() << "chart size " << chartview.size() << chartview.chart()->size();
		auto size = chartview.size();
		QPixmap p(size);
		QPainter painter(&p);
		painter.fillRect(0, 0, size.width(), size.height(), Qt::white);
		chartview.render(&painter);
		QString str = "~" + chartview.chart()->title() + ".png";
		str.replace(" ", "_");
		p.save(str, "PNG");
	}

	float4 build_color(float4 colors[], float v[], float4 tf)
	{
		float t = 0;
		int max_index = 0;
		float max_value = v[0];
		// merge visible transfer function components
		for (int i = 0; i < tf_component_number; i++)
		{
			t += v[i];
			if (v[i] > max_value)
			{
				max_value = v[i];
				max_index = i;
			}
		}
		float w = t < 0 ? 0 : (t > 1 ? 1 : t);
		float4 ans = w > 0 ? colors[max_index] : tf;
		ans.w = w;
		return ans;
	}

	float4 blend_colors(float4 colors[], float v[], float4 tf)
	{
		float t = 0;
		int max_index = 0;
		float max_value = v[0];
		// merge visible transfer function components
		for (int i = 0; i < tf_component_number; i++)
		{
			t += v[i];
			if (v[i] > max_value)
			{
				max_value = v[i];
				max_index = i;
			}
		}
		if (t>0)
		{
			qreal weights[D_MAX_TF_COMPONENTS] = { 0 };
			float4 c = make_float4(0, 0, 0, 0);
			for (int i = 0; i < tf_component_number; i++)
			{
				weights[i] = v[i] / t;
				c += weights[i] * colors[i];
			}
			float w = t < 0 ? 0 : (t > 1 ? 1 : t);
			float4 ans = w > 0 ? c : tf;
			ans.w = w;
			return ans;
		}
		else
		{
			float w = t < 0 ? 0 : (t > 1 ? 1 : t);
			float4 ans = w > 0 ? colors[max_index] : tf;
			ans.w = w;
			return ans;
		}
	}

	void hide_extra_tf_component_frames()
	{
		QWidget *w[D_MAX_TF_COMPONENTS] = { ui.frame,ui.frame_2,ui.frame_3,ui.frame_4,ui.frame_5 };
		for (int i = 0; i < D_MAX_TF_COMPONENTS; i++)
		{
			w[i]->setVisible(i < tf_component_number);
		}
	}

	void draw_all_histograms()
	{
		draw_histogram(get_relative_visibility_histogram(), chartView_relative);
		draw_histogram(get_global_visibility_histogram(), chartView_global);
		draw_histogram(get_local_visibility_histogram(), chartView_local);
	}

	void draw_transfer_functions()
	{
		auto tf = get_tf_array();
		draw_transfer_function(tf, chartView_tf);
		draw_transfer_function(tf, chartView_sum);
	}

	void set_tf_component_number(int arg1)
	{
		tf_component_number = arg1 < 1 ? 1 : (arg1 > D_MAX_TF_COMPONENTS ? D_MAX_TF_COMPONENTS : arg1);
		hide_extra_tf_component_frames();
	}

	void save_tf_component_properties(const char *file)
	{
		printf("save TF component properties to %s\n", file);
		std::ofstream os(file);
		//cereal::XMLOutputArchive archive(os);
		cereal::JSONOutputArchive archive(os);
		archive(CEREAL_NVP(tf_component_number));

		QAbstractButton *buttons[D_MAX_TF_COMPONENTS] = {
			ui.toolButton,
			ui.toolButton_2,
			ui.toolButton_3,
			ui.toolButton_8,
			ui.toolButton_9
		};
		float4 tf_component_colors[D_MAX_TF_COMPONENTS] = { 0 };
		for (int i = 0; i < D_MAX_TF_COMPONENTS; i++)
		{
			tf_component_colors[i] = get_button_color(*buttons[i]);
		}
		archive(CEREAL_NVP(tf_component_colors));
		archive(CEREAL_NVP(tf_component_weights));
		std::string tf_component_titles[D_MAX_TF_COMPONENTS];
		for (int i = 0; i < D_MAX_TF_COMPONENTS; i++)
		{
			tf_component_titles[i] = chartView_features[i].chart()->title().toStdString();
		}
		archive(CEREAL_NVP(tf_component_titles));
	}

	void load_tf_component_properties(const char *file)
	{
		QAbstractButton *buttons[D_MAX_TF_COMPONENTS] = {
			ui.toolButton,
			ui.toolButton_2,
			ui.toolButton_3,
			ui.toolButton_8,
			ui.toolButton_9
		};
		float4 tf_component_colors[D_MAX_TF_COMPONENTS] = { 0 };
		std::string tf_component_titles[D_MAX_TF_COMPONENTS];
		QDoubleSpinBox *spinboxes[D_MAX_TF_COMPONENTS] = {
			ui.doubleSpinBox,
			ui.doubleSpinBox_2,
			ui.doubleSpinBox_3,
			ui.doubleSpinBox_4,
			ui.doubleSpinBox_5
		};

		std::ifstream is(file);
		if (is.is_open())
		{
			printf("load TF component properties from %s\n", file);
			int regionSize = get_region_size();
			//cereal::XMLInputArchive archive(is);
			cereal::JSONInputArchive archive(is);
			archive(CEREAL_NVP(tf_component_number));
			set_tf_component_number(tf_component_number);
			ui.spinBox->setValue(tf_component_number);

			archive(CEREAL_NVP(tf_component_colors));
			for (int i = 0; i < D_MAX_TF_COMPONENTS; i++)
			{
				set_button_color(*buttons[i], float4_to_QColor(tf_component_colors[i]));
			}
			archive(CEREAL_NVP(tf_component_weights));
			for (int i = 0; i < D_MAX_TF_COMPONENTS; i++)
			{
				spinboxes[i]->setValue(tf_component_weights[i]);
			}
			archive(CEREAL_NVP(tf_component_titles));
			for (int i = 0; i < D_MAX_TF_COMPONENTS; i++)
			{
				chartView_features[i].chart()->setTitle(QString::fromStdString(tf_component_titles[i]));
			}
		}
		else
		{
			printf("cannot open %s\n", file);
		}
	}

private slots:
	void draw_transfer_functions_and_histograms()
	{
		draw_transfer_functions();
		draw_all_histograms();
	}

	void add_transfer_function_component(float tf_component[], QChartView &chartView)
	{
		auto histogram = get_relative_visibility_histogram();
		for (int i = 0; i < D_BIN_COUNT; i++)
		{
			tf_component[i] = histogram[i] > 0 ? histogram[i] : 0;
		}

		draw_transfer_function_component(tf_component, get_tf_array(), chartView);
	}

	void set_button_color_to_component_peak_color(QAbstractButton &button, const float tf_component[], const float4 tf[])
	{
		float max = 0;
		int index = -1;
		for (int i = 0; i < D_BIN_COUNT; i++)
		{
			if (tf_component[i] > max)
			{
				index = i;
				max = tf_component[i];
			}
		}
		if (-1 != index)
		{
			set_button_color(button, float4_to_QColor(tf[index]));
		}
		std::cout << "index=" << index << "\t max=" << max << std::endl;
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

	void on_pushButton_3_clicked();

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

    void on_checkBox_6_clicked();

    void on_pushButton_5_clicked();

    void on_pushButton_6_clicked();

    void on_checkBox_stateChanged(int arg1);

    void on_checkBox_2_stateChanged(int arg1);

    void on_toolButton_clicked();

    void on_toolButton_2_clicked();

    void on_toolButton_3_clicked();

    void on_toolButton_4_clicked();

    void on_pushButton_clicked();

    void on_toolButton_5_clicked();

    void on_toolButton_6_clicked();

    void on_toolButton_7_clicked();

    void on_doubleSpinBox_valueChanged(double arg1);

    void on_doubleSpinBox_2_valueChanged(double arg1);

    void on_doubleSpinBox_3_valueChanged(double arg1);

    void on_action_Clear_transfer_function_components_triggered();

    void on_spinBox_valueChanged(int arg1);

    void on_pushButton_2_clicked();

    void on_toolButton_8_clicked();

    void on_toolButton_9_clicked();

    void on_action_Smooth_transfer_functions_triggered();

    void on_action_Refresh_transfer_functions_and_histograms_triggered();

    void on_doubleSpinBox_4_valueChanged(double arg1);

    void on_doubleSpinBox_5_valueChanged(double arg1);

    void on_toolButton_10_clicked();

    void on_toolButton_11_clicked();

    void on_action_Modify_region_size_triggered();

    void on_toolButton_12_clicked();

    void on_toolButton_13_clicked();

    void on_toolButton_14_clicked();

    void on_toolButton_15_clicked();

    void on_toolButton_16_clicked();

    void on_action_Save_charts_to_images_triggered();

    void on_actionLoad_TF_component_properties_triggered();

    void on_actionSave_TF_component_properties_as_triggered();

    void on_action_Blend_TF_components_triggered();

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

	QChartView chartView_sum;
	QChartView chartView_features[D_MAX_TF_COMPONENTS];
	qreal tf_component_weights[D_MAX_TF_COMPONENTS] = { 0 };
	int tf_component_number = D_MAX_TF_COMPONENTS;
};
