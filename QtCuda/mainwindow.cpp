#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
{
	this->setGeometry(
		QStyle::alignedRect(
			Qt::LeftToRight,
			Qt::AlignRight|Qt::AlignVCenter,
			this->size(),
			qApp->desktop()->availableGeometry()
		)
	);
	this->move(this->pos() - QPoint(400, 300));
	ui.setupUi(this);

	update_color(QColor::fromRgbF(D_RGBA[0], D_RGBA[1], D_RGBA[2], D_RGBA[3]));
	ui.verticalLayout->addWidget(&chartView_tf);
	ui.verticalLayout->addWidget(&chartView_relative);
	ui.verticalLayout->addWidget(&chartView_global);
	ui.verticalLayout->addWidget(&chartView_local);

	QBoxLayout *layouts[D_MAX_TF_COMPONENTS] = {
		ui.horizontalLayout,
		ui.horizontalLayout_2,
		ui.horizontalLayout_3,
		ui.horizontalLayout_4,
		ui.horizontalLayout_5
	};

	// Set chart titles
	chartView_tf.chart()->setTitle("Transfer function");
	chartView_relative.chart()->setTitle("Relative visibility histogram");
	chartView_global.chart()->setTitle("Global visibility histogram");
	chartView_local.chart()->setTitle("Local visibility histogram");
	for (int i = 0; i < D_MAX_TF_COMPONENTS; i++)
	{
		char str[_MAX_PATH];
		sprintf(str, "Component %d", i);
		chartView_features[i].chart()->setTitle(str);
		layouts[i]->addWidget(&chartView_features[i]);
	}
	chartView_sum.chart()->setTitle("Merged Transfer function");
	ui.horizontalLayout_6->addWidget(&chartView_sum);

	for (int i = 0; i < D_MAX_TF_COMPONENTS; i++)
	{
		tf_component_weights[i] = 1;
	}

	update_screenshots();

	// draw transfer functions and histograms after rendering is done
	QTimer::singleShot(1000, this, [this]() {
		update_all_transfer_functions_and_histograms();
		ui.spinBox->setValue(D_MAX_TF_COMPONENTS - 2);
	});
}

void MainWindow::on_pushButton_3_clicked()
{
	apply_tf_editing();
	delay_draw_transfer_function_and_visibility_histograms();

	//const int n = 11;
	//int r = 2;
	//float d[n];
	//calc_gaussian_kernel(d, 2 * r + 1, sigma(r));
	//for (int i = 0; i < 2 * r + 1; i++)
	//{
	//	std::cout << d[i] << std::ends;
	//}
	//std::cout << std::endl;
	//r = 4;
	//calc_gaussian_kernel(d, 2 * r + 1, sigma(r));
	//for (int i = 0; i < 2 * r + 1; i++)
	//{
	//	std::cout << d[i] << std::ends;
	//}
	//std::cout << std::endl;
}

void MainWindow::on_checkBox_3_clicked()
{
    if (apply_time_varying_tf_editing)
    {
        *apply_time_varying_tf_editing = ui.checkBox_3->isChecked();
        std::cout << "apply_time_varying_tf_editing=" << (*apply_time_varying_tf_editing ? "true" : "false") << std::endl;
		if (ui.checkBox_3->isChecked())
		{
			ui.checkBox_5->setChecked(false);
			on_checkBox_5_clicked();
		}
    }
}

void MainWindow::on_checkBox_4_clicked()
{
    if (apply_time_varying_tf_reset)
    {
        *apply_time_varying_tf_reset = ui.checkBox_4->isChecked();
        std::cout << "apply_time_varying_tf_reset=" << (*apply_time_varying_tf_reset ? "true" : "false") << std::endl;
    }
}

void MainWindow::on_action_About_triggered()
{
    std::cout<<((float)5/2)<<std::endl;
	int ret = QMessageBox::information(this, tr("QtCuda Volume Visualizer"),
		tr("Copyright (c) 2017 Trinity College Dublin, The University of Dublin.\n" "All rights reserved."), QMessageBox::Ok);
}

void MainWindow::on_action_Exit_triggered()
{
	QApplication::quit();
}

void MainWindow::on_action_Open_triggered()
{
	QString filename = QFileDialog::getOpenFileName(this, tr("Mhd files"), QDir::currentPath(), tr("Mhd files (*.mhd);;All files (*.*)"));
	if (!filename.isNull())
	{
		auto str = QFileInfo(filename).fileName();
		load_mhd_file(str.toStdString());
	}
}

void MainWindow::on_actionOpen_Files_triggered()
{
	std::vector<std::string> filelist;
	QStringList filenames = QFileDialog::getOpenFileNames(this, tr("Raw files"), QDir::currentPath(), tr("Raw files (*.raw);;All files (*.*)"));
	if (!filenames.isEmpty())
	{
		for (int i = 0;i < filenames.count();i++)
		{
			auto filename = QFileInfo(filenames.at(i)).fileName();
			filelist.push_back(filename.toStdString());
		}
		add_volume_to_list_for_update_from_vector(filelist);
	}
}

void MainWindow::on_actionOpen_transfer_function_triggered()
{
	QString filename = QFileDialog::getOpenFileName(this, tr("Open transfer function"), QDir::currentPath(), tr("Transfer function (*.tfi);;All Files (*)"));
	if (!filename.isNull())
	{
		qDebug() << filename;
		openTransferFunctionFromVoreenXML(filename.toStdString().c_str());
		bind_tf_texture();
		draw_transfer_functions_and_histograms();
	}
}

void MainWindow::on_actionSave_transfer_function_as_triggered()
{
	QString filename = QFileDialog::getSaveFileName(this, tr("Save transfer function"), QDir::currentPath(), tr("Transfer function (*.tfi);;All Files (*)"));
	if (!filename.isNull())
	{
		qDebug() << filename;
		save_tf_array_to_voreen_XML(filename.toStdString().c_str());
	}
}

void MainWindow::on_actionLoad_view_and_region_triggered()
{
	QString filename = QFileDialog::getOpenFileName(this, tr("Open view"), QDir::currentPath() + QDir::separator() + "~view.xml", tr("View and selection (*.xml);;All Files (*)"));
	if (!filename.isNull())
	{
		// extract filename from path
		auto str = QFileInfo(filename).fileName();
		qDebug() << str;
		load_view(str.toStdString().c_str());
	}
}

void MainWindow::on_actionSave_view_and_region_as_triggered()
{
	QString filename = QFileDialog::getSaveFileName(this, tr("Save view"), QDir::currentPath() + QDir::separator() + "~view.xml", tr("View and selection (*.xml);;All Files (*)"));
	if (!filename.isNull())
	{
		// extract filename from path
		auto str = QFileInfo(filename).fileName();
		qDebug() << str;
		save_view(str.toStdString().c_str());
	}
}

void MainWindow::on_checkBox_5_clicked()
{
	if (apply_time_varying_vws_optimization)
	{
		*apply_time_varying_vws_optimization = ui.checkBox_5->isChecked();
		std::cout << "apply_time_varying_vws_optimization=" << (*apply_time_varying_vws_optimization ? "true" : "false") << std::endl;
		if (ui.checkBox_5->isChecked())
		{
			ui.checkBox_3->setChecked(false);
			on_checkBox_3_clicked();
		}
	}
}

void MainWindow::on_checkBox_6_clicked()
{
	if (calc_temporal_visibility)
	{
		*calc_temporal_visibility = ui.checkBox_6->isChecked();
		std::cout << "calc_temporal_visibility=" << (*calc_temporal_visibility ? "true" : "false") << std::endl;
	}
}

void MainWindow::on_pushButton_5_clicked()
{
	apply_temporal_tf_editing();
	delay_draw_transfer_function_and_visibility_histograms();
}

void MainWindow::on_pushButton_6_clicked()
{
	update_screenshots();
}

void MainWindow::on_checkBox_stateChanged(int arg1)
{
	if (apply_alpha && apply_color)
	{
		*apply_alpha = ui.checkBox->isChecked();
		std::cout << "on_checkBox_stateChanged \t Apply alpha: " << (*apply_alpha ? "true" : "false") << "\t Apply color: " << (*apply_color ? "true" : "false") << std::endl;
	}
}

void MainWindow::on_checkBox_2_stateChanged(int arg1)
{
	if (apply_alpha && apply_color)
	{
		*apply_color = ui.checkBox_2->isChecked();
		std::cout << "on_checkBox_2_stateChanged \t Apply alpha: " << (*apply_alpha ? "true" : "false") << "\t Apply color: " << (*apply_color ? "true" : "false") << std::endl;
	}
}

void MainWindow::on_toolButton_clicked()
{
	set_button_color_dialog(*ui.toolButton);
}

void MainWindow::on_toolButton_2_clicked()
{
	set_button_color_dialog(*ui.toolButton_2);
}

void MainWindow::on_toolButton_3_clicked()
{
	set_button_color_dialog(*ui.toolButton_3);
}

void MainWindow::on_toolButton_4_clicked()
{
	auto c = QColorDialog::getColor(color);
	if (c.isValid())
	{
		update_color(c);
	}
}

void MainWindow::on_pushButton_clicked()
{
	// merge transfer function components
	const qreal N = D_BIN_COUNT - 1;
	auto tf = get_tf_array();
	auto tf_sum = get_tf_component_sum();
	memset(tf_sum, 0, sizeof(float)*D_BIN_COUNT);
	float *components[D_MAX_TF_COMPONENTS];
	float4 sum[D_BIN_COUNT] = { 0 };
	float4 colors[D_MAX_TF_COMPONENTS] = { 0 };
	QAbstractButton *buttons[D_MAX_TF_COMPONENTS] = {
		ui.toolButton,
		ui.toolButton_2,
		ui.toolButton_3,
		ui.toolButton_8,
		ui.toolButton_9
	};
	for (int i = 0; i < D_MAX_TF_COMPONENTS; i++)
	{
		components[i] = get_tf_component(i);
		colors[i] = get_button_color(*buttons[i]);
	}

	//qreal total = 0;
	//qreal weights[D_MAX_TF_COMPONENTS] = { 0 };
	//if (ui.action_Blend_TF_components->isChecked())
	//{
	//	qreal normalized_weights[D_MAX_TF_COMPONENTS] = { 0 };
	//	for (int i = 0; i < tf_component_number; i++)
	//	{
	//		total += tf_component_weights[i];
	//	}
	//	if (total > 0)
	//	{
	//		for (int i = 0; i < tf_component_number; i++)
	//		{
	//			normalized_weights[i] = tf_component_weights[i] / total;
	//		}
	//		memcpy(weights, normalized_weights, D_MAX_TF_COMPONENTS * sizeof(qreal));
	//	}
	//}
	//if (total <= 0)
	//{
	//	memcpy(weights, tf_component_weights, D_MAX_TF_COMPONENTS * sizeof(qreal));
	//}

	if (ui.action_Blend_TF_components->isChecked())
	{
		// blend tf component colors
		for (int i = 0; i < D_BIN_COUNT; i++)
		{
			float tmp[D_MAX_TF_COMPONENTS];
			for (int j = 0; j < D_MAX_TF_COMPONENTS; j++)
			{
				tmp[j] = components[j][i] * tf_component_weights[j];
			}
			sum[i] = blend_colors(colors, tmp, tf[i]);
			tf_sum[i] = sum[i].w;
		}
	}
	else
	{
		// pick dominant color
		for (int i = 0; i < D_BIN_COUNT; i++)
		{
			float tmp[D_MAX_TF_COMPONENTS];
			for (int j = 0; j < D_MAX_TF_COMPONENTS; j++)
			{
				tmp[j] = components[j][i] * tf_component_weights[j];
			}
			sum[i] = build_color(colors, tmp, tf[i]);
			tf_sum[i] = sum[i].w;
		}
	}

	memcpy(tf, sum, sizeof(float4)*D_BIN_COUNT);
	bind_tf_texture();
	draw_transfer_functions();
}

void MainWindow::on_toolButton_5_clicked()
{
	calculate_visibility_without_editing_tf();
	const int i = 0;
	delay_add_transfer_function_component(get_tf_component(i), chartView_features[i]);
	delay_set_button_color_to_component_peak_color(*ui.toolButton, get_tf_component(i), get_tf_array());
}

void MainWindow::on_toolButton_6_clicked()
{
	calculate_visibility_without_editing_tf();
	const int i = 1;
	delay_add_transfer_function_component(get_tf_component(i), chartView_features[i]);
	delay_set_button_color_to_component_peak_color(*ui.toolButton_2, get_tf_component(i), get_tf_array());
}

void MainWindow::on_toolButton_7_clicked()
{
	calculate_visibility_without_editing_tf();
	const int i = 2;
	delay_add_transfer_function_component(get_tf_component(i), chartView_features[i]);
	delay_set_button_color_to_component_peak_color(*ui.toolButton_3, get_tf_component(i), get_tf_array());
}

void MainWindow::on_doubleSpinBox_valueChanged(double arg1)
{
	tf_component_weights[0] = arg1;
}

void MainWindow::on_doubleSpinBox_2_valueChanged(double arg1)
{
	tf_component_weights[1] = arg1;
}

void MainWindow::on_doubleSpinBox_3_valueChanged(double arg1)
{
	tf_component_weights[2] = arg1;
}

void MainWindow::on_action_Clear_transfer_function_components_triggered()
{
	clear_transfer_function_components();
}

void MainWindow::on_spinBox_valueChanged(int arg1)
{
	//tf_component_number = arg1 < 1 ? 1 : (arg1 > D_MAX_TF_COMPONENTS ? D_MAX_TF_COMPONENTS : arg1);
	//hide_extra_tf_component_frames();
	set_tf_component_number(arg1);
}

void MainWindow::on_pushButton_2_clicked()
{
	reset_transfer_function();
	delay_draw_transfer_function_and_visibility_histograms();

	////float gaussian1[R1*R1*R1] = { 0 };
	//float a;
	//std::ifstream myfile;
	//myfile.open("~gaussian_5_5_5.txt");
	//myfile >> a;

	//int n = R1*R1*R1;
	//for (int i=0;i<n;i++)
	//{
	//	myfile >> a;
	//	gaussian1[i] = a;
	//	std::cout << gaussian1[i] << std::ends;
	//}
	//std::cout << std::endl;
	//myfile.close();
}

void MainWindow::on_toolButton_8_clicked()
{
	set_button_color_dialog(*ui.toolButton_8);
}

void MainWindow::on_toolButton_9_clicked()
{
	set_button_color_dialog(*ui.toolButton_9);
}

void MainWindow::on_action_Smooth_transfer_functions_triggered()
{
	update_all_transfer_functions_and_histograms();
}

void MainWindow::on_action_Refresh_transfer_functions_and_histograms_triggered()
{
	update_all_transfer_functions_and_histograms();
}

void MainWindow::on_doubleSpinBox_4_valueChanged(double arg1)
{
	tf_component_weights[3] = arg1;
}

void MainWindow::on_doubleSpinBox_5_valueChanged(double arg1)
{
	tf_component_weights[4] = arg1;
}

void MainWindow::on_toolButton_10_clicked()
{
	calculate_visibility_without_editing_tf();
	const int i = 3;
	delay_add_transfer_function_component(get_tf_component(i), chartView_features[i]);
	delay_set_button_color_to_component_peak_color(*ui.toolButton_8, get_tf_component(i), get_tf_array());
}

void MainWindow::on_toolButton_11_clicked()
{
	calculate_visibility_without_editing_tf();
	const int i = 4;
	delay_add_transfer_function_component(get_tf_component(i), chartView_features[i]);
	delay_set_button_color_to_component_peak_color(*ui.toolButton_9, get_tf_component(i), get_tf_array());
}

void MainWindow::on_action_Modify_region_size_triggered()
{
	int region_size = get_region_size();
	char str[_MAX_PATH];
	sprintf(str, "%d", region_size);
	const int MAX_SIZE = min(D_WIDTH, D_HEIGHT) / 2;
	char label[_MAX_PATH];
	sprintf(label, "Enter a new region size in the range (0,%d]", MAX_SIZE);
	bool ok;
	QString text = QInputDialog::getText(this, tr("Modify region size"),
		tr(label), QLineEdit::Normal, tr(str), &ok);
	if (ok && !text.isEmpty())
	{
		QTextStream s(&text);
		int size;
		s >> size;
		if (size != region_size && size > 0 && size <= MAX_SIZE)
		{
			set_region_size(size);
			std::cout << "region size " << get_region_size() << std::endl;
		}
	}
}

void MainWindow::on_toolButton_12_clicked()
{
	enter_chart_title(0);
}

void MainWindow::on_toolButton_13_clicked()
{
	enter_chart_title(1);
}

void MainWindow::on_toolButton_14_clicked()
{
	enter_chart_title(2);
}

void MainWindow::on_toolButton_15_clicked()
{
	enter_chart_title(3);
}

void MainWindow::on_toolButton_16_clicked()
{
	enter_chart_title(4);
}

void MainWindow::on_action_Save_charts_to_images_triggered()
{
    save_chart_to_image(chartView_tf);
    save_chart_to_image(chartView_relative);
    save_chart_to_image(chartView_global);
    save_chart_to_image(chartView_local);
    save_chart_to_image(chartView_sum);
    for (int i = 0; i < tf_component_number; i++)
    {
        save_chart_to_image(chartView_features[i]);
    }
}

void MainWindow::on_actionLoad_TF_component_properties_triggered()
{
	QString filename = QFileDialog::getOpenFileName(this, tr("Open properties"), QDir::currentPath() + QDir::separator() + "~properties.json", tr("Properties (*.json);;All Files (*)"));
	if (!filename.isNull())
	{
		// extract filename from path
		auto str = QFileInfo(filename).fileName();
		qDebug() << str;
		load_tf_component_properties(str.toStdString().c_str());
	}
}

void MainWindow::on_actionSave_TF_component_properties_as_triggered()
{
	QString filename = QFileDialog::getSaveFileName(this, tr("Save properties"), QDir::currentPath() + QDir::separator() + "~properties.json", tr("Properties (*.json);;All Files (*)"));
	if (!filename.isNull())
	{
		// extract filename from path
		auto str = QFileInfo(filename).fileName();
		qDebug() << str;
		save_tf_component_properties(str.toStdString().c_str());
	}
}

void MainWindow::on_action_Blend_TF_components_triggered()
{

}
