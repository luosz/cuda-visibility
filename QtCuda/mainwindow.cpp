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

	//// enable auto fill background
	//ui.toolButton_4->setAutoFillBackground(true);

	update_color(QColor::fromRgbF(D_RGBA[0], D_RGBA[1], D_RGBA[2], D_RGBA[3]));
	ui.verticalLayout->addWidget(&chartView_tf);
	ui.verticalLayout->addWidget(&chartView_relative);
	ui.verticalLayout->addWidget(&chartView_global);
	ui.verticalLayout->addWidget(&chartView_local);

	ui.horizontalLayout->addWidget(&chartView_features[0]);
	ui.horizontalLayout_2->addWidget(&chartView_features[1]);
	ui.horizontalLayout_3->addWidget(&chartView_features[2]);

	// Set chart titles
	chartView_tf.chart()->setTitle("Transfer function");
	chartView_relative.chart()->setTitle("Relative visibility histogram");
	chartView_global.chart()->setTitle("Global visibility histogram");
	chartView_local.chart()->setTitle("Local visibility histogram");
	chartView_features[0].chart()->setTitle("Transfer function component 0");
	chartView_features[1].chart()->setTitle("Transfer function component 1");
	chartView_features[2].chart()->setTitle("Transfer function component 2");

	ui.horizontalLayout_9->addWidget(&chartView_sum);
	chartView_sum.chart()->setTitle("Merged transfer function");

	// show the first tab by default
	ui.tabWidget_2->setCurrentIndex(0);

	for (int i = 0; i < D_MAX_TF_COMPONENTS; i++)
	{
		tf_component_weights[i] = 1;
	}

	delay_draw_transfer_function_and_visibility_histograms(1000);
	update_screenshots();
}

void MainWindow::on_pushButton_2_clicked()
{
	update_all_transfer_functions_and_histograms();
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

void MainWindow::on_pushButton_4_clicked()
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
		draw_transfer_function_and_histograms();
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
	QString filename = QFileDialog::getOpenFileName(this, tr("Open view"), QDir::currentPath(), tr("View (*.xml);;All Files (*)"));
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
	QString filename = QFileDialog::getSaveFileName(this, tr("Save view"), QDir::currentPath(), tr("View (*.xml);;All Files (*)"));
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
	const qreal N = D_BIN_COUNT - 1;
	auto tf = get_tf_array();
	auto tf_sum = get_tf_component3();
	auto tf0 = get_tf_component0();
	auto tf1 = get_tf_component1();
	auto tf2 = get_tf_component2();
	memset(tf_sum, 0, sizeof(float)*D_BIN_COUNT);
	float4 sum[D_BIN_COUNT] = { 0 };
	float4 colors[3] = { 0 };
	colors[0] = get_button_color(*ui.toolButton);
	colors[1] = get_button_color(*ui.toolButton_2);
	colors[2] = get_button_color(*ui.toolButton_3);

	for (int i = 0; i < D_BIN_COUNT; i++)
	{
		float t0 = tf0[i] * tf_component_weights[0];
		float t1 = tf1[i] * tf_component_weights[1];
		float t2 = tf2[i] * tf_component_weights[2];
		float t = t0 + t1 + t2;
		tf_sum[i] = t < 0 ? 0 : (t > 1 ? 1 : t);
		sum[i] = build_color(colors, t0, t1, t2, tf[i]);
	}
	memcpy(tf, sum, sizeof(float4)*D_BIN_COUNT);
	bind_tf_texture();
	draw_transfer_function(tf, chartView_sum);
}

void MainWindow::on_toolButton_5_clicked()
{
	calculate_visibility_without_editing_tf();
	delay_add_transfer_function_component(get_tf_component0(), chartView_features[0]);
	delay_set_button_color_to_component_peak_color(*ui.toolButton, get_tf_component0(), get_tf_array());
}

void MainWindow::on_toolButton_6_clicked()
{
	calculate_visibility_without_editing_tf();
	delay_add_transfer_function_component(get_tf_component1(), chartView_features[1]);
	delay_set_button_color_to_component_peak_color(*ui.toolButton_2, get_tf_component1(), get_tf_array());
}

void MainWindow::on_toolButton_7_clicked()
{
	calculate_visibility_without_editing_tf();
	delay_add_transfer_function_component(get_tf_component2(), chartView_features[2]);
	delay_set_button_color_to_component_peak_color(*ui.toolButton_3, get_tf_component2(), get_tf_array());
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

void MainWindow::on_action_Weights_of_Transfer_function_componments_triggered()
{
	char str[_MAX_PATH];
	sprintf(str, "%g %g %g %g", tf_component_weights[0], tf_component_weights[1], tf_component_weights[2], tf_component_weights[3]);
	char label[_MAX_PATH];
	sprintf(label, "Enter transfer function component weights (%d numbers in range [0,1] separated by space)", D_MAX_TF_COMPONENTS);
	bool ok;
	QString text = QInputDialog::getText(this, tr("Transfer function component weights"),
		tr(label), QLineEdit::Normal, tr(str), &ok);
	if (ok && !text.isEmpty())
	{
		QTextStream s(&text);
		s >> tf_component_weights[0] >> tf_component_weights[1] >> tf_component_weights[2] >> tf_component_weights[3];
		for (int i = 0; i < D_MAX_TF_COMPONENTS; i++)
		{
			tf_component_weights[i] = tf_component_weights[i] < 0 ? 0 : (tf_component_weights[i] > 1 ? 1 : tf_component_weights[i]);
		}
		std::cout << "tf_component_weights " << tf_component_weights[0] << " " << tf_component_weights[1] << " " << tf_component_weights[2] << " " << tf_component_weights[3] << std::endl;
	}
}

void MainWindow::on_action_Smooth_transfer_functions_triggered()
{
	update_all_transfer_functions_and_histograms();
}
