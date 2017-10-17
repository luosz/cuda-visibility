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
	this->move(this->pos() - QPoint(179, 260));
	ui.setupUi(this);
	update_color(QColor::fromRgbF(D_RGBA[0], D_RGBA[1], D_RGBA[2], D_RGBA[3]));
	//ui.graphicsView->setScene(&scene);
	ui.verticalLayout->addWidget(&chartView);
	ui.verticalLayout->addWidget(&chartView2);
	QTimer::singleShot(2000, this, SLOT(show_transfer_function()));
}

void MainWindow::on_pushButton_clicked()
{
	auto c = QColorDialog::getColor(color);
	if (c.isValid())
	{
		update_color(c);
	}
}

void MainWindow::on_checkBox_clicked()
{
	if (apply_alpha && apply_color)
	{
		*apply_alpha = ui.checkBox->isChecked();
		std::cout << "Apply alpha: " << (*apply_alpha ? "true" : "false") << "\t Apply color: " << (*apply_color ? "true" : "false") << std::endl;
	}
}

void MainWindow::on_checkBox_2_clicked()
{
	if (apply_alpha && apply_color)
	{
		*apply_color = ui.checkBox_2->isChecked();
		std::cout << "Apply alpha: " << (*apply_alpha ? "true" : "false") << "\t Apply color: " << (*apply_color ? "true" : "false") << std::endl;
	}
}

void MainWindow::on_pushButton_2_clicked()
{
	show_transfer_function();
}

void MainWindow::on_pushButton_3_clicked()
{
	apply_tf_editing();
	QTimer::singleShot(500, this, SLOT(show_transfer_function()));

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
	QTimer::singleShot(500, this, SLOT(show_transfer_function()));

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
        std::cout << "apply_time_varying_tf=" << (*apply_time_varying_tf_editing ? "true" : "false") << std::endl;
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
		tr("Copyright (c) 2017 The Trinity Centre for Creative Technologies & Media Engineering, Trinity College Dublin, The University of Dublin.\n" "All rights reserved."), QMessageBox::Ok);
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
		show_transfer_function();
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
	}
}
