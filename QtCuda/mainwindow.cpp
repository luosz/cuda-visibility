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
	auto p = get_tf_array();
	const qreal N = D_BIN_COUNT - 1;

	auto chart = chartView.chart();
	chart->removeAllSeries();
	chart->legend()->hide();
	for (int i = 0; i < D_BIN_COUNT; i++)
	{
		auto c = QColor::fromRgbF((qreal)p[i].x, (qreal)p[i].y, (qreal)p[i].z);
		auto line = new QLineSeries();
		line->append(i / N, (qreal)p[i].w);
		line->append(i / N, 0);
		line->setColor(c);
		chart->addSeries(line);
	}
	chart->createDefaultAxes();
	chart->setTitle("Transfer function");
	chartView.setRenderHint(QPainter::Antialiasing);

	auto p2 = get_relative_visibility_histogram();
	auto chart2 = chartView2.chart();
	chart2->removeAllSeries();
	chart2->legend()->hide();
	for (int i = 0; i < D_BIN_COUNT; i++)
	{
		auto c = QColor::fromRgbF(0.5, 0.5, 0.5);
		auto line = new QLineSeries();
		line->append(i / N, 0);
		line->append(i / N, (qreal)p2[i]);
		line->setColor(c);
		chart2->addSeries(line);
	}
	chart2->createDefaultAxes();
	chart2->setTitle("Smoothed relative visibility histogram");
	chartView2.setRenderHint(QPainter::Antialiasing);
}

void MainWindow::on_pushButton_3_clicked()
{
	apply_blending_operation();

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
