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
	ui.graphicsView->setScene(&scene);
	ui.gridLayout->addWidget(&chartView);
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
	const qreal N = D_BIN_COUNT - 1;
	QVector<QScatterSeries*> points;
	QVector<QLineSeries*> lines;
	scene.clear();
	auto p = get_tf_array();
	for (int i = 0; i < D_BIN_COUNT; i++)
	{
		auto c = QColor::fromRgbF((qreal)p[i].x, (qreal)p[i].y, (qreal)p[i].z);
		if (p[i].w>0.5f/D_BIN_COUNT)
		{
			scene.addLine(i, N, i, N-(qreal)p[i].w*N, QPen(c));
		}

		if (i < N)
		{
			auto s2 = new QLineSeries();
			s2->append(i / N, (qreal)p[i].w);
			s2->append((i + 1) / N, (qreal)p[i + 1].w);
			s2->setColor(c);
			lines.append(s2);
		}else
		{
			auto series = new QScatterSeries();
			series->append(i / N, (qreal)p[i].w);
			series->setColor(c);
			series->setMarkerSize(4);
			points.append(series);
		}
	}
	ui.graphicsView->fitInView(0, 0, 255, 255, Qt::KeepAspectRatio);

	QChart *chart = new QChart();
	chart->legend()->hide();
	for (int i = 0; i < lines.size(); i++)
	{
		chart->addSeries(lines[i]);
	}
	for (int i = 0; i < points.size(); i++)
	{
		chart->addSeries(points[i]);
	}
	chart->createDefaultAxes();
	chart->setTitle("Transfer function");

	chartView.setChart(chart);
	chartView.setRenderHint(QPainter::Antialiasing);
}

void MainWindow::on_pushButton_3_clicked()
{
	const int n = 11;
	int r = 2;
	float d[n];
	calc_gaussian_kernel(d, 2 * r + 1, sigma(r));
	for (int i = 0; i < 2 * r + 1; i++)
	{
		std::cout << d[i] << std::ends;
	}
	std::cout << std::endl;
	r = 4;
	calc_gaussian_kernel(d, 2 * r + 1, sigma(r));
	for (int i = 0; i < 2 * r + 1; i++)
	{
		std::cout << d[i] << std::ends;
	}
	std::cout << std::endl;
}

void MainWindow::on_pushButton_4_clicked()
{
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
