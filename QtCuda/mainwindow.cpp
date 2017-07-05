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
	QLineSeries *series = new QLineSeries();

	scene.clear();
	auto p = get_tf_array();
	const qreal N = D_BIN_COUNT - 1;
	for (int i = 0; i < D_BIN_COUNT; i++)
	{
		if (p[i].w>1.f/D_BIN_COUNT)
		{
			scene.addLine(i, N, i, N-(qreal)p[i].w*N, QPen(QColor::fromRgbF((qreal)p[i].x, (qreal)p[i].y, (qreal)p[i].z)));
		}
		series->append(i/N, (qreal)p[i].w);
	}
	ui.graphicsView->fitInView(0, 0, 255, 255, Qt::KeepAspectRatio);

	QChart *chart = new QChart();
	chart->legend()->hide();
	chart->addSeries(series);
	chart->createDefaultAxes();
	chart->setTitle("Transfer function");

	chartView.setChart(chart);
	chartView.setRenderHint(QPainter::Antialiasing);
}
