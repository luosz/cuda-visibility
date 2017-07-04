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
	for (int i = 0; i < D_BIN_COUNT; i++)
	{
		//std::cout << p[i].x << " " << p[i].y << " " << p[i].z << " " << p[i].w << std::endl;
		//series->append(i, (qreal)p[i].w);
	}

	QLineSeries *series0 = new QLineSeries();
	QLineSeries *series1 = new QLineSeries();

	*series0 << QPointF(1, 5) << QPointF(3, 7) << QPointF(7, 6) << QPointF(9, 7) << QPointF(12, 6)
		<< QPointF(16, 7) << QPointF(18, 5);
	*series1 << QPointF(1, 3) << QPointF(3, 4) << QPointF(7, 3) << QPointF(8, 2) << QPointF(12, 3)
		<< QPointF(16, 4) << QPointF(18, 3);

	QAreaSeries *series = new QAreaSeries(series0, series1);
	series->setName("Batman");
	QPen pen(0x059605);
	pen.setWidth(3);
	series->setPen(pen);

	QLinearGradient gradient(QPointF(0, 0), QPointF(0, 1));
	gradient.setColorAt(0.0, 0x3cc63c);
	gradient.setColorAt(0.5, 0x0000ff);
	gradient.setColorAt(1.0, 0x26f626);
	gradient.setCoordinateMode(QGradient::ObjectBoundingMode);
	series->setBrush(gradient);

	QChart *chart = new QChart();
	chart->addSeries(series);
	chart->setTitle("Simple areachart example");
	chart->createDefaultAxes();
	chart->axisX()->setRange(0, 20);
	chart->axisY()->setRange(0, 10);

	if (!chartView)
	{
		chartView = new QChartView(chart);
		chartView->setRenderHint(QPainter::Antialiasing);
		ui.gridLayout->addWidget(chartView);
	}else
	{
		chartView->setChart(chart);
	}
}
