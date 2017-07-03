#pragma once

#include <string>
#include <iostream>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QLabel>
#include <QtWidgets/QCheckBox>
#include <QApplication>
#include <QDesktopWidget>
#include <QColorDialog>
#include <QTextStream>
#include "ui_mainwindow.h"
#include "def.h"

typedef float(*Pointer)[4];

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QWidget *parent = Q_NULLPTR);
	void SetText(std::string content)
	{
		ui.label->setText(QString::fromUtf8(content.data(), content.size()));
	}

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
		sample_palette.setColor(QPalette::Window, color);
		sample_palette.setColor(QPalette::WindowText, c2);
		ui.label->setPalette(sample_palette);
		ui.label->setText(color.name());
	}

	void set_color_pointer(Pointer p, bool *alpha, bool *color)
	{
		color_array = p;
		apply_alpha = alpha;
		apply_color = color;
	}

private slots:
    void on_pushButton_clicked();

    void on_checkBox_clicked();

    void on_checkBox_2_clicked();

private:
	Ui::MainWindowClass ui;
	QColor color;
	Pointer color_array = NULL;
	bool *apply_alpha = NULL;
	bool *apply_color = NULL;
};
