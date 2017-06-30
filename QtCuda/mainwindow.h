#pragma once

#include <string>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QLabel>
#include "ui_mainwindow.h"

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QWidget *parent = Q_NULLPTR);
	void SetText(std::string content)
	{
		ui.label->setText(QString::fromUtf8(content.data(), content.size()));
	}

private slots:
    void on_pushButton_clicked();

private:
	Ui::MainWindowClass ui;
};
