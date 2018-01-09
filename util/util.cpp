#include <arrayfire.h>
#include <iostream>
#include <cstdio>

using namespace af;

int main(int argc, char *argv[])
{
	af::info();
	af::Window wnd("Image Demo");
	std::cout << ASSETS_DIR << std::endl;
	// load images
	array img = loadImage("../QtCuda/~screenshot_0.ppm", true);
	array img1 = loadImage("../QtCuda/~screenshot_1.ppm", true);
	array img2 = loadImage("../QtCuda/~screenshot_2.ppm", true);
	array img3 = loadImage("../QtCuda/~screenshot_3.ppm", true);

	std::cout << img.dims(0) << "\t" << img.dims(1) << std::endl;
	print("img / 255", (img / 255).rows(500, 506).cols(500, 504));
	print("img / 255.f", (img / 255.f).rows(500, 506).cols(500, 504));

	while (!wnd.close())
	{
		wnd.grid(1, 4);
		wnd(0, 0).image(img / 255, "~screenshot_0.ppm");
		wnd(0, 1).image(img1 / 255, "~screenshot_1.ppm");
		wnd(0, 2).image(img2 / 255, "~screenshot_2.ppm");
		wnd(0, 3).image(img3 / 255, "~screenshot_3.ppm");
		wnd.show();
	}

	return 0;
}
