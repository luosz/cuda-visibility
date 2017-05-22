// cpp.cpp : Defines the entry point for the console application.
//

#include <iostream>

template<typename T>
void merge_sort_recursive(T a[], T b[], int start, int end)
{
	if (start >= end)
		return;
	int len = end - start, mid = start + (len >> 1);
	int start1 = start, end1 = mid, start2 = mid + 1, end2 = end;
	merge_sort_recursive(a, b, start1, end1);
	merge_sort_recursive(a, b, start2, end2);
	int i = start;
	while (start1 <= end1 && start2 <= end2)
	{
		b[i++] = a[start2] < a[start1] ? a[start2++] : a[start1++];
	}
	while (start1 <= end1)
	{
		b[i++] = a[start1++];
	}
	while (start2 <= end2)
	{
		b[i++] = a[start2++];
	}
	for (i = start; i <= end; i++)
	{
		a[i] = b[i];
	}
}

template<typename T>
void merge_sort(T a[], const int N)
{
	auto b = new T[N];
	merge_sort_recursive<T>(a, b, 0, N - 1);
	delete[] b;
}

int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize * 2] = { 0 };
	int d[arraySize * 2] = { 0 };
	const int N = arraySize*2;

	int j = 0;
	for (int i = 0; i < arraySize; i++)
	{
		c[j++] = a[i];
		c[j++] = b[i];
	}

	std::cout << sizeof(d) << std::endl;

	std::cout << sizeof(int) << std::endl;

	memcpy(d, c, sizeof(d));

	for (int i=0;i<N;i++)
	{
		std::cout << d[i] << "\t";
	}
	std::cout << std::endl;

	merge_sort(d, N);

	for (int i = 0;i < N;i++)
	{
		std::cout << d[i] << "\t";
	}
	std::cout << std::endl;

    return 0;
}

