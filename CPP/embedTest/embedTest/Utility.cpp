#include "Utility.h"

HMODULE GCM() {
	HMODULE hModule = NULL;
	GetModuleHandleEx(
		GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
		(LPCTSTR)GCM,
		&hModule
	);
	
	return hModule;
}

std::pair<std::vector<UCHAR>, DWORD> GetEmbedData(int first, int second) {
	HRSRC hRes = FindResource(GCM(), MAKEINTRESOURCE(first), MAKEINTRESOURCE(second));
	HGLOBAL hData = LoadResource(GCM(), hRes);
	DWORD hSize = SizeofResource(GCM(), hRes);

	//char* hFinal = (char*)LockResource(hData);
	UCHAR* hFinal = reinterpret_cast<UCHAR*>(LockResource(hData));
	std::vector<UCHAR> weightBuffer;


	weightBuffer.assign(hFinal, hFinal + hSize);

	std::pair<std::vector<UCHAR>, DWORD> outs;
	outs.first = std::move(weightBuffer);
	outs.second = hSize;

	return outs;
}

std::string GetTimeString() {
	auto now = std::chrono::system_clock::now();

	std::time_t time = std::chrono::system_clock::to_time_t(now);
	std::tm local_time = *std::localtime(&time);

	char date[80] = { 0 };
	std::strftime(date, sizeof(date), "%Y_%m_%d_%I_%M", &local_time);

	return std::string(date);
}


void PaddingTEST(std::string path) {
	cv::Mat img = cv::imread(path);

	cv::Mat show = ImagePadding(img, 720);
	//cv::Mat show = ResizeKeepingRatio(img, 720);

	cv::imshow("Show", show);
	cv::waitKey(0);
}

cv::Mat ImagePadding(const cv::Mat& img, int size) {
	cv::Mat resized = ResizeKeepingRatio(img, size);

	int width = resized.cols;
	int height = resized.rows;

	int pivot = std::min(width, height);
	float rate = std::max(width, height) / (float)(std::min(width, height));

	//cv::Mat rtnMat(size, size, CV_8UC3);
	cv::Mat rtnMat = cv::Mat::zeros(size, size, CV_8UC3);

	if (width >= height) {
		memcpy(rtnMat.data, resized.data, resized.total() * resized.elemSize());
		//rtnMat = std::move(cv::Mat());
	}
	else {
		for (int i = 0; i < size; i++) {
			int cpSize = resized.elemSize() * width;
			int next = size * resized.elemSize() * i;
			int orgNext = cpSize * i;
			memcpy(rtnMat.data + next, resized.data + orgNext, cpSize);
		}
	}

	return rtnMat;
}

cv::Mat ResizeKeepingRatio(const cv::Mat& src, int pivot) {
	cv::Mat dst;
	float ratio = MIN(src.cols, (float)src.rows) / MAX(src.cols, (float)src.rows);

	cv::Size sz;
	int direction = src.cols > src.rows;
	sz = direction ? cv::Size(pivot, pivot * ratio) : cv::Size(pivot * ratio, pivot);

	cv::resize(src, dst, sz);
	return dst;
}

std::vector<std::string> GetImageFileFromDir(std::string path) {
	std::vector<std::string> names;

	std::filesystem::directory_iterator iter(path);

	for (const auto& file : iter)
	{
		std::string ext = file.path().extension().string();
		std::transform(ext.begin(), ext.end(), ext.begin(), [](auto chr) { return std::tolower(chr); });

		if (ext == ".jpg" || ext == ".jpeg" || ext == ".tif") {
			names.push_back(file.path().string());
		}
	}

	return names;
}

std::vector<std::string> GetImageFilesFromPath(std::string path)
{
	std::vector<std::string> fileVec;

	if (std::filesystem::is_directory(path)) {
		fileVec = GetImageFileFromDir(path);
	}
	else if (std::filesystem::is_regular_file(path)) {
		fileVec.push_back(path);
	}

	return fileVec;
}