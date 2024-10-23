#include "HanCharRecog.h"

cv::Mat HanCharRecog::Preprocessing(std::string imagePath) {
	mInputImage = cv::imread(imagePath);

	cv::Mat blob;
	cv::resize(mInputImage, blob, cv::Size(64, 64));

	blob.convertTo(blob, CV_32FC3);

	blob = blob / (float)255;

	return blob;
}

cv::Mat HanCharRecog::Preprocessing(cv::Mat mat) {
	//cv::Mat blob = mat.clone();
	cv::Mat blob = mat;
	cv::resize(blob, blob, cv::Size(64, 64));

	blob.convertTo(blob, CV_32FC3);

	blob = blob / (float)255;

	return blob;
}

void HanCharRecog::Test(std::string imagePath) {
	//std::function<cv::Mat(HanCharRecog&, std::string)> funcPtr = &HanCharRecog::Preprocessing;
	//Inference(funcPtr, imagePath);
	
	Inference(Preprocessing(imagePath));

	//GetResult();
}

wchar_t HanCharRecog::GetArgMax() {
	int result[3] = { 0, };

	for (int i = 0; i < mOutputTensors.size(); i++) {
		int curOutputSize = mOutputTensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
		const float* data = mOutputTensors[i].GetTensorMutableData<float>();

		float sum = 0;
		//float max = -std::numeric_limits<float>::max(); // winapi.h ...
		float max = -FLT_MAX;

		int idx = 0;
		for (int j = 0; j < curOutputSize; j++) {
			sum += data[j];
			if (data[j] > max) {
				max = data[j];
				idx = j;
			}
		}
		result[i] = idx;
	}

	setlocale(LC_ALL, "korean");

	wchar_t korChar = 0xAC00 + 588 * result[0] + 28 * result[1] + result[2];
	//std::wcout << "1 : " << result[0] << ", 2 : " << result[1] << ", 3 : " << result[2] << std::endl;
	//wprintf(L"%c ", korChar);
	return korChar;
}

wchar_t HanCharRecog::GetResult(std::string imagePath) {
	Inference(Preprocessing(imagePath));

#ifdef IMSHOW
	cv::imshow("charImage", mInputImage);
	cv::waitKey(0);
	cv::destroyWindow("charImage");
#endif

	return GetArgMax();
}

wchar_t HanCharRecog::GetResult(cv::Mat crop) {
	Inference(Preprocessing(crop));

#ifdef IMSHOW
	cv::imshow("charImage", crop);
	cv::waitKey(0);
	cv::destroyWindow("charImage");
#endif

	return GetArgMax();
}

